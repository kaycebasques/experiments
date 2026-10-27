from concurrent.futures import ThreadPoolExecutor
from json import dump, load
from os import environ
from pathlib import Path
from sys import argv

import requests
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForMultimodalLM, AutoTokenizer

DEFAULT_THREAD_ID = 21941278


def fetch_raw(thread_id: int = DEFAULT_THREAD_ID):
    base_url = 'https://hacker-news.firebaseio.com/v0/item'
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20)
    session.mount('https://', adapter)

    def fetch_item(item_id):
        response = session.get(f'{base_url}/{item_id}.json', timeout=10)
        response.raise_for_status()
        return response.json()

    story = fetch_item(thread_id)
    kid_ids = story.get('kids', [])
    with ThreadPoolExecutor(max_workers=20) as executor:
        comments = list(executor.map(fetch_item, kid_ids))

    bwd = environ['BUILD_WORKSPACE_DIRECTORY']
    json_filename = f'{bwd}/{thread_id}_raw.json'
    with open(json_filename, 'w', encoding='utf-8') as f:
        dump(comments, f, indent=2)


class EmbeddingModel(SentenceTransformer):

    def __init__(
        self,
        model_id: str = 'google/embeddinggemma-2',
        text_only: bool = True,
        truncate_dim: int | None = None,
        max_seq_length: int = 8192,
    ):
        config_kwargs = (
            {'vision_config': None, 'audio_config': None} if text_only else {}
        )
        super().__init__(
            model_id,
            config_kwargs=config_kwargs,
            truncate_dim=truncate_dim,
        )
        self.model_id = model_id
        self.text_only = text_only
        self.max_seq_length = max_seq_length
        self.dimension = self.get_embedding_dimension()


def record_predictions(predictions: list[str]) -> None:
    """Record the verbatim predictions extracted from the comment.

    Args:
        predictions: A list of strings where each string is one verbatim prediction from the comment.
    """


class LLM:

    def __init__(self, model_id: str = 'google/gemma-4-E2B-it'):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_id,
            dtype='auto',
            vision_config=None,
            audio_config=None,
        )

    def extract_predictions(self, text: str) -> list[str]:
        messages = [
            {
                'role': 'system',
                'content': (
                    'Extract every distinct prediction from the user comment '
                    'and pass them to the `record_predictions` tool. Each '
                    'string in `predictions` must represent ONE complete '
                    'prediction copied verbatim from the comment (strip only '
                    'leading list markers like "1.", "-", or "*" and HTML '
                    'tags like "<p>"). When a numbered item, bullet point, or '
                    'paragraph uses multiple sentences to describe or '
                    'elaborate on the same prediction, keep all of those '
                    'sentences together in a single string. Exclude '
                    'non-prediction content such as introductory framing, '
                    'category headers, or closing remarks.'
                ),
            },
            {
                'role': 'user',
                'content': (
                    'My predictions:<p>'
                    '1. Electric cars become the norm. All cars are fully '
                    'autonomous on highways.<p>'
                    '2. VR will take off big time. Concerns about VR porn '
                    'being too realistic will be raised.<p>'
                    'Other thoughts:<p>'
                    '1. Cargo pants will become popular again.<p>'
                    'Good luck everyone!'
                ),
            },
            {
                'role': 'assistant',
                'tool_calls': [
                    {
                        'type': 'function',
                        'function': {
                            'name': 'record_predictions',
                            'arguments': {
                                'predictions': [
                                    'Electric cars become the norm. All cars are fully autonomous on highways.',
                                    'VR will take off big time. Concerns about VR porn being too realistic will be raised.',
                                    'Cargo pants will become popular again.',
                                ]
                            },
                        },
                    }
                ],
                'tool_responses': [
                    {
                        'name': 'record_predictions',
                        'response': {'status': 'ok'},
                    }
                ],
                'content': 'Done.',
            },
            {'role': 'user', 'content': text},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tools=[record_predictions],
            tokenize=True,
            return_dict=True,
            return_tensors='pt',
            add_generation_prompt=True,
            enable_thinking=False,
        ).to(self.model.device)
        input_len = inputs['input_ids'].shape[-1]
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,
        )
        try:
            parsed = self.tokenizer.parse_response(
                outputs[0][input_len:],
                prefix=inputs['input_ids'][0],
                tools=[record_predictions],
            )
        except ValueError:
            return []
        tool_calls = parsed.get('tool_calls') or []
        if not tool_calls:
            return []
        return (
            tool_calls[0].get('function', {}).get('arguments', {}).get('predictions')
            or []
        )


class Prediction:

    _records: dict[Path, list[dict]] = {}

    def __init__(
        self,
        text: str,
        id: int,
        model: EmbeddingModel,
        thread_id: int = DEFAULT_THREAD_ID,
    ):
        self.text = text
        self.id = id
        self.model = model
        self.thread_id = thread_id
        bwd = environ['BUILD_WORKSPACE_DIRECTORY']
        self.path = Path(f'{bwd}/{self.thread_id}_predictions.json')
        record = self._find()
        if record is not None:
            self.clustering = record['clustering']
            self.similarity = record['similarity']
        else:
            self.clustering = self._embed('Clustering')
            self.similarity = self._embed('SentenceSimilarity')
            self._save()

    def _load(self) -> list[dict]:
        if self.path not in Prediction._records:
            if not self.path.exists():
                Prediction._records[self.path] = []
            else:
                with open(self.path, encoding='utf-8') as f:
                    Prediction._records[self.path] = load(f)
        return Prediction._records[self.path]

    def _find(self) -> dict | None:
        for record in self._load():
            if record['id'] == self.id and record['text'] == self.text:
                return record
        return None

    def _embed(self, prompt_name: str) -> list[float]:
        return self.model.encode(
            self.text,
            prompt_name=prompt_name,
            normalize_embeddings=True,
        ).tolist()

    def _save(self) -> None:
        records = self._load()
        records.append({
            'id': self.id,
            'text': self.text,
            'clustering': self.clustering,
            'similarity': self.similarity,
        })
        tmp_path = self.path.with_suffix('.json.tmp')
        with open(tmp_path, 'w', encoding='utf-8') as f:
            dump(records, f)
        tmp_path.replace(self.path)


class Comment:

    def __init__(self, id, text, author):
        self.id = id
        self.text = text
        self.author = author


class Thread:

    def __init__(self, id: int = DEFAULT_THREAD_ID):
        self.id = id
        bwd = environ['BUILD_WORKSPACE_DIRECTORY']
        raw_path = Path(f'{bwd}/{self.id}_raw.json')
        if not raw_path.exists():
            fetch_raw(self.id)
        with open(raw_path, encoding='utf-8') as f:
            raw_data = load(f)
        self.comments = [
            Comment(item['id'], item['text'], item['by'])
            for item in raw_data
            if item
            and not item.get('deleted')
            and 'text' in item
            and 'by' in item
        ]


thread_id = DEFAULT_THREAD_ID
for arg in argv[1:]:
    if arg.startswith('--id='):
        thread_id = int(arg.split('=', 1)[1])
    else:
        raise ValueError(f'Invalid argument: {arg} (use --id=<id>)')

thread = Thread(id=thread_id)
llm = LLM()
embedding_model = EmbeddingModel()
bwd = environ['BUILD_WORKSPACE_DIRECTORY']
predictions_path = Path(f'{bwd}/{thread_id}_predictions.json')
existing_by_id: dict[int, list[str]] = {}
if predictions_path.exists():
    with open(predictions_path, encoding='utf-8') as f:
        for record in load(f):
            existing_by_id.setdefault(record['id'], []).append(record['text'])

total_comments = len(thread.comments)
total_predictions = 0
for i, comment in enumerate(thread.comments, start=1):
    texts = (
        existing_by_id[comment.id]
        if comment.id in existing_by_id
        else llm.extract_predictions(comment.text)
    )
    predictions = [
        Prediction(
            text=p,
            id=comment.id,
            model=embedding_model,
            thread_id=thread_id,
        )
        for p in texts
    ]
    total_predictions += len(predictions)
    print(
        f'[{i}/{total_comments}] Comment {comment.id} ({comment.author}): '
        f'{len(predictions)} predictions (total: {total_predictions})',
        flush=True,
    )
