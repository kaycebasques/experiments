from urllib.parse import urljoin

from bs4 import BeautifulSoup
import requests
from transformers import AutoModelForMultimodalLM, AutoTokenizer

USER_AGENT = (
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
)


def record_decision(actionable: bool, summary: str) -> None:
    """Record whether the story might be actionable in terms of an investment and a summary of why.

    Args:
        actionable: True if the story might be actionable in terms of an investment, False otherwise.
        summary: Assessment of why and how the story is actionable in terms of an investment (empty string if not actionable).
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

    def evaluate(
        self, title: str, content: str | None = None
    ) -> tuple[bool, str]:
        if content is None:
            messages = [
                {
                    'role': 'system',
                    'content': (
                        'Determine whether the Hacker News story title might '
                        'be actionable in terms of an investment (e.g. '
                        'company news, product launches, regulatory actions, '
                        'industry shifts, market trends, or financial events) '
                        'and pass the result to the `record_decision` tool. '
                        'If `actionable` is True, `summary` should contain '
                        'your initial assessment of why it is actionable for '
                        'an individual investor based in the US with a global '
                        'Schwab account. If `actionable` is False, set '
                        '`summary` to an empty string.'
                    ),
                },
                {
                    'role': 'user',
                    'content': (
                        'Title: TSMC announces 2nm chip production ahead of '
                        'schedule'
                    ),
                },
                {
                    'role': 'assistant',
                    'tool_calls': [
                        {
                            'type': 'function',
                            'function': {
                                'name': 'record_decision',
                                'arguments': {
                                    'actionable': True,
                                    'summary': (
                                        'Early 2nm chip production is a '
                                        'positive catalyst for TSMC and '
                                        'semiconductor equipment suppliers '
                                        'tradable via a US or global Schwab '
                                        'account.'
                                    ),
                                },
                            },
                        }
                    ],
                    'tool_responses': [
                        {
                            'name': 'record_decision',
                            'response': {'status': 'ok'},
                        }
                    ],
                    'content': 'Done.',
                },
                {
                    'role': 'user',
                    'content': (
                        'Title: Why is functional programming so hard to learn?'
                    ),
                },
                {
                    'role': 'assistant',
                    'tool_calls': [
                        {
                            'type': 'function',
                            'function': {
                                'name': 'record_decision',
                                'arguments': {
                                    'actionable': False,
                                    'summary': '',
                                },
                            },
                        }
                    ],
                    'tool_responses': [
                        {
                            'name': 'record_decision',
                            'response': {'status': 'ok'},
                        }
                    ],
                    'content': 'Done.',
                },
                {'role': 'user', 'content': f'Title: {title}'},
            ]
        else:
            messages = [
                {
                    'role': 'system',
                    'content': (
                        'Analyze the Hacker News story title and the full '
                        'article contents, then pass `actionable` (bool) and '
                        '`summary` (str) to the `record_decision` tool. '
                        '`summary` must explain how the story is actionable '
                        'for an individual investor based in the US who has a '
                        'global Charles Schwab account, based on your '
                        'analysis of the full article contents.'
                    ),
                },
                {
                    'role': 'user',
                    'content': (
                        'Title: TSMC announces 2nm chip production ahead of '
                        'schedule\n\n'
                        'Article:\n'
                        'Taiwan Semiconductor Manufacturing Co. reported today '
                        'that trial yields at its Kaohsiung 2nm facility have '
                        'surpassed 70%, pulling volume production forward by '
                        'two quarters for major customers including Apple and '
                        'Nvidia. Capital expenditure guidance for EUV tools '
                        'supplied by ASML and Tokyo Electron was raised by 8%.'
                    ),
                },
                {
                    'role': 'assistant',
                    'tool_calls': [
                        {
                            'type': 'function',
                            'function': {
                                'name': 'record_decision',
                                'arguments': {
                                    'actionable': True,
                                    'summary': (
                                        'The article confirms 70%+ 2nm trial '
                                        'yields at Kaohsiung and an 8% capex '
                                        'increase, directly benefiting TSMC '
                                        '(NYSE: TSM), fabless customers Apple '
                                        '(AAPL) and Nvidia (NVDA), and EUV '
                                        'suppliers ASML (ASML) and Tokyo '
                                        'Electron (8035.T on the Tokyo Stock '
                                        'Exchange via a Schwab Global '
                                        'account).'
                                    ),
                                },
                            },
                        }
                    ],
                    'tool_responses': [
                        {
                            'name': 'record_decision',
                            'response': {'status': 'ok'},
                        }
                    ],
                    'content': 'Done.',
                },
                {
                    'role': 'user',
                    'content': f'Title: {title}\n\nArticle:\n{content}',
                },
            ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tools=[record_decision],
            tokenize=True,
            return_dict=True,
            return_tensors='pt',
            add_generation_prompt=True,
            enable_thinking=False,
        ).to(self.model.device)
        input_len = inputs['input_ids'].shape[-1]
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )
        parsed = self.tokenizer.parse_response(
            outputs[0][input_len:],
            prefix=inputs['input_ids'][0],
            tools=[record_decision],
        )
        args = parsed['tool_calls'][0]['function']['arguments']
        return bool(args['actionable']), str(args['summary'])


class Submission:

    def __init__(self, title: str, url: str, llm: LLM):
        self.title = title
        self.url = urljoin('https://news.ycombinator.com/', url)
        self.actionable, self.summary = llm.evaluate(self.title)
        if self.actionable:
            content = self._fetch_content()
            if content:
                self.actionable, self.summary = llm.evaluate(
                    self.title, content
                )

    def _fetch_content(self) -> str | None:
        try:
            response = requests.get(
                self.url,
                headers={'User-Agent': USER_AGENT},
                timeout=10,
            )
            response.raise_for_status()
        except requests.RequestException:
            return None
        content_type = response.headers.get('Content-Type', '')
        if 'text/html' not in content_type:
            return None
        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup(
            ['script', 'style', 'noscript', 'svg', 'header', 'footer', 'nav']
        ):
            tag.decompose()
        root = soup.find('article') or soup.find('main') or soup.body or soup
        text = root.get_text(separator='\n', strip=True)
        return text[:8000] if text else None


class Day:

    def __init__(self, date: str = '2024-03-01', llm: LLM | None = None):
        self.date = date
        self.llm = llm if llm is not None else LLM()
        self.url = f'https://news.ycombinator.com/front?day={self.date}'
        response = requests.get(self.url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        self.submissions = [
            Submission(a.text, a['href'], self.llm)
            for a in soup.select('.submission .titleline > a')
        ]


day = Day()
for submission in day.submissions:
    if submission.actionable:
        print(submission.title)
        print(submission.url)
        print(submission.summary)
        print()
