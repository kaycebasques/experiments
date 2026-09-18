from json import dumps
from os import environ
from pathlib import Path
from re import IGNORECASE, findall
from time import sleep

from chonkie import TokenChunker
from dotenv import load_dotenv
from google import genai
from google.genai import types
from puremagic import from_string
from requests import get
import polars as pl


def _headers():
    token = environ.get('GITHUB_TOKEN')
    return {
        'Accept': 'application/vnd.github.raw+json',
        'User-Agent': 'experiments/20260917',
        'Authorization': f'Bearer {token}',
    }


def _req(url):
    rate_limit = (403, 429)
    while True:
        res = get(url, headers=_headers(), allow_redirects=True, timeout=15)
        if res.status_code in rate_limit:
            sleep(10)
            continue
        return res


def _parse(chunk):
    urls = []
    html = findall(
        r'<img\s+[^>]*?src=["\']([^"\']+)["\']', chunk, IGNORECASE
    )
    urls.extend(html)
    md = findall(
        r'!\[[^\]]*\]\(([^)\s]+)(?:\s+["\'][^"\']*["\'])?\)', chunk
    )
    urls.extend(md)
    seen = set()
    return [u for u in urls if not (u in seen or seen.add(u))]


def _download(url):
    res = _req(url)
    image = res.content
    mime = from_string(image, mime=True)
    return {'bytes': image, 'mime': mime}


def _images(chunk):
    urls = _parse(chunk)
    images = []
    for url in urls:
        if not url.startswith('https://'):
            continue
        if url.endswith('.svg'):
            continue
        if 'shields.io' in url:
            continue
        if 'badge' in url:
            continue
        images.append(_download(url))
    return images


class Issue:

    def __init__(self, number, title, body, cache=None, on_embed=None):
        self.number = number
        self.title = title
        self.body = body
        self.cache = cache if cache is not None else {}
        self.on_embed = on_embed
        self.embeddings = []
        self._embed()

    def _embed(self):
        max_tokens = 8192
        tokens_per_image = 258
        image_count = len(_parse(self.body))
        image_tokens = image_count * tokens_per_image
        # 0 images = 8192 tokens per chunk
        # 1 image  = 8192 - (1 * 258) tokens per chunk
        # 2 images = 8192 - (2 * 258) tokens per chunk
        # ideally you only reduce the chunk size when a given
        # chunk actually has image(s) in it but that seems
        # like it will require some very complex logic.
        chunk_size = max_tokens - image_tokens
        chunker = TokenChunker(chunk_size=chunk_size)
        chunks = chunker.chunk(self.body)
        gemini = None
        for chunk in chunks:
            if chunk.text in self.cache:
                embedding = self.cache[chunk.text]
            else:
                if gemini is None:
                    gemini = genai.Client(api_key=environ.get('GEMINI_API_KEY'))
                images = [i for i in _images(chunk.text) if i['mime'] != 'application/xml']
                query = dumps({'title': self.title, 'body': chunk.text})
                contents = []
                if len(images) == 0:
                    # text-only input should use task type
                    contents.append(f'task: classification | query: {query}')
                else:
                    # multimodal input should not use task type
                    contents.append(query)
                    for i in images:
                        print(i['mime'])
                        part = types.Part.from_bytes(data=i['bytes'], mime_type=i['mime'])
                        contents.append(part)
                res = gemini.models.embed_content(
                    model='gemini-embedding-2',
                    contents=contents,
                )
                embedding = res.embeddings[0].values
                print(embedding[0:10])
                self.cache[chunk.text] = embedding
                if self.on_embed:
                    self.on_embed(self.number, self.title, chunk.text, embedding)

            self.embeddings.append({
                'issue_number': self.number,
                'title': self.title,
                'chunk_text': chunk.text,
                'embedding': embedding,
            })


class Repo:

    def __init__(self, output_path='embeddings.parquet'):
        self.owner = environ.get('GITHUB_OWNER')
        self.repo = environ.get('GITHUB_REPO')
        if 'BUILD_WORKING_DIRECTORY' in environ and not Path(output_path).is_absolute():
            self.output_path = str(Path(environ['BUILD_WORKING_DIRECTORY']) / output_path)
        else:
            self.output_path = output_path
        self.records = []
        self.cache = {}
        self._load_cache()
        self.issues = []
        self._issues()

    def _load_cache(self):
        if Path(self.output_path).exists():
            df = pl.read_parquet(self.output_path)
            self.records = df.to_dicts()
            for row in self.records:
                self.cache[row['chunk_text']] = row['embedding']

    def save_embedding(self, issue_number, title, chunk_text, embedding):
        self.records.append({
            'issue_number': issue_number,
            'title': title,
            'chunk_text': chunk_text,
            'embedding': embedding,
        })
        self.save_parquet()

    def to_dataframe(self):
        if not self.records:
            return pl.DataFrame()
        dim = len(self.records[0]['embedding'])
        schema = {
            'issue_number': pl.Int64,
            'title': pl.String,
            'chunk_text': pl.String,
            'embedding': pl.Array(pl.Float32, shape=dim),
        }
        return pl.DataFrame(self.records, schema=schema)

    def save_parquet(self, path=None):
        target = path or self.output_path
        df = self.to_dataframe()
        if not df.is_empty():
            df.write_parquet(target)
            print(f'Saved {len(df)} embeddings to {target}')
        return df

    def _issues(self):
        page = 1
        per_page = 100
        while True:
            url = (
                f'https://api.github.com/repos/{self.owner}/{self.repo}/issues'
                f'?state=open&per_page={per_page}&page={page}'
            )
            res = _req(url)
            data = res.json()
            issues = [item for item in data if 'pull_request' not in item]
            for i in issues:
                number = i['number']
                title = i['title']
                body = i['body']
                issue = Issue(
                    number,
                    title,
                    body,
                    cache=self.cache,
                    on_embed=self.save_embedding,
                )
                self.issues.append(issue)
            if len(data) < per_page:
                break
            page += 1


load_dotenv()
repo = Repo()
