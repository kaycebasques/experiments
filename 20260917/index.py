from os import environ
from pathlib import Path
from re import IGNORECASE, findall
from time import sleep

from chonkie import TokenChunker
from dotenv import load_dotenv
from google import genai
from google.genai import types
from requests import get


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
    mime_type = res.headers.get('Content-Type')
    if url.endswith('.jpg') or url.endswith('.jpeg'):
        mime_type = 'image/jpeg'
    elif url.endswith('.gif'):
        mime_type = 'image/gif'
    elif url.endswith('.png'):
        mime_type = 'image/png'
    elif url.endswith('.webp'):
        mime_type = 'image/webp'
    return (res.content, mime_type)


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

    def __init__(self, number, title, body):
        self.number = number
        self.title = title
        self.body = body
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
        gemini = genai.Client(api_key=environ.get('GEMINI_API_KEY'))
        for chunk in chunks:
            images = _images(chunk.text)
            query = {'title': self.title, 'body': chunk.text}
            contents = []
            if len(images) == 0:
                # text-only input should use task type
                contents.append(f'task: classification | query: {query}')
            else:
                # multimodal input should not use task type
                contents.append(query)
                for image in images:
                    contents.append(types.Part.from_bytes(data=image[0], mime_type=image[1]))
            res = gemini.models.embed_content(
                model='gemini-embedding-2',
                contents=contents,
            )
            embedding = res.embeddings[0].values
            print(embedding[0:10])


        #     # Format content with task instruction
        #     raw_text = chunk_text if chunk_text.strip() else " "
        #     formatted_text = prepare_query_and_document(raw_text)
        #     contents: list[Any] = [formatted_text]
        #     # Parse and attach embedded images
        #     image_urls = extract_image_urls(chunk_text)
        #     if image_urls:
        #         for img_url in image_urls:
        #             image_data = download_image(img_url)
        #             if image_data:
        #                 img_bytes, mime_type, _ = image_data
        #                 img_counter += 1
        #                 part = types.Part.from_bytes(
        #                     data=img_bytes,
        #                     mime_type=mime_type,
        #                 )
        #                 contents.append(part)
        #                 print(f"Attached image part from {img_url}")
        #     # Generate Gemini embedding with text-only fallback if image
        #     # embedding is rejected
        #     try:
        #         embed_res = genai_client.models.embed_content(
        #             model=EMBEDDING_MODEL,
        #             contents=contents,
        #         )
        #     except Exception as e:  # pylint: disable=broad-exception-caught
        #         if len(contents) > 1:
        #             print(
        #                 "Warning: Multimodal embedding failed for"
        #                 f" {file_prefix} ({e}). Retrying with text-only"
        #                 " content..."
        #             )
        #             embed_res = genai_client.models.embed_content(
        #                 model=EMBEDDING_MODEL,
        #                 contents=[formatted_text],
        #             )
        #         else:
        #             raise e
        #     embedding_values = embed_res.embeddings[0].values
        #     print(
        #         f"  Generated embedding for {label} chunk {file_prefix}"
        #         f" ({len(embedding_values)} dims, hash {chunk_hash[:8]}...)"
        #     )



class Repo:

    def __init__(self):
        self.owner = environ.get('GITHUB_OWNER')
        self.repo = environ.get('GITHUB_REPO')
        self.issues = []
        self._issues()

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
                issue = Issue(number, title, body)
                self.issues.append(issue)
            if len(data) < per_page:
                break
            page += 1


load_dotenv()
repo = Repo()
