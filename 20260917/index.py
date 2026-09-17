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


class Issue:

    def __init__(self, number, title, body):
        self.number = number
        self.title = title
        self.body = body
        self._images()
        self._embedding()

    def _parse(self):
        urls = []
        html = findall(
            r'<img\s+[^>]*?src=["\']([^"\']+)["\']', self.body, IGNORECASE
        )
        urls.extend(html)
        md = findall(
            r'!\[[^\]]*\]\(([^)\s]+)(?:\s+["\'][^"\']*["\'])?\)', self.body
        )
        urls.extend(md)
        seen = set()
        return [u for u in urls if not (u in seen or seen.add(u))]

    def _download(self, url):
        res = _req(url)
        return res.content, res.headers

    def _images(self):
        urls = self._parse()
        for url in urls:
            if not url.startswith('https://'):
                continue
            if url.endswith('.svg'):
                continue
            if 'shields.io' in url:
                continue
            if 'badge' in url:
                continue
            print(url)
            content, headers = self._download(url)
            print(headers.get('Content-Type'))

    def _embedding(self):
        pass


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
