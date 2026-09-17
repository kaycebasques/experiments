from os import environ
from pathlib import Path
from time import sleep

from chonkie import TokenChunker
from dotenv import load_dotenv
from google import genai
from google.genai import types
from requests import get


def _headers(token):
    return {
        'Accept': 'application/vnd.github.raw+json',
        'User-Agent': 'experiments/20260917',
        'Authorization': f'Bearer {token}',
    }


def _req(url, token):
    rate_limit_status_codes = (403, 429)
    while True:
        res = get(url, headers=_headers(token), allow_redirects=True, timeout=15)
        if res.status_code in rate_limit_status_codes:
            sleep(60)
            continue
        return res


class Env:

    def __init__(self):
        path = Path(environ.get('BUILD_WORKSPACE_DIRECTORY')) / '.env'
        load_dotenv(path)
        self.gemini = environ.get('GEMINI_API_KEY')
        self.github = environ.get('GITHUB_TOKEN')
        self.owner = environ.get('GITHUB_OWNER')
        self.repo = environ.get('GITHUB_REPO')


class Issue:

    def __init__(self, number, title, body):
        self.number = number
        self.title = title
        self.body = body
        self._images()
        self._embedding()

    def _images(self):
        pass

    def _embedding(self):
        pass


class Repo:

    def __init__(self, owner, repo, token):
        self.owner = owner
        self.repo = repo
        self.token = token
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
            res = _req(url, self.token)
            data = res.json()
            issues = [item for item in data if 'pull_request' not in item]
            for i in issues:
                self.issues.append(Issue(i['number'], i['title'], i['body']))
            if len(data) < per_page:
                break
            page += 1


env = Env()
repo = Repo(env.owner, env.repo, env.github)
print(repo.issues[100].number)
