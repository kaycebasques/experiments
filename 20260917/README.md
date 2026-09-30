# 20260917

All workflows assume that your current working directory is the directory
containing this repo.

## Quickstart

1. Create a classic GitHub personal access token: https://github.com/settings/tokens

   The only scope it needs is `public_repo`.

2. Create a Gemini API key: https://aistudio.google.com/apikey

3. Create an `.env` file in the same directory as this README with the
   following vars configured:

   ```
   GEMINI_API_KEY=…
   OWNER=…
   REPO=…
   GITHUB_TOKEN=…
   ```

4. `../bzl run :index`

## Development

### Update Python dependencies

1. Edit `pypi.txt`.
2. `../bzl run //:pypi.update`
