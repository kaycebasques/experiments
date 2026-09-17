# 20260917

## Quickstart

1. Create an `.env` file in the same directory as this README with the
   following vars configured:

   ```
   GEMINI_API_KEY=…
   OWNER=…
   REPO=…
   GITHUB_TOKEN=…
   ```

2. `../bzl run :index`

## Development

### Update Python dependencies

1. Edit `pypi.txt`.
2. `../bzl run //:pypi.update`
