import argparse
from json import dumps
from os import environ
from pathlib import Path
import signal
from time import sleep
from typing import TypedDict

from dotenv import load_dotenv
from google import genai
from google.genai import errors
from google.genai import types
import numpy as np
import polars as pl

# Polars spawns a pool of background Rust worker threads without masking SIGINT.
# When Ctrl-C is pressed during a blocking SSL_read on the main thread (e.g.
# waiting on a Gemini API call), the kernel may deliver SIGINT to a sleeping
# Polars worker thread instead of the main thread, so Python's signal handler
# only sets a pending flag and cannot raise KeyboardInterrupt until the network
# read finishes. Restoring SIG_DFL lets the kernel terminate the process
# immediately on Ctrl-C.
signal.signal(signal.SIGINT, signal.SIG_DFL)


class Response(TypedDict):
    dupes: list[list[int]]
    nondupes: list[int]


def resolve_path(path_str):
    path = Path(path_str)
    if not path.is_absolute() and 'BUILD_WORKING_DIRECTORY' in environ:
        return Path(environ['BUILD_WORKING_DIRECTORY']) / path
    return path


def analyze_cluster(gemini, cluster, issue_titles, issue_bodies, issue_images):
    config = types.GenerateContentConfig(
        response_mime_type='application/json',
        response_schema=Response,
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=1)
        ),
    )
    models = ('gemini-3.8-flash', 'gemini-3.7-flash', 'gemini-3.6-flash')
    contents = [
        'Analyze the following cluster of GitHub issues and identify which issues '
        'are duplicates of each other and which are not.\n'
        'Return a JSON object with:\n'
        '- "dupes": list of lists of issue numbers, where each inner list is a set of issues that are duplicates of each other '
        '(e.g. [[1, 2, 3], [4, 5, 6]] means 1, 2, and 3 are duplicates of each other, and 4, 5, and 6 are duplicates of each other).\n'
        '- "nondupes": list of issue numbers that are not duplicates of any other issue in this cluster.\n'
        f'Every issue number in {sorted(cluster)} must be placed in either "dupes" or "nondupes".'
    ]
    for num in sorted(cluster):
        issue_data = dumps({
            'issue_number': num,
            'title': issue_titles[num],
            'body': issue_bodies[num],
        })
        contents.append(f'Issue #{num}:\n{issue_data}')
        for img in issue_images[num]:
            contents.append(
                types.Part.from_bytes(data=img['bytes'], mime_type=img['mime'])
            )

    retryable = (429, 500, 502, 503, 504)
    while True:
        for model in models:
            try:
                response = gemini.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
                return response.parsed
            except errors.APIError as e:
                if e.code in retryable:
                    print(f'  [{model} returned {e.code} {e.status}, retrying...]', flush=True)
                    continue
                raise
        sleep(5)


def find_clusters(parquet_path='embeddings.parquet', threshold=0.9):
    path = resolve_path(parquet_path)
    if not path.exists():
        print(f"Error: '{parquet_path}' does not exist. Run ':index' first to generate embeddings.")
        return

    df = pl.read_parquet(path)
    if df.is_empty():
        print(f"Parquet file '{parquet_path}' is empty.")
        return

    n_rows = len(df)
    print(f"Loaded {n_rows} embedding records from {parquet_path}", flush=True)

    # Extract metadata columns
    issue_numbers = df['issue_number'].to_numpy()
    titles = df['title'].to_list()
    bodies = df['body'].to_list()
    chunk_texts = df['chunk_text'].to_list()
    images_col = df['images'].to_list()

    # Extract embeddings matrix (N, D) - zero-copy from Polars
    # Gemini embeddings are already unit-normalized, so dot product is exact cosine similarity.
    matrix = df['embedding'].to_numpy(allow_copy=False)

    print(f"Exhaustively comparing embeddings (similarity threshold >= {threshold:.2f})...\n", flush=True)

    issue_neighbors = {}
    issue_titles = {}
    issue_bodies = {}
    issue_images = {}
    pairwise_matches = []
    seen_pairs = set()

    # Loop through each embedding and search across all other embeddings
    for i in range(n_rows):
        src_issue = int(issue_numbers[i])
        src_title = titles[i]
        issue_titles[src_issue] = src_title
        issue_bodies[src_issue] = bodies[i]
        if src_issue not in issue_images:
            issue_images[src_issue] = []
        for img in images_col[i]:
            if not any(existing['bytes'] == img['bytes'] for existing in issue_images[src_issue]):
                issue_images[src_issue].append(img)

        # Dot product on unit-normalized vectors gives cosine similarity
        sims = matrix[i] @ matrix.T

        for j in range(n_rows):
            tgt_issue = int(issue_numbers[j])
            # Skip chunks belonging to the same issue
            if src_issue == tgt_issue:
                continue

            score = float(sims[j])
            if score >= threshold:
                pair_key = (min(src_issue, tgt_issue), max(src_issue, tgt_issue))
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    pairwise_matches.append({
                        'issue_a': src_issue,
                        'title_a': src_title,
                        'chunk_a': chunk_texts[i],
                        'issue_b': tgt_issue,
                        'title_b': titles[j],
                        'chunk_b': chunk_texts[j],
                        'score': score,
                    })

                issue_neighbors.setdefault(src_issue, set()).add(tgt_issue)
                issue_neighbors.setdefault(tgt_issue, set()).add(src_issue)

    if not pairwise_matches:
        print(f"No similar issue pairs found with similarity >= {threshold:.2f}.")
        return

    owner = environ['GITHUB_OWNER']
    repo = environ['GITHUB_REPO']
    pairwise_matches.sort(key=lambda x: x['score'], reverse=True)
    print(f"Found {len(pairwise_matches)} similar issue pair(s):\n", flush=True)

    # Group connected components into clusters
    visited = set()
    clusters = []
    for issue in sorted(issue_neighbors.keys()):
        if issue not in visited:
            component = []
            queue = [issue]
            visited.add(issue)
            while queue:
                curr = queue.pop(0)
                component.append(curr)
                for neighbor in issue_neighbors[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            if len(component) > 1:
                clusters.append(component)

    clusters.sort(key=len, reverse=True)
    print(f"Discovered {len(clusters)} cluster(s) of related issues:\n", flush=True)
    gemini = genai.Client(api_key=environ['GEMINI_API_KEY'])
    for c_idx, cluster in enumerate(clusters, 1):
        print(f"Cluster #{c_idx} ({len(cluster)} issues):", flush=True)
        analysis = analyze_cluster(
            gemini, cluster, issue_titles, issue_bodies, issue_images
        )
        dupes = analysis['dupes']
        nondupes = analysis['nondupes']
        for g_idx, group in enumerate(dupes, 1):
            label = f"Duplicates (set #{g_idx}):" if len(dupes) > 1 else "Duplicates:"
            print(f"\n  {label}", flush=True)
            for num in sorted(group):
                print(f"    - https://github.com/{owner}/{repo}/issues/{num} - {issue_titles[num]}", flush=True)
        if nondupes:
            print("\n  Non-duplicates:", flush=True)
            for num in sorted(nondupes):
                print(f"    - https://github.com/{owner}/{repo}/issues/{num} - {issue_titles[num]}", flush=True)
        print(flush=True)


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Cluster similar GitHub issues by embedding cosine similarity."
    )
    parser.add_argument(
        '--path',
        default='embeddings.parquet',
        help='Path to embeddings.parquet (default: embeddings.parquet)',
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.9,
        help='Cosine similarity threshold between 0.0 and 1.0 (default: 0.9)',
    )
    args = parser.parse_args()
    find_clusters(parquet_path=args.path, threshold=args.threshold)


if __name__ == '__main__':
    main()
