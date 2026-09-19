import argparse
from os import environ
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
import polars as pl


def resolve_path(path_str):
    path = Path(path_str)
    if not path.is_absolute() and 'BUILD_WORKING_DIRECTORY' in environ:
        return Path(environ['BUILD_WORKING_DIRECTORY']) / path
    return path


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
    print(f"Loaded {n_rows} embedding records from {parquet_path}")

    # Extract metadata columns
    issue_numbers = df['issue_number'].to_numpy()
    titles = df['title'].to_list()
    chunk_texts = df['chunk_text'].to_list()

    # Extract embeddings matrix (N, D) - zero-copy from Polars
    # Gemini embeddings are already unit-normalized, so dot product is exact cosine similarity.
    matrix = df['embedding'].to_numpy(allow_copy=False)

    print(f"Exhaustively comparing embeddings (similarity threshold >= {threshold:.2f})...\n")

    issue_neighbors = {}
    issue_titles = {}
    pairwise_matches = []
    seen_pairs = set()

    # Loop through each embedding and search across all other embeddings
    for i in range(n_rows):
        src_issue = issue_numbers[i]
        src_title = titles[i]
        issue_titles[src_issue] = src_title

        # Dot product on unit-normalized vectors gives cosine similarity
        sims = matrix[i] @ matrix.T

        for j in range(n_rows):
            tgt_issue = issue_numbers[j]
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
    print(f"Found {len(pairwise_matches)} similar issue pair(s):\n")

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
                for neighbor in issue_neighbors.get(curr, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            if len(component) > 1:
                clusters.append(component)

    clusters.sort(key=len, reverse=True)
    print(f"Discovered {len(clusters)} cluster(s) of related issues:\n")
    for c_idx, cluster in enumerate(clusters, 1):
        print(f"Cluster #{c_idx} ({len(cluster)} issues):")
        for num in sorted(cluster):
            print(f"  - https://github.com/{owner}/{repo}/issues/{num} - {issue_titles.get(num, '')}")
        print()


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
