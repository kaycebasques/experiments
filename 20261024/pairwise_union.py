from json import load
from os import environ
from pathlib import Path
from sys import argv

import numpy as np

thread_id = 21941278
threshold = 0.92
min_size: int | None = None
max_size: int | None = None
for arg in argv[1:]:
    if arg.startswith('--id='):
        thread_id = int(arg.split('=', 1)[1])
    elif arg.startswith('--threshold='):
        threshold = float(arg.split('=', 1)[1])
    elif arg.startswith('--min='):
        min_size = int(arg.split('=', 1)[1])
    elif arg.startswith('--max='):
        max_size = int(arg.split('=', 1)[1])
    else:
        raise ValueError(
            f'Invalid argument: {arg} '
            '(use --id=<id>, --threshold=0.XX, --min=X, and --max=X)'
        )

bwd = environ['BUILD_WORKSPACE_DIRECTORY']
predictions_path = Path(f'{bwd}/{thread_id}_predictions.json')
with open(predictions_path, encoding='utf-8') as f:
    records = load(f)

n = len(records)
emb = np.array([r['clustering'] for r in records], dtype=np.float32)
sim_matrix = emb @ emb.T

rows, cols = np.triu_indices(n, k=1)
mask = sim_matrix[rows, cols] >= threshold
pair_rows = rows[mask]
pair_cols = cols[mask]
pair_sims = sim_matrix[pair_rows, pair_cols]
order = np.argsort(-pair_sims)

parent = list(range(n))
members: list[list[int]] = [[i] for i in range(n)]
vec_sums = emb.copy()


def find(x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


for k in order:
    ra = find(int(pair_rows[k]))
    rb = find(int(pair_cols[k]))
    if ra == rb:
        continue
    size_a = len(members[ra])
    size_b = len(members[rb])
    avg_sim = float(vec_sums[ra] @ vec_sums[rb]) / (size_a * size_b)
    if avg_sim >= threshold:
        if size_a < size_b:
            ra, rb = rb, ra
        parent[rb] = ra
        members[ra].extend(members[rb])
        members[rb] = []
        vec_sums[ra] = vec_sums[ra] + vec_sums[rb]

clusters = [m for m in members if m]
clusters.sort(key=len, reverse=True)
filtered = [
    c
    for c in clusters
    if (min_size is None or len(c) >= min_size)
    and (max_size is None or len(c) <= max_size)
]

print(
    f'Pairwise Union-Find with All-Members Check '
    f'(id={thread_id}, threshold={threshold}, min={min_size}, max={max_size}): '
    f'Loaded {n} predictions -> {len(clusters)} clusters '
    f'({len(filtered)} matching filter)'
)

for rank, cluster in enumerate(filtered, start=1):
    idx_arr = np.array(cluster)
    centroid = emb[idx_arr].mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    sims = emb[idx_arr] @ centroid
    sorted_idx = idx_arr[np.argsort(-sims)]
    print(f'\nCluster {rank} (size={len(cluster)}):')
    for idx in sorted_idx:
        print(f'  - [{records[idx]["id"]}] {records[idx]["text"]}')
