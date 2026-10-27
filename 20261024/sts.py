from json import load
from os import environ
from pathlib import Path
from sys import argv

import numpy as np

thread_id = 21941278
threshold = 0.88
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

emb = np.array([r['similarity'] for r in records], dtype=np.float32)
sim_matrix = emb @ emb.T

degrees = (sim_matrix >= threshold).sum(axis=1)
order = np.argsort(-degrees)

clusters: list[list[int]] = []

for idx in order:
    i = int(idx)
    best_c = -1
    best_sim = -1.0
    for c_idx, members in enumerate(clusters):
        sims = sim_matrix[i, members]
        if np.all(sims >= threshold):
            mean_sim = float(sims.mean())
            if mean_sim > best_sim:
                best_sim = mean_sim
                best_c = c_idx
    if best_c != -1:
        clusters[best_c].append(i)
    else:
        clusters.append([i])

clusters.sort(key=len, reverse=True)
filtered = [
    c
    for c in clusters
    if (min_size is None or len(c) >= min_size)
    and (max_size is None or len(c) <= max_size)
]

print(
    f'STS All-Members Clique Clustering '
    f'(id={thread_id}, threshold={threshold}, min={min_size}, max={max_size}): '
    f'Loaded {len(records)} predictions -> {len(clusters)} total clusters '
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
