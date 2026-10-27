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

emb = np.array([r['clustering'] for r in records], dtype=np.float32)
sim_matrix = emb @ emb.T

degrees = (sim_matrix >= threshold).sum(axis=1)
order = np.argsort(-degrees)

clusters: list[list[int]] = []
centroid_sums: list[np.ndarray] = []

for idx in order:
    vec = emb[idx]
    if centroid_sums:
        sizes = np.array([len(c) for c in clusters], dtype=np.float32)
        sims = (np.array(centroid_sums, dtype=np.float32) @ vec) / sizes
        best_c = int(np.argmax(sims))
        best_sim = float(sims[best_c])
    else:
        best_c = -1
        best_sim = -1.0

    if best_sim >= threshold:
        clusters[best_c].append(int(idx))
        centroid_sums[best_c] = centroid_sums[best_c] + vec
    else:
        clusters.append([int(idx)])
        centroid_sums.append(vec.copy())

paired = sorted(
    zip(clusters, centroid_sums), key=lambda x: len(x[0]), reverse=True
)
filtered = [
    (c, s)
    for c, s in paired
    if (min_size is None or len(c) >= min_size)
    and (max_size is None or len(c) <= max_size)
]

print(
    f'Leader / Running-Centroid (id={thread_id}, threshold={threshold}, min={min_size}, max={max_size}): '
    f'Loaded {len(records)} predictions -> {len(paired)} clusters '
    f'({len(filtered)} matching filter)'
)

for rank, (members, c_sum) in enumerate(filtered, start=1):
    member_arr = np.array(members)
    centroid = c_sum / np.linalg.norm(c_sum)
    sims = emb[member_arr] @ centroid
    sorted_members = member_arr[np.argsort(-sims)]
    leader = members[0]
    print(
        f'\nCluster {rank} (size={len(members)}, '
        f'seed=[{records[leader]["id"]}] {records[leader]["text"]}):'
    )
    for idx in sorted_members:
        print(f'  - [{records[idx]["id"]}] {records[idx]["text"]}')
