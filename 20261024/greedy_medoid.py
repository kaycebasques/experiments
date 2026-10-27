from json import load
from os import environ
from pathlib import Path
from sys import argv

import numpy as np

thread_id = 21941278
threshold = 0.93
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
adj = sim_matrix >= threshold

unassigned = np.ones(len(records), dtype=bool)
clusters: list[tuple[int, np.ndarray]] = []

while unassigned.any():
    active_adj = adj & unassigned[None, :]
    counts = np.where(unassigned, active_adj.sum(axis=1), -1)
    hub = int(np.argmax(counts))
    members = np.where(active_adj[hub])[0]
    member_sims = sim_matrix[hub, members]
    members = members[np.argsort(-member_sims)]
    clusters.append((hub, members))
    unassigned[members] = False

clusters.sort(key=lambda c: len(c[1]), reverse=True)
filtered = [
    c
    for c in clusters
    if (min_size is None or len(c[1]) >= min_size)
    and (max_size is None or len(c[1]) <= max_size)
]

print(
    f'Greedy Medoid (id={thread_id}, threshold={threshold}, min={min_size}, max={max_size}): '
    f'Loaded {len(records)} predictions -> {len(clusters)} clusters '
    f'({len(filtered)} matching filter)'
)

for rank, (hub, members) in enumerate(filtered, start=1):
    print(
        f'\nCluster {rank} (size={len(members)}, '
        f'medoid=[{records[hub]["id"]}] {records[hub]["text"]}):'
    )
    for idx in members:
        print(f'  - [{records[idx]["id"]}] {records[idx]["text"]}')
