from json import load
from os import environ
from pathlib import Path
from sys import argv

import numpy as np
from sklearn.cluster import AgglomerativeClustering

thread_id = 21941278
threshold = 0.08
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

clustering_emb = np.array([r['clustering'] for r in records], dtype=np.float32)

model = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=threshold,
    metric='cosine',
    linkage='average',
)
labels = model.fit_predict(clustering_emb)
unique, counts = np.unique(labels, return_counts=True)
sorted_clusters = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)
filtered = [
    (cid, c)
    for cid, c in sorted_clusters
    if (min_size is None or c >= min_size)
    and (max_size is None or c <= max_size)
]

print(
    f'Agglomerative (id={thread_id}, threshold={threshold}, min={min_size}, max={max_size}): '
    f'Loaded {len(records)} predictions -> {len(unique)} clusters '
    f'({len(filtered)} matching filter)'
)

for cluster_id, count in filtered:
    indices = np.where(labels == cluster_id)[0]
    centroid = clustering_emb[indices].mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    sims = clustering_emb[indices] @ centroid
    top_idx = indices[np.argsort(-sims)]
    print(f'\nCluster {cluster_id} (size={count}):')
    for idx in top_idx:
        print(f'  - [{records[idx]["id"]}] {records[idx]["text"]}')
