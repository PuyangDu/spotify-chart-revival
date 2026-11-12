"""Pooled AP and micro-aggregated top-1% metrics with a per-country-week budget."""

import numpy as np
from sklearn.metrics import average_precision_score


def ranking_metrics(labels, scores, groups, song_ids):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores)
    groups = np.asarray(groups)
    song_ids = np.asarray(song_ids)
    if not (len(labels) == len(scores) == len(groups) == len(song_ids)):
        raise ValueError("Metric inputs have different lengths")
    if not len(labels) or not np.isfinite(scores).all():
        raise ValueError("Metrics require nonempty examples and finite scores")
    hits = selected = 0
    per_group = []
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        k = max(1, int(np.ceil(0.01 * len(indices))))
        # Stable, model-independent tie breaking using sorted Spotify track IDs.
        order = np.lexsort((song_ids[indices], -scores[indices]))[:k]
        found = int(labels[indices[order]].sum())
        hits += found
        selected += k
        per_group.append(
            {
                "group": int(group),
                "candidates": len(indices),
                "k": k,
                "positives": int(labels[indices].sum()),
                "hits": found,
            }
        )
    positives = int(labels.sum())
    return {
        "n": len(labels),
        "positives": positives,
        "prevalence": float(labels.mean()),
        "average_precision": float(average_precision_score(labels, scores)) if positives else 0.0,
        "precision_at_1pct": hits / selected,
        "recall_at_1pct": hits / positives if positives else 0.0,
        "selected": selected,
        "hits": hits,
        "country_weeks": len(per_group),
        "groups": per_group,
    }
