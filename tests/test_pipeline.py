"""Boundary, leakage, and ranking-budget tests on independently specified examples."""

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from spotify_revival.metrics import ranking_metrics
from spotify_revival.model import RevivalRGCN
from spotify_revival.prepare import (
    TRAIN_END,
    VALID_END,
    ChartHistory,
    build_snapshot,
    eligible,
    split_for_day,
)
from spotify_revival.train import fit_scalers, tensorize


def history(events, n_days=300):
    frame = pd.DataFrame(events, columns=["ID", "day", "rank", "streams"])
    frame = frame.sort_values(["day", "rank"])
    return ChartHistory(frame, np.ones(n_days, dtype=bool), {"a": 0, "b": 1, "c": 2})


def test_eligibility_boundaries():
    first = np.array([20, 21, 0, 0, 0])
    last = np.array([171, 100, 172, 200, -1])
    assert eligible(first, last, 200).tolist() == [True, False, False, False, False]


def test_split_embargo():
    assert split_for_day(TRAIN_END - 29) == "train"
    assert split_for_day(TRAIN_END - 28) is None
    assert split_for_day(TRAIN_END) == "validation"
    assert split_for_day(VALID_END - 29) == "validation"
    assert split_for_day(VALID_END - 28) is None
    assert split_for_day(VALID_END) is None


def test_labels_exclude_today_include_day28_and_count_distinct_days():
    h = history(
        [
            ("a", 0, 10, 10),
            ("a", 201, 20, 10),
            ("a", 202, 20, 10),
            ("a", 228, 20, 10),
            ("b", 0, 20, 10),
            ("b", 200, 20, 10),
            ("b", 201, 20, 10),
            ("b", 202, 20, 10),
            ("b", 229, 20, 10),
        ]
    )
    assert h.label_counts(200).tolist() == [3, 2, 0]
    h.complete[228] = False
    assert not h.complete_window(200)


def test_future_edits_change_labels_but_not_model_inputs():
    past = [("a", 0, 50, 100), ("a", 100, 90, 30), ("b", 190, 10, 1000)]
    future = [("a", 201, 1, 10000), ("a", 202, 1, 10000), ("a", 228, 1, 10000), ("c", 205, 2, 9000)]
    incidence = sparse.csr_matrix(np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float32))
    audio = np.ones((3, 20), dtype=np.float32)
    before, _ = build_snapshot(200, [history(past)], incidence, audio)
    after, _ = build_snapshot(200, [history(past + future)], incidence, audio)
    for key in before:
        if key != "y":
            np.testing.assert_array_equal(before[key], after[key], err_msg=key)
    assert before["y"].tolist() == [0]
    assert after["y"].tolist() == [1]
    assert not after["song_active"][2]
    assert not np.any(after["edge_index"] == 2)


def test_top_budget_is_per_group_and_ties_use_track_id():
    # Two groups of 101 each => two selections per group, not three overall.
    y = np.zeros(202)
    y[[0, 1, 101, 102]] = 1
    result = ranking_metrics(y, np.zeros(202), np.repeat([0, 1], 101), np.arange(202))
    assert result["selected"] == 4
    assert result["hits"] == 4
    assert result["precision_at_1pct"] == result["recall_at_1pct"] == 1


def test_model_has_graph_gradients_and_finite_outputs():
    events = [("a", 0, 10, 100), ("b", 0, 20, 80), ("b", 195, 10, 120)]
    incidence = sparse.csr_matrix(np.array([[1, 1, 0]], dtype=np.float32))
    snapshot, _ = build_snapshot(200, [history(events)], incidence, np.ones((3, 20)))
    scalers = fit_scalers([snapshot])
    batch = tensorize(snapshot, scalers)
    torch.manual_seed(224)
    model = RevivalRGCN(
        [batch[k].shape[1] for k in ["song_x", "artist_x", "country_x"]],
        batch["pair_x"].shape[1],
        dropout=0,
    )
    output = model(batch)
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert any(layer.weight.grad.abs().sum() > 0 for layer in model.layers)
