"""Recompute saved prediction metrics without retraining or touching the test period."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from .metrics import ranking_metrics
from .model import RevivalRGCN
from .train import evaluate, load_arrays, tensorize


def checkpoint_predictions(checkpoint_path, data_dir, results_dir):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    torch.set_num_threads(config["threads"])
    manifest_bytes = (data_dir / "manifest.json").read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != config["manifest_sha256"]:
        raise ValueError("Prepared data manifest differs from the training run")
    manifest = json.loads(manifest_bytes)
    scalers = {}
    for key, values in json.loads((results_dir / "scalers.json").read_text()).items():
        scaler = StandardScaler()
        scaler.mean_ = np.asarray(values["mean"])
        scaler.scale_ = np.asarray(values["scale"])
        scaler.n_features_in_ = len(scaler.mean_)
        scalers[key] = scaler
    validation = [
        tensorize(load_arrays(data_dir / item["file"]), scalers)
        for item in manifest["snapshots"]
        if item["split"] == "validation"
    ]
    model = RevivalRGCN(
        config["input_dims"],
        config["pair_dim"],
        config["hidden"],
        config["dropout"],
        config["train_prevalence"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    return evaluate(model, validation)[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument(
        "--checkpoint", type=Path, help="Also verify fresh inference from this checkpoint"
    )
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    recorded = json.loads((args.results / "metrics.json").read_text())
    with np.load(args.results / "validation_predictions.npz", allow_pickle=False) as data:
        if args.checkpoint:
            prediction = checkpoint_predictions(args.checkpoint, args.data, args.results)
            np.testing.assert_allclose(prediction, data["rgcn"], rtol=1e-6, atol=1e-8)
            receipt = {
                "status": "passed",
                "predictions_checked": len(prediction),
                "max_absolute_difference": float(np.max(np.abs(prediction - data["rgcn"]))),
            }
            (args.results / "checkpoint_verification.json").write_text(
                json.dumps(receipt, indent=2) + "\n"
            )
            print("Checkpoint inference matches saved predictions.")
        for name in ("recency", "rgcn"):
            actual = ranking_metrics(data["y"], data[name], data["group"], data["song"])
            for metric in ("average_precision", "precision_at_1pct", "recall_at_1pct"):
                np.testing.assert_allclose(actual[metric], recorded[name][metric], rtol=1e-12)
            print(name, json.dumps({k: v for k, v in actual.items() if k != "groups"}))


if __name__ == "__main__":
    main()
