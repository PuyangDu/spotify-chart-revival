"""Compare train/validation BCE using the same frozen checkpoint with dropout disabled."""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F

from spotify_revival.model import RevivalRGCN
from spotify_revival.train import load_arrays, tensorize


@torch.no_grad()
def main():
    root = Path(__file__).resolve().parents[1]
    checkpoint = torch.load(
        root / "checkpoints/rgcn_best.pt", weights_only=True, map_location="cpu"
    )
    config = checkpoint["config"]
    torch.set_num_threads(config["threads"])
    model = RevivalRGCN(
        config["input_dims"], config["pair_dim"], config["hidden"], config["dropout"]
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    scalers = {}
    for key, values in json.loads((root / "results/scalers.json").read_text()).items():
        scaler = StandardScaler()
        scaler.mean_ = np.asarray(values["mean"])
        scaler.scale_ = np.asarray(values["scale"])
        scaler.n_features_in_ = len(scaler.mean_)
        scalers[key] = scaler
    manifest = json.loads((root / "data/processed/manifest.json").read_text())
    result = {"checkpoint_epoch": checkpoint["epoch"], "mode": "eval; dropout disabled"}
    q = config["train_prevalence"]
    for split in ("train", "validation"):
        count = np.zeros(2, dtype=np.int64)
        loss_sum = np.zeros(2, dtype=np.float64)
        for item in manifest["snapshots"]:
            if item["split"] != split:
                continue
            batch = tensorize(load_arrays(root / "data/processed" / item["file"]), scalers)
            losses = F.binary_cross_entropy_with_logits(model(batch), batch["y"], reduction="none")
            for label in (0, 1):
                mask = batch["y"] == label
                count[label] += int(mask.sum())
                loss_sum[label] += float(losses[mask].double().sum())
        prevalence = float(count[1] / count.sum())
        result[split] = {
            "examples": int(count.sum()),
            "positive_fraction": prevalence,
            "frozen_eval_bce": float(loss_sum.sum() / count.sum()),
            "negative_class_bce": float(loss_sum[0] / count[0]),
            "positive_class_bce": float(loss_sum[1] / count[1]),
            "bce_at_training_class_proportions": float(
                (1 - q) * loss_sum[0] / count[0] + q * loss_sum[1] / count[1]
            ),
            "constant_train_prior_bce": float(
                -prevalence * np.log(q) - (1 - prevalence) * np.log1p(-q)
            ),
        }
    (root / "results/loss_diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
