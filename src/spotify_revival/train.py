"""Train one R-GCN, select its epoch on 2020 AP, and compare a recency rule."""

import argparse
import copy
import hashlib
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import sklearn
import torch
import torch_geometric
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F

from .metrics import ranking_metrics
from .model import RevivalRGCN

FEATURE_KEYS = ["song_x", "artist_x", "country_x", "pair_x"]


def load_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def fit_scalers(train):
    scalers = {key: StandardScaler() for key in FEATURE_KEYS}
    for batch in train:
        for key, scaler in scalers.items():
            values = batch[key]
            if key == "song_x":
                values = values[batch["song_active"]]
            elif key == "artist_x":
                values = values[batch["artist_active"]]
            if len(values):
                scaler.partial_fit(values)
    return scalers


def tensorize(batch, scalers):
    result = {}
    for key, array in batch.items():
        if key in scalers:
            array = scalers[key].transform(array).astype(np.float32)
            if key in ("song_x", "artist_x"):
                array[~batch[key.replace("_x", "_active")]] = 0
        if key in (*FEATURE_KEYS, "edge_index", "edge_type", "song", "country", "y"):
            result[key] = torch.from_numpy(array)
    result["day"] = int(batch["day"])
    result["gap"] = batch["gap"]
    return result


def collect_targets(batches, n_countries):
    return {
        "y": np.concatenate([batch["y"].numpy() for batch in batches]),
        "group": np.concatenate(
            [batch["day"] * n_countries + batch["country"].numpy() for batch in batches]
        ),
        "song": np.concatenate([batch["song"].numpy() for batch in batches]),
        "recency": np.concatenate([-batch["gap"].astype(np.float64) for batch in batches]),
    }


@torch.no_grad()
def evaluate(model, batches):
    model.eval()
    loss, total, predictions = 0.0, 0, []
    for batch in batches:
        logits = model(batch)
        loss += float(F.binary_cross_entropy_with_logits(logits, batch["y"], reduction="sum"))
        total += len(logits)
        predictions.append(logits.sigmoid().numpy())
    return loss / total, np.concatenate(predictions)


def compact(metrics):
    return {key: value for key, value in metrics.items() if key != "groups"}


def run(args):
    started = time.monotonic()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=True)
    args.checkpoints.mkdir(parents=True, exist_ok=True)
    manifest_path = args.data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    train_arrays = [
        load_arrays(args.data / item["file"])
        for item in manifest["snapshots"]
        if item["split"] == "train"
    ]
    val_arrays = [
        load_arrays(args.data / item["file"])
        for item in manifest["snapshots"]
        if item["split"] == "validation"
    ]
    scalers = fit_scalers(train_arrays)
    scaler_state = {
        key: {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(), "fit_split": "train"}
        for key, scaler in scalers.items()
    }
    (args.output / "scalers.json").write_text(json.dumps(scaler_state, indent=2) + "\n")
    train = [tensorize(batch, scalers) for batch in train_arrays]
    validation = [tensorize(batch, scalers) for batch in val_arrays]
    del train_arrays, val_arrays
    targets = collect_targets(validation, len(manifest["markets"]))
    baseline = ranking_metrics(targets["y"], targets["recency"], targets["group"], targets["song"])
    (args.output / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
    print("RECENCY", json.dumps(compact(baseline)), flush=True)
    train_size = sum(len(batch["y"]) for batch in train)
    train_positive = sum(int(batch["y"].sum()) for batch in train)
    config = {
        "hidden": args.hidden,
        "dropout": args.dropout,
        "seed": args.seed,
        "max_epochs": args.epochs,
        "patience": args.patience,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "threads": args.threads,
        "layers": 2,
        "relations": 6,
        "device": "cpu",
        "loss": "unweighted binary cross entropy; all candidates; no negative sampling",
        "selection": "maximum 2020 validation AP",
        "batch": "one dated graph",
        "input_dims": [train[0][key].shape[1] for key in FEATURE_KEYS[:3]],
        "pair_dim": train[0]["pair_x"].shape[1],
        "train_prevalence": train_positive / train_size,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "versions": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torch_geometric": torch_geometric.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    model = RevivalRGCN(
        config["input_dims"],
        config["pair_dim"],
        args.hidden,
        args.dropout,
        config["train_prevalence"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_ap, best_epoch, best_state = -1.0, 0, None
    history = []
    rng = np.random.default_rng(args.seed)
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.monotonic()
        model.train()
        train_loss = 0.0
        for index in rng.permutation(len(train)):
            batch = train[index]
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = F.binary_cross_entropy_with_logits(logits, batch["y"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += float(loss.detach()) * len(logits)
        val_loss, prediction = evaluate(model, validation)
        metrics = ranking_metrics(targets["y"], prediction, targets["group"], targets["song"])
        row = {
            "epoch": epoch,
            "train_bce": train_loss / train_size,
            "validation_bce": val_loss,
            "validation_ap": metrics["average_precision"],
            "seconds": time.monotonic() - epoch_started,
        }
        history.append(row)
        print("EPOCH", json.dumps(row), flush=True)
        (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if metrics["average_precision"] > best_ap:
            best_ap, best_epoch = metrics["average_precision"], epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {"state_dict": best_state, "config": config, "epoch": best_epoch},
                args.checkpoints / "rgcn_best.pt",
            )
        if epoch - best_epoch >= args.patience:
            break
    model.load_state_dict(best_state)
    val_loss, prediction = evaluate(model, validation)
    rgcn = ranking_metrics(targets["y"], prediction, targets["group"], targets["song"])
    results = {
        "evaluation_split": "2020 validation; test period untouched",
        "selected_epoch": best_epoch,
        "epochs_run": len(history),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training_examples": train_size,
        "training_positives": train_positive,
        "rgcn": compact(rgcn),
        "recency": compact(baseline),
        "selected_validation_bce": val_loss,
        "total_seconds": time.monotonic() - started,
        "top1pct_aggregation": "Select ceil(0.01*N) per country-week; pool hits/budgets/positives",
        "ap_aggregation": "Pooled across all validation examples",
        "tie_break": "Lexicographically sorted Spotify track ID",
    }
    (args.output / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    (args.output / "rgcn_groups.json").write_text(json.dumps(rgcn["groups"], indent=2) + "\n")
    np.savez_compressed(args.output / "validation_predictions.npz", **targets, rgcn=prediction)
    print("FINAL", json.dumps(results, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=224)
    parser.add_argument("--threads", type=int, default=2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
