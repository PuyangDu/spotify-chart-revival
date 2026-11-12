# Spotify chart revival

Code for forecasting older songs returning to Spotify country charts.
This repository contains source code, tests, dependency configuration, and usage instructions only.
Data, checkpoints, experiment outputs, and report artifacts are not tracked.

## Setup

The pipeline was developed with Python 3.13 on CPU.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

## Run

```bash
python -m spotify_revival.download --markets all
python -m spotify_revival.prepare
python -m pytest -q
python scripts/audit_data.py
python -m spotify_revival.train
python -m spotify_revival.evaluate --checkpoint checkpoints/rgcn_best.pt
python scripts/diagnose_loss.py
```

The public source is [MGD+](https://zenodo.org/records/8086643).
The downloader verifies the publisher's archive checksum and extracted member CRCs.
It writes data to `data/raw/`; preparation writes weekly graphs to `data/processed/`.
The default modeled countries are US, UK, Canada, and Brazil. Pass `--markets` to preparation to change them.
Preparation refuses to overwrite an existing manifest; use a fresh `--output` directory.

Eligible songs first charted at least 180 days before a Sunday forecast and were absent on all dates
in `[t-28, t]`. A positive label requires at least three chart days in `(t, t+28]`.
Training uses 2017–2019, validation uses 2020, and outcome windows crossing split boundaries are excluded.
The test period is not prepared or evaluated by the milestone commands.

Training defaults to a two-layer R-GCN, hidden dimension 32, dropout 0.1, Adam learning rate 0.001,
weight decay 0.0001, seed 224, at most 20 epochs, and patience 5. Selection uses validation AP.
The recency baseline scores a song as the negative number of days since its latest local chart appearance;
it needs no fitting. Both models rank the same candidate sets.

AP is pooled across validation examples. Precision/recall at 1% select `ceil(0.01*N)` candidates
separately per country-week before pooling hits and counts. Ties use sorted Spotify track IDs.

Local outputs go to `results/` and `checkpoints/`. The evaluator recomputes metrics and optionally
verifies inference from the saved checkpoint. The diagnostic script compares train/validation loss
using the same frozen checkpoint in evaluation mode and reports class-conditional BCE and prevalence.

## Code layout

- `src/spotify_revival/`: download, preparation, R-GCN, training, metrics, and evaluation.
- `scripts/audit_data.py`: independent raw-record audit of labels and graph edges.
- `scripts/diagnose_loss.py`: comparable loss diagnostics without retraining.
- `tests/`: temporal boundaries, leakage invariance, ranking budgets, and graph gradients.
