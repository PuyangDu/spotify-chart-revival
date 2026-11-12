"""Independently reconstruct candidate sets, labels, and country edges from raw rows."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]
    processed = root / "data/processed"
    manifest = json.loads((processed / "manifest.json").read_text())
    song_map = {song: i for i, song in enumerate(manifest["song_ids"])}
    dates = ["2018-06-03", "2019-11-03", "2020-03-01", "2020-11-29"]
    checked_examples = checked_edges = 0
    for country, market in enumerate(manifest["markets"]):
        raw = pd.read_csv(root / "data/raw" / f"spotify_charts_regional_{market}.csv", sep="\t")
        raw["date"] = pd.to_datetime(raw.Date)
        raw = raw[(raw.Chart == "Top 200") & pd.to_numeric(raw.Position).between(1, 200)]
        raw = raw.drop_duplicates(["ID", "date"])
        for date in dates:
            info = next(s for s in manifest["snapshots"] if s["date"] == date)
            t = pd.Timestamp(date)
            snapshot = np.load(processed / info["file"], allow_pickle=False)
            past = raw[raw.date <= t]
            history = past.groupby("ID").date.agg(["min", "max"])
            candidates = history[
                (history["min"] <= t - pd.Timedelta(days=180))
                & (history["max"] < t - pd.Timedelta(days=28))
            ]
            target = snapshot["country"] == country
            actual_songs = snapshot["song"][target]
            expected_songs = sorted(song_map[song] for song in candidates.index)
            np.testing.assert_array_equal(actual_songs, expected_songs)
            future = raw[(raw.date > t) & (raw.date <= t + pd.Timedelta(days=28))]
            future_counts = future.groupby("ID").date.nunique()
            expected_y = [
                int(future_counts.get(manifest["song_ids"][song], 0) >= 3)
                for song in expected_songs
            ]
            np.testing.assert_array_equal(snapshot["y"][target], expected_y)
            gap = [
                (t - history.loc[manifest["song_ids"][song], "max"]).days for song in expected_songs
            ]
            np.testing.assert_array_equal(snapshot["gap"][target], gap)
            for relation, mask in [
                (2, history["max"] >= t - pd.Timedelta(days=27)),
                (4, history["max"] < t - pd.Timedelta(days=27)),
            ]:
                edges = snapshot["edge_index"][:, snapshot["edge_type"] == relation]
                node = len(manifest["song_ids"]) + len(manifest["artist_ids"]) + country
                actual = sorted(edges[0, edges[1] == node])
                expected = sorted(song_map[song] for song in history.index[mask])
                np.testing.assert_array_equal(actual, expected)
                checked_edges += len(expected)
            checked_examples += len(expected_songs)
    result = {
        "status": "passed",
        "dates": dates,
        "countries": manifest["markets"],
        "candidate_labels_and_gaps_checked": checked_examples,
        "song_country_edges_checked": checked_edges,
        "method": "Independent raw-row groupby reconstruction; no preparation helpers",
    }
    (root / "results/independent_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
