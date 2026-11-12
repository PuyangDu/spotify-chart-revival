"""Build dated graphs and weekly revival labels without looking past each cutoff."""

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .download import DEFAULT_MARKETS

EPOCH = pd.Timestamp("2017-01-01")
TRAIN_END = (pd.Timestamp("2020-01-01") - EPOCH).days
VALID_END = (pd.Timestamp("2021-01-01") - EPOCH).days
AUDIO = [
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "speechiness",
    "valence",
    "loudness",
    "tempo",
    "duration_ms",
]
RELATIONS = [
    "song_to_artist",
    "artist_to_song",
    "song_to_recent_country",
    "recent_country_to_song",
    "song_to_historical_country",
    "historical_country_to_song",
]
PAIR_FEATURES = [
    "log_first_seen_age",
    "log_days_since_chart",
    "log_chart_days",
    "best_rank_scaled",
    "last_rank_scaled",
    "log_last_streams",
    "log_chart_days_90",
    "log_chart_days_180",
    "other_recent_markets",
    "log_other_recent_chart_days",
    "log_other_recent_streams",
    "log_artist_recent_other_tracks",
    "log_collaborator_recent_activity",
    "month_sin",
    "month_cos",
    "local_coverage_28",
    "mean_coverage_28",
]


def date_string(day):
    return str((EPOCH + pd.Timedelta(days=int(day))).date())


def split_for_day(day):
    """Outcome windows must finish strictly before the next calendar-year split."""
    if day + 28 < TRAIN_END:
        return "train"
    if TRAIN_END <= day and day + 28 < VALID_END:
        return "validation"
    return None


def eligible(first, last, day):
    """Absent on all 29 dates in [t-28, t], and first observed at least 180 days ago."""
    return (first <= day - 180) & (last < day - 28) & (last >= 0)


def parse_artists(value):
    if not isinstance(value, str):
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []
    return sorted({str(item) for item in parsed}) if isinstance(parsed, list) else []


def load_chart(path, end_day):
    raw = pd.read_csv(path, sep="\t", usecols=["ID", "Position", "Streams", "Date", "Chart"])
    raw["day"] = (pd.to_datetime(raw.Date, errors="coerce") - EPOCH).dt.days
    raw["rank"] = pd.to_numeric(raw.Position, errors="coerce")
    raw = raw[
        (raw.Chart == "Top 200") & raw.day.between(0, end_day) & raw["rank"].between(1, 200)
    ].copy()
    raw["day"] = raw.day.astype(np.int32)
    raw["streams"] = pd.to_numeric(
        raw.Streams.astype(str).str.replace(",", "", regex=False), errors="coerce"
    )
    if raw.streams.isna().any() or raw.ID.isna().any():
        raise ValueError(f"Missing track IDs or streams in {path}")
    duplicates = int(raw.duplicated(["ID", "day"]).sum())
    raw = raw.sort_values(["day", "rank"]).drop_duplicates(["ID", "day"])
    counts = raw.groupby("day").agg(rows=("ID", "size"), ranks=("rank", "nunique"))
    complete = np.zeros(end_day + 1, dtype=bool)
    complete[counts.index] = (counts.rows.eq(200) & counts.ranks.eq(200)).to_numpy()
    quality = {
        "rows_through_2020": len(raw),
        "unique_track_ids": int(raw.ID.nunique()),
        "duplicate_track_days_removed": duplicates,
        "complete_days_through_2020": int(complete.sum()),
        "incomplete_or_missing_days": int((~complete).sum()),
        "date_min": date_string(raw.day.min()),
        "date_max": date_string(raw.day.max()),
    }
    return raw[["ID", "day", "rank", "streams"]], complete, quality


class ChartHistory:
    """Incremental past-only state; future data is accessed solely by label_counts."""

    def __init__(self, frame, complete, song_map):
        self.song = frame.ID.map(song_map).to_numpy(dtype=np.int64)
        self.day = frame.day.to_numpy(dtype=np.int32)
        self.rank = frame["rank"].to_numpy(dtype=np.float32)
        self.streams = frame.streams.to_numpy(dtype=np.float32)
        self.complete = complete
        self.n = len(song_map)
        self.position = 0
        self.first = np.full(self.n, 100000, dtype=np.int32)
        self.last = np.full(self.n, -1, dtype=np.int32)
        self.count = np.zeros(self.n, dtype=np.float32)
        self.best = np.full(self.n, 201.0, dtype=np.float32)
        self.last_rank = np.full(self.n, 201.0, dtype=np.float32)
        self.last_streams = np.zeros(self.n, dtype=np.float32)

    def window(self, start, end, streams=False):
        left, right = np.searchsorted(self.day, [start, end + 1])
        weights = self.streams[left:right] if streams else None
        return np.bincount(self.song[left:right], weights=weights, minlength=self.n).astype(
            np.float32
        )

    def advance(self, day):
        stop = np.searchsorted(self.day, day, side="right")
        if stop < self.position:
            raise ValueError("History must advance chronologically")
        section = slice(self.position, stop)
        ids, dates = self.song[section], self.day[section]
        np.minimum.at(self.first, ids, dates)
        np.maximum.at(self.last, ids, dates)
        np.add.at(self.count, ids, 1)
        np.minimum.at(self.best, ids, self.rank[section])
        # Explicitly select the last row per track; repeated fancy-index writes are ambiguous.
        if len(ids):
            _, reverse_positions = np.unique(ids[::-1], return_index=True)
            positions = stop - 1 - reverse_positions
            self.last_rank[self.song[positions]] = self.rank[positions]
            self.last_streams[self.song[positions]] = self.streams[positions]
        self.position = stop

    def complete_window(self, day):
        return (
            day >= 28 and day + 28 < len(self.complete) and self.complete[day - 28 : day + 29].all()
        )

    def label_counts(self, day):
        return self.window(day + 1, day + 28)


def graph_edges(active, artist_song, recent, seen, n_songs, n_artists):
    artist, song = artist_song.nonzero()
    keep = active[song]
    song, artist = song[keep], artist[keep] + n_songs
    edges = [np.array([song, artist]), np.array([artist, song])]
    for mask in (recent, seen & ~recent):
        song, country = np.nonzero(mask)
        country = country + n_songs + n_artists
        edges.extend([np.array([song, country]), np.array([country, song])])
    edge_type = np.concatenate(
        [np.full(e.shape[1], r, dtype=np.int64) for r, e in enumerate(edges)]
    )
    return np.concatenate(edges, axis=1).astype(np.int64), edge_type


def build_snapshot(day, histories, artist_song, audio):
    n_songs = len(audio)
    n_artists = artist_song.shape[0]
    for history in histories:
        history.advance(day)
    first = np.column_stack([h.first for h in histories])
    last = np.column_stack([h.last for h in histories])
    counts = np.column_stack([h.count for h in histories])
    recent_days = np.column_stack([h.window(day - 27, day) for h in histories])
    recent_streams = np.column_stack([h.window(day - 27, day, streams=True) for h in histories])
    coverage = np.array(
        [h.complete[day - 27 : day + 1].mean() for h in histories], dtype=np.float32
    )
    seen = last >= 0
    active = seen.any(axis=1)
    recent = seen & (last >= day - 27)
    recent_song = recent.any(axis=1).astype(np.float32)
    artist_active_tracks = np.asarray(artist_song @ active.astype(np.float32)).ravel()
    artist_recent = np.asarray(artist_song @ recent_song).ravel()
    artist_counts = np.asarray(artist_song @ counts.sum(axis=1)).ravel()
    artist_streams = np.asarray(artist_song @ recent_streams.sum(axis=1)).ravel()
    credited_count = np.asarray(artist_song.sum(axis=0)).ravel()
    # Exclude the target song from each credited artist's recent breadth.
    artist_other = np.asarray(artist_song.T @ artist_recent).ravel() - recent_song * credited_count
    active_artist_song = artist_song.multiply(active[None, :]).tocsr()
    collaborators = (active_artist_song @ active_artist_song.T).tocsr()
    collaborators.setdiag(0)
    collaborators.eliminate_zeros()
    collaborators.data[:] = 1
    degree = np.asarray(collaborators.sum(axis=1)).ravel()
    collaborator_activity = (collaborators @ artist_recent) / np.maximum(1, degree)
    collaborator_activity = np.asarray(artist_song.T @ collaborator_activity).ravel()
    month = (EPOCH + pd.Timedelta(days=int(day))).month
    seasonal = [np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)]

    song_history = np.column_stack(
        [
            np.log1p(np.maximum(0, day - first.min(axis=1))),
            np.log1p(np.where(active, day - last.max(axis=1), 0)),
            np.log1p(counts.sum(axis=1)),
            seen.sum(axis=1),
            recent.sum(axis=1),
            np.log1p(recent_days.sum(axis=1)),
            np.log1p(recent_streams.sum(axis=1)),
            np.log1p(artist_other),
            np.log1p(collaborator_activity),
            np.full(n_songs, coverage.mean()),
        ]
    )
    song_x = np.column_stack([audio, song_history]).astype(np.float32)
    song_x[~active] = 0  # Future-only IDs are inert, isolated placeholders, never model inputs.
    artist_x = np.column_stack(
        [
            np.log1p(artist_active_tracks),
            np.log1p(artist_recent),
            np.log1p(artist_counts),
            np.log1p(artist_streams),
            np.full(n_artists, coverage.mean()),
        ]
    ).astype(np.float32)
    country_x = np.column_stack(
        [
            np.log1p(seen.sum(axis=0)),
            np.log1p(recent.sum(axis=0)),
            np.log1p(recent_days.sum(axis=0)),
            np.log1p(recent_streams.sum(axis=0)),
            coverage,
        ]
    ).astype(np.float32)
    edge_index, edge_type = graph_edges(active, artist_song, recent, seen, n_songs, n_artists)
    songs, countries, pairs, labels, gaps = [], [], [], [], []
    skipped = []
    for country, history in enumerate(histories):
        if not history.complete_window(day):
            skipped.append(country)
            continue
        ids = np.flatnonzero(eligible(history.first, history.last, day))
        if not len(ids):
            continue
        pair = np.column_stack(
            [
                np.log1p(day - history.first[ids]),
                np.log1p(day - history.last[ids]),
                np.log1p(history.count[ids]),
                history.best[ids] / 200,
                history.last_rank[ids] / 200,
                np.log1p(history.last_streams[ids]),
                np.log1p(history.window(day - 89, day)[ids]),
                np.log1p(history.window(day - 179, day)[ids]),
                recent[ids].sum(axis=1) - recent[ids, country],
                np.log1p(recent_days[ids].sum(axis=1) - recent_days[ids, country]),
                np.log1p(recent_streams[ids].sum(axis=1) - recent_streams[ids, country]),
                np.log1p(artist_other[ids]),
                np.log1p(collaborator_activity[ids]),
                np.full(len(ids), seasonal[0]),
                np.full(len(ids), seasonal[1]),
                np.full(len(ids), coverage[country]),
                np.full(len(ids), coverage.mean()),
            ]
        )
        songs.append(ids)
        countries.append(np.full(len(ids), country, dtype=np.int64))
        pairs.append(pair.astype(np.float32))
        labels.append((history.label_counts(day)[ids] >= 3).astype(np.float32))
        gaps.append((day - history.last[ids]).astype(np.int32))
    if not songs:
        return None, skipped
    result = {
        "song_x": song_x,
        "artist_x": artist_x,
        "country_x": country_x,
        "song_active": active,
        "artist_active": artist_active_tracks > 0,
        "edge_index": edge_index,
        "edge_type": edge_type,
        "song": np.concatenate(songs),
        "country": np.concatenate(countries),
        "pair_x": np.concatenate(pairs),
        "y": np.concatenate(labels),
        "gap": np.concatenate(gaps),
        "day": np.array(day, dtype=np.int64),
    }
    return result, skipped


def prepare(raw_dir, output, markets):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError("Choose a fresh output directory; prepared data already exists")
    charts, coverage, quality = [], [], {}
    for market in markets:
        frame, complete, info = load_chart(
            raw_dir / f"spotify_charts_regional_{market}.csv", VALID_END - 1
        )
        charts.append(frame)
        coverage.append(complete)
        quality[market] = info
        print(f"Loaded {market}: {info}", flush=True)
    song_ids = sorted(set().union(*(set(frame.ID) for frame in charts)))
    song_map = {song: i for i, song in enumerate(song_ids)}
    metadata = pd.read_csv(
        raw_dir / "spotify_hit_songs_dataset_complete.csv",
        sep="\t",
        usecols=["song_id", "artist_id", *AUDIO],
    ).drop_duplicates("song_id")
    metadata = metadata.set_index("song_id").reindex(song_ids)
    credits = metadata.artist_id.map(parse_artists)
    artist_ids = sorted(set().union(*credits))
    artist_map = {artist: i for i, artist in enumerate(artist_ids)}
    rows, cols = [], []
    for song, artists in enumerate(credits):
        for artist in artists:
            rows.append(artist_map[artist])
            cols.append(song)
    artist_song = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(artist_ids), len(song_ids))
    )
    audio = metadata[AUDIO].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    missing = ~np.isfinite(audio)
    audio[missing] = 0
    audio = np.column_stack([audio, missing.astype(np.float32)])
    histories = [
        ChartHistory(frame, complete, song_map) for frame, complete in zip(charts, coverage)
    ]
    manifest = {
        "markets": markets,
        "song_ids": song_ids,
        "artist_ids": artist_ids,
        "relations": RELATIONS,
        "pair_features": PAIR_FEATURES,
        "audio_features": AUDIO,
        "snapshots": [],
    }
    summary = {
        "quality": quality,
        "songs_through_2020": len(song_ids),
        "artists_through_2020": len(artist_ids),
        "countries": len(markets),
        "songs_missing_artist_metadata": int(credits.map(len).eq(0).sum()),
        "songs_with_missing_audio": int(missing.any(axis=1).sum()),
        "splits": {},
        "skipped_incomplete_country_weeks": 0,
        "test_period": "2021-2022 reserved; not prepared or evaluated",
    }
    for day in range(182, VALID_END - 28, 7):  # 2017-07-02 is the first eligible Sunday.
        split = split_for_day(day)
        if split is None:
            continue
        snapshot, skipped = build_snapshot(day, histories, artist_song, audio)
        summary["skipped_incomplete_country_weeks"] += len(skipped)
        if snapshot is None:
            continue
        filename = f"{split}/{date_string(day)}.npz"
        (output / split).mkdir(exist_ok=True)
        np.savez_compressed(output / filename, **snapshot)
        info = {
            "file": filename,
            "day": day,
            "date": date_string(day),
            "split": split,
            "examples": len(snapshot["y"]),
            "positives": int(snapshot["y"].sum()),
            "active_songs": int(snapshot["song_active"].sum()),
            "active_artists": int(snapshot["artist_active"].sum()),
            "directed_edges": snapshot["edge_index"].shape[1],
            "country_weeks": len(np.unique(snapshot["country"])),
        }
        manifest["snapshots"].append(info)
        if len(manifest["snapshots"]) % 20 == 0:
            print(f"Prepared {date_string(day)}: {info['examples']} candidates", flush=True)
    for split in ("train", "validation"):
        items = [s for s in manifest["snapshots"] if s["split"] == split]
        n, positive = sum(s["examples"] for s in items), sum(s["positives"] for s in items)
        summary["splits"][split] = {
            "examples": n,
            "positives": positive,
            "prevalence": positive / n,
            "snapshots": len(items),
            "country_weeks": sum(s["country_weeks"] for s in items),
            "first_forecast": items[0]["date"],
            "last_forecast": items[-1]["date"],
        }
    summary["last_snapshot"] = manifest["snapshots"][-1]
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--markets", nargs="+", default=DEFAULT_MARKETS)
    args = parser.parse_args()
    prepare(args.raw, args.output, args.markets)


if __name__ == "__main__":
    main()
