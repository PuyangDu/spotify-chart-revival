"""Download the versioned MGD+ archive and verify its publisher checksum."""

import argparse
import hashlib
import json
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

SOURCE = "https://zenodo.org/records/8086643"
URL = "https://zenodo.org/api/records/8086643/files/MGDplus.zip/content"
MD5 = "4e06a42f266f08d84e9cf6045c4ae3c3"
DEFAULT_MARKETS = ["us", "gb", "ca", "br"]


def checksum(path, algorithm="md5"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(destination, markets):
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "MGDplus.zip"
    if not archive.exists() or checksum(archive) != MD5:
        partial = archive.with_suffix(".part")
        for attempt in range(3):
            try:
                request = urllib.request.Request(URL, headers={"User-Agent": "CS224W-research/0.1"})
                with (
                    urllib.request.urlopen(request, timeout=60) as response,
                    partial.open("wb") as out,
                ):
                    shutil.copyfileobj(response, out, length=1024 * 1024)
                if checksum(partial) != MD5:
                    raise ValueError("MGD+ archive does not match the publisher's MD5 checksum")
                partial.replace(archive)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2**attempt)
    records = []
    with zipfile.ZipFile(archive) as bundle:
        charts = [
            name
            for name in bundle.namelist()
            if name.startswith("charts/regional/spotify_charts_regional_") and name.endswith(".csv")
        ]
        if markets == ["all"]:
            selected = charts
        else:
            selected = [f"charts/regional/spotify_charts_regional_{m}.csv" for m in markets]
        selected += [
            "songs/spotify_hit_songs_dataset_complete.csv",
            "artists/spotify_artists_info_complete.csv",
        ]
        for member in selected:
            # ZipFile validates each member's CRC; flatten only explicitly selected files.
            target = destination / Path(member).name
            with bundle.open(member) as source, target.open("wb") as out:
                shutil.copyfileobj(source, out)
            records.append(
                {
                    "member": member,
                    "bytes": target.stat().st_size,
                    "sha256": checksum(target, "sha256"),
                }
            )
            print(f"Verified {target.name}", flush=True)
    manifest = {
        "source": SOURCE,
        "download_url": URL,
        "archive_md5": MD5,
        "archive_bytes": archive.stat().st_size,
        "files": records,
    }
    (destination / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--markets", nargs="+", default=DEFAULT_MARKETS)
    args = parser.parse_args()
    download(args.output, args.markets)


if __name__ == "__main__":
    main()
