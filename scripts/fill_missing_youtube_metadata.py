#!/usr/bin/env python3
"""Backfill missing YouTube metadata in a CSV using yt-dlp and video_id."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile


FIELDS_TO_FILL = [
    "url",
    "channel",
    "title",
    "uploader_id",
    "uploader",
    "channel_id",
    "upload_date",
    "view_count",
    "duration",
    "categories",
]

YTDLP_PRINT_FIELDS = [
    ("id", "%(id)s"),
    ("url", "%(webpage_url)s"),
    ("channel", "%(channel)s"),
    ("title", "%(title)s"),
    ("uploader_id", "%(uploader_id)s"),
    ("uploader", "%(uploader)s"),
    ("channel_id", "%(channel_id)s"),
    ("upload_date", "%(upload_date)s"),
    ("view_count", "%(view_count)s"),
    ("duration", "%(duration)s"),
    ("categories", "%(categories)s"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fill missing YouTube metadata in a CSV by looking up rows with a "
            "video_id and blank metadata columns using yt-dlp."
        )
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="unified_dataset.csv",
        help="Path to the dataset CSV. Defaults to unified_dataset.csv.",
    )
    parser.add_argument(
        "--video-id-column",
        default="video_id",
        help="Column that stores the YouTube video ID. Defaults to video_id.",
    )
    parser.add_argument(
        "--write-in-place",
        action="store_true",
        help="Overwrite the input CSV in place. Otherwise writes a *_filled.csv copy.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only fetch metadata for the first N matching rows.",
    )
    return parser.parse_args()


def ensure_yt_dlp() -> str:
    yt_dlp_path = shutil.which("yt-dlp")
    if yt_dlp_path:
        return yt_dlp_path
    raise SystemExit(
        "yt-dlp is not installed or not on PATH. Install it first, then rerun "
        "this script."
    )


def is_blank(value: str | None) -> bool:
    return value is None or str(value).strip() == ""


def categories_to_string(raw_value: str) -> str:
    value = raw_value.strip()
    if not value or value == "NA":
        return ""

    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return ""
        parts = []
        for chunk in inner.split(","):
            cleaned = chunk.strip().strip("'").strip('"')
            if cleaned:
                parts.append(cleaned)
        return "|".join(parts)
    return value


def fetch_metadata(video_id: str, yt_dlp_path: str) -> dict[str, str]:
    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [yt_dlp_path, "--skip-download", "--no-warnings"]
    for _, template in YTDLP_PRINT_FIELDS:
        cmd.extend(["--print", template])
    cmd.append(url)

    completed = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip() or "unknown yt-dlp error"
        raise RuntimeError(f"yt-dlp failed for {video_id}: {stderr}")

    lines = [line.rstrip("\n") for line in completed.stdout.splitlines()]
    if len(lines) < len(YTDLP_PRINT_FIELDS):
        raise RuntimeError(
            f"yt-dlp returned {len(lines)} lines for {video_id}, "
            f"expected {len(YTDLP_PRINT_FIELDS)}."
        )

    data = {
        name: value
        for (name, _), value in zip(YTDLP_PRINT_FIELDS, lines[: len(YTDLP_PRINT_FIELDS)])
    }
    data["categories"] = categories_to_string(data.get("categories", ""))
    return data


def output_path_for(csv_path: Path, write_in_place: bool) -> Path:
    if write_in_place:
        return csv_path
    return csv_path.with_name(f"{csv_path.stem}_filled{csv_path.suffix}")


def main() -> int:
    args = parse_args()
    yt_dlp_path = ensure_yt_dlp()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if args.video_id_column not in fieldnames:
        raise SystemExit(
            f"Missing video ID column {args.video_id_column!r}. Found: {fieldnames}"
        )

    missing_columns = [field for field in FIELDS_TO_FILL if field not in fieldnames]
    if missing_columns:
        raise SystemExit(
            f"Missing required metadata columns in {csv_path}: {missing_columns}"
        )

    video_ids_to_fetch: list[str] = []
    for row in rows:
        video_id = (row.get(args.video_id_column) or "").strip()
        if not video_id:
            continue
        if any(is_blank(row.get(field)) for field in FIELDS_TO_FILL) and video_id not in video_ids_to_fetch:
            video_ids_to_fetch.append(video_id)

    if args.limit is not None:
        video_ids_to_fetch = video_ids_to_fetch[: args.limit]

    print(f"rows in dataset: {len(rows)}")
    print(f"video IDs to backfill: {len(video_ids_to_fetch)}")

    fetched_by_video_id: dict[str, dict[str, str]] = {}
    failures: list[tuple[str, str]] = []

    for index, video_id in enumerate(video_ids_to_fetch, start=1):
        try:
            fetched_by_video_id[video_id] = fetch_metadata(video_id, yt_dlp_path)
            print(f"[{index}/{len(video_ids_to_fetch)}] fetched {video_id}")
        except Exception as exc:  # noqa: BLE001
            failures.append((video_id, str(exc)))
            print(f"[{index}/{len(video_ids_to_fetch)}] failed {video_id}: {exc}", file=sys.stderr)

    updates = 0
    for row in rows:
        video_id = (row.get(args.video_id_column) or "").strip()
        fetched = fetched_by_video_id.get(video_id)
        if not fetched:
            continue
        for field in FIELDS_TO_FILL:
            if is_blank(row.get(field)):
                row[field] = fetched.get(field, "")
                updates += 1

    destination = output_path_for(csv_path, args.write_in_place)
    if args.write_in_place:
        with NamedTemporaryFile("w", newline="", encoding="utf-8", delete=False, dir=csv_path.parent) as tmp:
            writer = csv.DictWriter(tmp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            temp_name = tmp.name
        os.replace(temp_name, csv_path)
    else:
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"metadata fields updated: {updates}")
    print(f"output written to: {destination}")

    if failures:
        print(f"video IDs still missing metadata: {len(failures)}", file=sys.stderr)
        for video_id, error in failures:
            print(f"  - {video_id}: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
