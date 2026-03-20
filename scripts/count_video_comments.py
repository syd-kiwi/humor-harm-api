#!/usr/bin/env python3
"""Count downloaded comment records for videos listed in a dataset CSV.

The repository's `comments/*.json` files are newline-delimited JSON (one comment per
line), so this script counts valid JSON objects line-by-line and matches them to the
`video_id` values in a dataset such as `unified_dataset.csv`.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

DEFAULT_DATASET = Path("/home/kiwi-pandas/Documents/humor-harm-api/unified_dataset.csv")
DEFAULT_COMMENTS_DIR = Path("/home/kiwi-pandas/Documents/humor-harm-api/comments")
DEFAULT_OUTPUT = Path("/home/kiwi-pandas/Documents/humor-harm-api/comment_counts_by_video.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count downloaded comments for each video in a dataset.",
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default=DEFAULT_DATASET,
        type=Path,
        help="Dataset CSV containing a video_id column (default: unified_dataset.csv).",
    )
    parser.add_argument(
        "--comments-dir",
        default=DEFAULT_COMMENTS_DIR,
        type=Path,
        help="Directory containing newline-delimited JSON comment files.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        type=Path,
        help="Where to write the per-video comment counts CSV.",
    )
    parser.add_argument(
        "--skip-output",
        action="store_true",
        help="Print the summary only and do not write a CSV.",
    )
    return parser.parse_args()


def count_comment_file(path: Path) -> tuple[int, int]:
    """Return (valid_comment_count, invalid_line_count) for one NDJSON file."""
    valid_lines = 0
    invalid_lines = 0

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                invalid_lines += 1
                continue
            if isinstance(payload, dict):
                valid_lines += 1
            else:
                invalid_lines += 1

    return valid_lines, invalid_lines


def load_dataset_rows(dataset_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with dataset_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if "video_id" not in fieldnames:
            raise SystemExit(
                f"Missing required 'video_id' column in dataset: {dataset_path}"
            )
        rows = list(reader)
    return rows, fieldnames


def build_output_rows(
    dataset_rows: Iterable[dict[str, str]], comments_dir: Path
) -> tuple[list[dict[str, str | int]], int]:
    output_rows: list[dict[str, str | int]] = []
    total_invalid_lines = 0

    for row in dataset_rows:
        video_id = (row.get("video_id") or "").strip()
        if not video_id:
            continue

        comment_path = comments_dir / f"{video_id}.json"
        count = 0
        invalid_lines = 0
        has_file = comment_path.exists()
        if has_file:
            count, invalid_lines = count_comment_file(comment_path)
            total_invalid_lines += invalid_lines

        output_row: dict[str, str | int] = {
            "id": (row.get("id") or "").strip(),
            "video_id": video_id,
            "title": (row.get("title") or "").strip(),
            "url": (row.get("url") or "").strip(),
            "downloaded_comment_count": count,
            "comment_file_found": "yes" if has_file else "no",
            "invalid_comment_lines": invalid_lines,
        }
        output_rows.append(output_row)

    output_rows.sort(
        key=lambda row: (
            int(row["downloaded_comment_count"]),
            str(row["video_id"]),
        ),
        reverse=True,
    )
    return output_rows, total_invalid_lines


def write_output_csv(output_path: Path, rows: list[dict[str, str | int]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "video_id",
                "title",
                "url",
                "downloaded_comment_count",
                "comment_file_found",
                "invalid_comment_lines",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, str | int]], total_invalid_lines: int) -> None:
    videos_with_comment_files = sum(1 for row in rows if row["comment_file_found"] == "yes")
    total_videos = len(rows)
    total_comments = sum(int(row["downloaded_comment_count"]) for row in rows)
    zero_comment_videos = sum(
        1 for row in rows if int(row["downloaded_comment_count"]) == 0
    )

    print(f"Dataset videos checked: {total_videos}")
    print(f"Videos with comment files: {videos_with_comment_files}")
    print(f"Videos with zero downloaded comments: {zero_comment_videos}")
    print(f"Total downloaded comments counted: {total_comments}")
    print(f"Invalid comment lines skipped: {total_invalid_lines}")

    if rows:
        top_row = rows[0]
        print(
            "Most comments in dataset: "
            f"{top_row['video_id']} ({top_row['downloaded_comment_count']})"
        )


if __name__ == "__main__":
    args = parse_args()
    dataset_rows, _fieldnames = load_dataset_rows(args.dataset)
    output_rows, total_invalid_lines = build_output_rows(dataset_rows, args.comments_dir)
    print_summary(output_rows, total_invalid_lines)

    if not args.skip_output:
        write_output_csv(args.output, output_rows)
        print(f"Wrote: {args.output}")
