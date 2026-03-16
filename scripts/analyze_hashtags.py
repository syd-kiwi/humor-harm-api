#!/usr/bin/env python3
"""Analyze hashtag usage in a CSV dataset of videos."""

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

HASHTAG_PATTERN = re.compile(r"(?<!\w)#([A-Za-z0-9_]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count the most popular hashtags in a CSV dataset. "
            "By default this scans description, tags, and title columns."
        )
    )
    parser.add_argument("csv_path", help="Path to dataset CSV file")
    parser.add_argument(
        "--columns",
        nargs="+",
        default=["description", "tags", "title"],
        help="Columns to scan for hashtags (default: description tags title)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=25,
        help="How many top hashtags to print (default: 25)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output CSV for hashtag counts",
    )
    return parser.parse_args()


def extract_hashtags(text: str) -> list[str]:
    return [m.lower() for m in HASHTAG_PATTERN.findall(text or "")]


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    hashtag_counts = Counter()
    hashtag_video_counts = Counter()

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing_columns = [c for c in args.columns if c not in (reader.fieldnames or [])]
        if missing_columns:
            raise ValueError(
                f"Missing column(s) in dataset: {', '.join(missing_columns)}. "
                f"Available columns: {', '.join(reader.fieldnames or [])}"
            )

        total_rows = 0
        rows_with_hashtags = 0

        for row in reader:
            total_rows += 1

            row_hashtags = []
            for col in args.columns:
                row_hashtags.extend(extract_hashtags(row.get(col, "")))

            if row_hashtags:
                rows_with_hashtags += 1

            hashtag_counts.update(row_hashtags)
            hashtag_video_counts.update(set(row_hashtags))

    print(f"Rows scanned: {total_rows}")
    print(f"Rows with >=1 hashtag: {rows_with_hashtags}")
    print(f"Unique hashtags found: {len(hashtag_counts)}\n")

    print(f"Top {args.top} hashtags by total mentions:")
    print("-" * 60)
    print(f"{'hashtag':<30} {'mentions':>10} {'videos':>10}")
    print("-" * 60)
    for hashtag, mentions in hashtag_counts.most_common(args.top):
        videos = hashtag_video_counts[hashtag]
        print(f"#{hashtag:<29} {mentions:>10} {videos:>10}")

    if args.output:
        out_path = Path(args.output)
        with out_path.open("w", encoding="utf-8", newline="") as out_f:
            writer = csv.writer(out_f)
            writer.writerow(["hashtag", "mentions", "videos"])
            for hashtag, mentions in hashtag_counts.most_common():
                writer.writerow([f"#{hashtag}", mentions, hashtag_video_counts[hashtag]])
        print(f"\nSaved full hashtag counts to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
