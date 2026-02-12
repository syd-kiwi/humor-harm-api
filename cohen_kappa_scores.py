#!/usr/bin/env python3
"""Compute Cohen's Kappa for first 50 rows and all rows from two score columns in a CSV."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute Cohen's Kappa using two score columns and print only two outputs: "
            "first 50 rows and all rows."
        )
    )
    p.add_argument("csv_path", help="Path to CSV file containing two score columns")
    p.add_argument("rater1", help="Column name for rater 1 scores")
    p.add_argument("rater2", help="Column name for rater 2 scores")
    return p.parse_args()


def cohen_kappa(values1: list[str], values2: list[str]) -> float:
    if len(values1) != len(values2):
        raise ValueError("Both rater score lists must be the same length.")
    n = len(values1)
    if n == 0:
        raise ValueError("No rows available to score.")

    agree = sum(1 for a, b in zip(values1, values2) if a == b)
    p_observed = agree / n

    counts1 = Counter(values1)
    counts2 = Counter(values2)
    categories = set(counts1) | set(counts2)
    p_expected = sum((counts1[c] / n) * (counts2[c] / n) for c in categories)

    if math.isclose(1.0 - p_expected, 0.0):
        return 1.0 if math.isclose(p_observed, 1.0) else 0.0

    return (p_observed - p_expected) / (1.0 - p_expected)


def load_scores(csv_path: str, rater1_col: str, rater2_col: str) -> tuple[list[str], list[str]]:
    s1: list[str] = []
    s2: list[str] = []

    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError("CSV is missing a header row.")
        missing = [c for c in [rater1_col, rater2_col] if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required column(s): {', '.join(missing)}")

        for row in reader:
            v1 = (row[rater1_col] or "").strip()
            v2 = (row[rater2_col] or "").strip()
            if v1 and v2:
                s1.append(v1)
                s2.append(v2)

    if not s1:
        raise ValueError("No non-empty paired scores found.")

    return s1, s2


def main() -> None:
    args = parse_args()
    scores1, scores2 = load_scores(args.csv_path, args.rater1, args.rater2)

    first_n = min(50, len(scores1))
    kappa_first_50 = cohen_kappa(scores1[:first_n], scores2[:first_n])
    kappa_all = cohen_kappa(scores1, scores2)

    print(f"Cohen's Kappa (first {first_n}): {kappa_first_50:.6f}")
    print(f"Cohen's Kappa (all {len(scores1)}): {kappa_all:.6f}")


if __name__ == "__main__":
    main()
