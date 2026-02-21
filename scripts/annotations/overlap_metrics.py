#!/usr/bin/env python3

import argparse
import json
import math
import re
from itertools import combinations
from pathlib import Path
from collections import Counter

import pandas as pd


def parse_multi_choice(cell):
    """
    Returns a set of topics for one annotation.
    Handles JSON dict like {"choices":[...]} and plain strings like "Work, Money".
    """
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return set()

    s = str(cell).strip()
    if not s:
        return set()

    lowered = s.lower()

    if lowered.startswith("{") or lowered.startswith("["):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                choices = obj.get("choices", [])
            elif isinstance(obj, list):
                choices = obj
            else:
                choices = []
            return set(str(c).strip() for c in choices if c)
        except Exception:
            pass

    parts = re.split(r"[;,|/]+", s)
    return set(p.strip() for p in parts if p.strip())


def jaccard(a, b):
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def compute_item_metrics(topic_sets):
    """
    topic_sets: list of sets, one per annotator for this item
    """
    n = len(topic_sets)
    if n < 2:
        return None

    pairs = list(combinations(topic_sets, 2))
    pairwise_overlap = [1.0 if (A & B) else 0.0 for A, B in pairs]
    pairwise_j = [jaccard(A, B) for A, B in pairs]

    pairwise_overlap_rate = sum(pairwise_overlap) / len(pairwise_overlap)
    pairwise_jaccard_mean = sum(pairwise_j) / len(pairwise_j)

    # Any topic chosen by at least 2 annotators
    counts = Counter()
    for s in topic_sets:
        for t in s:
            counts[t] += 1
    any_topic_shared_by_2plus = any(v >= 2 for v in counts.values())

    # Any topic chosen by all annotators
    if topic_sets:
        intersection_all = set.intersection(*topic_sets)
    else:
        intersection_all = set()
    any_topic_shared_by_all = len(intersection_all) > 0

    # Optional: size stats
    mean_topics_per_annotator = sum(len(s) for s in topic_sets) / n

    return {
        "n_annotators": n,
        "pairwise_overlap_rate": round(pairwise_overlap_rate, 6),
        "pairwise_jaccard_mean": round(pairwise_jaccard_mean, 6),
        "any_topic_shared_by_2plus": bool(any_topic_shared_by_2plus),
        "any_topic_shared_by_all": bool(any_topic_shared_by_all),
        "mean_topics_per_annotator": round(mean_topics_per_annotator, 6),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--item_col", default="id")
    ap.add_argument("--annotator_col", default="annotator")
    ap.add_argument("--topic_col", default="joke_topic")
    args = ap.parse_args()

    csv_path = Path(args.csv_path)
    df = pd.read_csv(csv_path)

    for c in [args.item_col, args.annotator_col, args.topic_col]:
        if c not in df.columns:
            raise SystemExit(f"Missing column: {c}. Columns are: {list(df.columns)}")

    per_item_rows = []
    grouped = df.groupby(args.item_col)

    for item_id, g in grouped:
        # One row per annotator per item
        g2 = g.drop_duplicates(subset=[args.annotator_col]).copy()
        topic_sets = [parse_multi_choice(x) for x in g2[args.topic_col].tolist()]

        metrics = compute_item_metrics(topic_sets)
        if metrics is None:
            continue

        row = {"id": item_id}
        row.update(metrics)
        per_item_rows.append(row)

    per_item = pd.DataFrame(per_item_rows)

    per_item_out = csv_path.with_name(f"{csv_path.stem}_topic_overlap_per_item.csv")
    per_item.to_csv(per_item_out, index=False)

    # Summary metrics
    if len(per_item) == 0:
        raise SystemExit("No items with 2 or more annotators found for topic overlap metrics.")

    summary = {
        "n_items_used": int(len(per_item)),
        "mean_pairwise_overlap_rate": float(per_item["pairwise_overlap_rate"].mean()),
        "median_pairwise_overlap_rate": float(per_item["pairwise_overlap_rate"].median()),
        "mean_pairwise_jaccard": float(per_item["pairwise_jaccard_mean"].mean()),
        "median_pairwise_jaccard": float(per_item["pairwise_jaccard_mean"].median()),
        "pct_any_topic_shared_by_2plus": float(per_item["any_topic_shared_by_2plus"].mean()),
        "pct_any_topic_shared_by_all": float(per_item["any_topic_shared_by_all"].mean()),
        "mean_topics_per_annotator": float(per_item["mean_topics_per_annotator"].mean()),
        "mean_annotators_per_item": float(per_item["n_annotators"].mean()),
    }

    summary_df = pd.DataFrame([summary])
    summary_out = csv_path.with_name(f"{csv_path.stem}_topic_overlap_summary.csv")
    summary_df.to_csv(summary_out, index=False)

    print(f"Wrote {per_item_out}")
    print(f"Wrote {summary_out}")


if __name__ == "__main__":
    main()