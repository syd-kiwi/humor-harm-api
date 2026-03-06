"""Score toxicity on comments and aggregate by humor vs dark humor buckets.

Expected input columns:
- text column (default: comment_text)
- label column that marks humor type (default: humor_type)

Humor labels are normalized into two buckets:
- humor
- dark_humor
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from detoxify import Detoxify


HUMOR_ALIASES = {
    "humor": "humor",
    "funny": "humor",
    "joke": "humor",
    "light": "humor",
    "dark": "dark_humor",
    "dark_humor": "dark_humor",
    "dark humor": "dark_humor",
    "black humor": "dark_humor",
    "gallows": "dark_humor",
}


def normalize_humor_label(value: str) -> str:
    key = str(value or "").strip().lower()
    return HUMOR_ALIASES.get(key, "unknown")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input CSV with comment text and humor labels")
    parser.add_argument("--out-comment", default="scripts/comment_analysis/comment_toxicity_with_humor.csv")
    parser.add_argument("--out-summary", default="scripts/comment_analysis/humor_dark_toxicity_summary.csv")
    parser.add_argument("--text-col", default="comment_text")
    parser.add_argument("--label-col", default="humor_type")
    parser.add_argument("--model", default="original", help="Detoxify model: original, unbiased, or multilingual")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input)
    if args.text_col not in df.columns:
        raise SystemExit(f"Missing text column '{args.text_col}'. Available: {list(df.columns)}")

    if args.label_col not in df.columns:
        df[args.label_col] = "unknown"

    df["humor_bucket"] = df[args.label_col].map(normalize_humor_label)
    df[args.text_col] = df[args.text_col].fillna("").astype(str).str.strip()
    df = df[df[args.text_col] != ""].copy()

    if df.empty:
        raise SystemExit("No non-empty comments found after filtering.")

    model = Detoxify(args.model)
    scores = model.predict(df[args.text_col].tolist())

    df["toxicity"] = scores["toxicity"]
    for key, values in scores.items():
        if key != "toxicity":
            df[key] = values

    summary = (
        df.groupby("humor_bucket", dropna=False)
        .agg(
            comments=(args.text_col, "count"),
            toxicity_mean=("toxicity", "mean"),
            toxicity_median=("toxicity", "median"),
            toxicity_max=("toxicity", "max"),
        )
        .reset_index()
        .sort_values(["humor_bucket"])
    )

    Path(args.out_comment).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(args.out_comment, index=False)
    summary.to_csv(args.out_summary, index=False)

    print(f"Wrote comment-level scores: {args.out_comment}")
    print(f"Wrote humor-bucket summary: {args.out_summary}")


if __name__ == "__main__":
    main()
