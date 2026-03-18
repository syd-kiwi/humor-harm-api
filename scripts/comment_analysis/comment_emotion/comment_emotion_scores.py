"""Predict emotion labels (feelings) for comments using an emotion model.

Default model:
- j-hartmann/emotion-english-distilroberta-base
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import pipeline


DEFAULT_MODEL = "j-hartmann/emotion-english-distilroberta-base"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        help="Input CSV with a comment text column",
    )
    parser.add_argument(
        "--comments-dir",
        help="Directory containing per-video JSON comment exports",
    )
    parser.add_argument("--output", default="scripts/comment_analysis/comment_emotion_scores.csv")
    parser.add_argument("--text-col", default="comment_text")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    if not args.input and not args.comments_dir:
        parser.error("Provide either --input (CSV) or --comments-dir (JSON exports).")
    if args.input and args.comments_dir:
        parser.error("Use only one input source: --input or --comments-dir.")

    return args


def load_comments_from_json(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    lines = raw.splitlines()
    if len(lines) > 1 and lines[0].lstrip().startswith("{"):
        comments = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                comments.append(obj)
        if comments:
            return comments

    obj = json.loads(raw)
    if isinstance(obj, list):
        return [item for item in obj if isinstance(item, dict)]
    if isinstance(obj, dict):
        return [obj]
    return []


def load_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    if args.input:
        return pd.read_csv(args.input)

    rows: list[dict[str, str]] = []
    comments_dir = Path(args.comments_dir)
    for json_path in sorted(comments_dir.glob("*.json")):
        video_id = json_path.stem
        for comment in load_comments_from_json(json_path):
            text = str(comment.get("text", "")).strip()
            if not text:
                continue
            rows.append(
                {
                    "video_id": video_id,
                    "source_file": str(json_path),
                    args.text_col: text,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    df = load_dataframe(args)
    if args.text_col not in df.columns:
        raise SystemExit(f"Missing text column '{args.text_col}'. Available: {list(df.columns)}")

    df[args.text_col] = df[args.text_col].fillna("").astype(str).str.strip()
    df = df[df[args.text_col] != ""].copy()

    if df.empty:
        raise SystemExit("No non-empty comments found after filtering.")

    clf = pipeline(
        "text-classification",
        model=args.model,
        return_all_scores=True,
        truncation=True,
    )

    texts = df[args.text_col].tolist()
    outputs = clf(texts, batch_size=args.batch_size, truncation=True, max_length=512)

    top_labels = []
    top_scores = []

    for i, candidates in enumerate(outputs):
        best = max(candidates, key=lambda x: float(x["score"]))
        top_labels.append(best["label"])
        top_scores.append(float(best["score"]))

        row_index = df.index[i]
        for c in candidates:
            label = str(c["label"]).strip().lower().replace(" ", "_")
            df.at[row_index, f"emotion_{label}"] = float(c["score"])

    df["emotion_label"] = top_labels
    df["emotion_score"] = top_scores

    summary = (
        df.groupby("emotion_label", dropna=False)
        .agg(
            comments=(args.text_col, "count"),
            mean_confidence=("emotion_score", "mean"),
        )
        .reset_index()
        .sort_values("comments", ascending=False)
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    summary_path = out.with_name(f"{out.stem}_summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"Wrote comment-level emotion scores: {out}")
    print(f"Wrote emotion summary: {summary_path}")


if __name__ == "__main__":
    main()
