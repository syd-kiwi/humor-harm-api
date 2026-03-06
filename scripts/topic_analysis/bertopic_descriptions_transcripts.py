"""Run BERTopic on combined video descriptions + transcripts.

Input CSV should include:
- description column (default: description)
- transcript column (default: transcript_text)
Optional:
- id column (default: video_id)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from bertopic import BERTopic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input CSV with description and transcript columns")
    parser.add_argument("--output", default="scripts/topic_analysis/bertopic_description_transcript_topics.csv")
    parser.add_argument("--topic-info-output", default="scripts/topic_analysis/bertopic_topic_info.csv")
    parser.add_argument("--id-col", default="video_id")
    parser.add_argument("--description-col", default="description")
    parser.add_argument("--transcript-col", default="transcript_text")
    parser.add_argument("--min-topic-size", type=int, default=10)
    return parser.parse_args()


def merge_text(description: str, transcript: str) -> str:
    d = str(description or "").strip()
    t = str(transcript or "").strip()
    if d and t:
        return f"description: {d}\n\ntranscript: {t}"
    return d or t


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input)
    required = [args.description_col, args.transcript_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}. Available: {list(df.columns)}")

    if args.id_col not in df.columns:
        df[args.id_col] = [f"row_{i}" for i in range(len(df))]

    docs = [
        merge_text(desc, transcript)
        for desc, transcript in zip(df[args.description_col].fillna(""), df[args.transcript_col].fillna(""))
    ]

    work = df.copy()
    work["document_text"] = docs
    work = work[work["document_text"].str.strip() != ""].copy()

    if work.empty:
        raise SystemExit("No rows with usable description/transcript text.")

    topic_model = BERTopic(min_topic_size=args.min_topic_size, calculate_probabilities=True)
    topics, probabilities = topic_model.fit_transform(work["document_text"].tolist())

    work["topic_id"] = topics
    work["topic_probability"] = [max(p) if p is not None else None for p in probabilities]
    work["topic_name"] = [topic_model.topic_labels_.get(topic_id, str(topic_id)) for topic_id in topics]

    output_cols = [args.id_col, args.description_col, args.transcript_col, "topic_id", "topic_probability", "topic_name"]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    work[output_cols].to_csv(out, index=False)

    topic_info = topic_model.get_topic_info()
    info_out = Path(args.topic_info_output)
    info_out.parent.mkdir(parents=True, exist_ok=True)
    topic_info.to_csv(info_out, index=False)

    print(f"Wrote document-topic assignments: {out}")
    print(f"Wrote topic metadata: {info_out}")


if __name__ == "__main__":
    main()
