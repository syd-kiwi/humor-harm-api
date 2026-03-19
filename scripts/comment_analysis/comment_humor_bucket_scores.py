#!/usr/bin/env python3
"""Compute comment sentiment, toxicity, and emotion scores by humor bucket.

This script joins the 1,211-row `unified_dataset.csv` with scraped YouTube comments
stored as JSON/JSONL files in `comments/`. It scores each non-empty comment with:

- sentiment probabilities (negative / neutral / positive)
- toxicity probabilities from Detoxify
- emotion probabilities from a Hugging Face emotion classifier

The script writes:
- a comment-level CSV with one row per scored comment
- a video-level CSV with mean scores per video
- a bucket-level CSV with comment-weighted averages for not humor / regular humor /
  dark humor
- a bucket-level CSV with video-weighted averages (mean of per-video means)

Example:
    python scripts/comment_analysis/comment_humor_bucket_scores.py \
        --dataset unified_dataset.csv \
        --comments-dir comments \
        --output-dir scripts/comment_analysis/output
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple


DEFAULT_SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
DEFAULT_EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"

COMMENT_FIELDS = [
    "video_id",
    "dataset_row_id",
    "humor_bucket",
    "comment_index",
    "comment_text",
    "sentiment_label",
    "sentiment_score",
    "sentiment_negative",
    "sentiment_neutral",
    "sentiment_positive",
    "emotion_label",
    "emotion_score",
]

BUCKETS = ("not_humor", "regular_humor", "dark_humor")


class MissingDependencyError(RuntimeError):
    """Raised when an optional scoring dependency is unavailable."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="/home/kiwi-pandas/Documents/humor-harm-api/unified_dataset.csv", help="Path to the 1211-row dataset CSV")
    parser.add_argument("--comments-dir", default="/home/kiwi-pandas/Documents/humor-harm-api/comments", help="Directory containing comment JSON/JSONL files")
    parser.add_argument(
        "--output-dir",
        default="scripts/comment_analysis/output",
        help="Directory where CSV outputs will be written",
    )
    parser.add_argument(
        "--sentiment-model",
        default=DEFAULT_SENTIMENT_MODEL,
        help="Hugging Face model id for sentiment classification",
    )
    parser.add_argument(
        "--emotion-model",
        default=DEFAULT_EMOTION_MODEL,
        help="Hugging Face model id for emotion classification",
    )
    parser.add_argument(
        "--toxicity-model",
        default="original",
        help="Detoxify model name: original, unbiased, or multilingual",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for model inference")
    parser.add_argument(
        "--max-comments-per-video",
        type=int,
        default=None,
        help="Optional cap on comments scored per video after filtering blank comments",
    )
    parser.add_argument(
        "--include-unknown",
        action="store_true",
        help="Include rows that do not map cleanly into the three requested humor buckets",
    )
    return parser.parse_args()


def require_dependencies():
    try:
        from detoxify import Detoxify  # type: ignore
        from transformers import pipeline  # type: ignore
    except ModuleNotFoundError as exc:
        raise MissingDependencyError(
            "This script requires `detoxify` and `transformers` (plus their runtime "
            "dependencies such as torch). Install them before running."
        ) from exc
    return Detoxify, pipeline


def normalize_humor_bucket(humor_presence: str, humor_type: str) -> str:
    presence = (humor_presence or "").strip().lower()
    kind = (humor_type or "").strip().lower()

    if presence == "not humor":
        return "not_humor"
    if kind == "dark humor":
        return "dark_humor"
    if presence == "humor":
        return "regular_humor"
    return "unknown"


def load_dataset(dataset_path: Path, include_unknown: bool) -> Tuple[List[Dict[str, str]], Counter]:
    rows: List[Dict[str, str]] = []
    counts: Counter = Counter()

    with dataset_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "video_id", "humor_presence", "humor_type"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Missing required dataset columns: {sorted(missing)}")

        for row in reader:
            bucket = normalize_humor_bucket(row.get("humor_presence", ""), row.get("humor_type", ""))
            if bucket == "unknown" and not include_unknown:
                counts[bucket] += 1
                continue

            clean_row = {
                "id": (row.get("id") or "").strip(),
                "video_id": (row.get("video_id") or "").strip(),
                "humor_bucket": bucket,
                "humor_presence": row.get("humor_presence", ""),
                "humor_type": row.get("humor_type", ""),
                "title": row.get("title", ""),
                "url": row.get("url", ""),
            }
            if not clean_row["video_id"]:
                continue
            rows.append(clean_row)
            counts[bucket] += 1

    return rows, counts


def load_comments_file(path: Path) -> List[Dict[str, object]]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    comments: List[Dict[str, object]] = []

    # Try JSON Lines first.
    if raw.lstrip().startswith("{") and "\n" in raw:
        for line in raw.splitlines():
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

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return []

    if isinstance(obj, list):
        return [item for item in obj if isinstance(item, dict)]
    if isinstance(obj, dict):
        return [obj]
    return []


def iter_scored_comment_inputs(
    dataset_rows: Sequence[Dict[str, str]], comments_dir: Path, max_comments_per_video: int | None
) -> Iterator[Dict[str, object]]:
    for row in dataset_rows:
        comments_path = comments_dir / f"{row['video_id']}.json"
        if not comments_path.exists():
            continue

        comment_index = 0
        kept_for_video = 0
        for comment in load_comments_file(comments_path):
            text = str(comment.get("text") or "").strip()
            if not text:
                comment_index += 1
                continue

            yield {
                "video_id": row["video_id"],
                "dataset_row_id": row["id"],
                "humor_bucket": row["humor_bucket"],
                "comment_index": comment_index,
                "comment_text": text,
            }
            comment_index += 1
            kept_for_video += 1
            if max_comments_per_video is not None and kept_for_video >= max_comments_per_video:
                break


def chunked(items: Sequence[Dict[str, object]], size: int) -> Iterator[Sequence[Dict[str, object]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def normalize_label(label: str) -> str:
    return (label or "").strip().lower().replace(" ", "_")


def sentiment_scores_from_result(result: Sequence[Dict[str, float]]) -> Tuple[str, float, Dict[str, float]]:
    label_scores: Dict[str, float] = {}
    top_label = ""
    top_score = -1.0

    for entry in result:
        label = normalize_label(str(entry.get("label", "")))
        score = float(entry.get("score", 0.0))
        label_scores[label] = score
        if score > top_score:
            top_label = label
            top_score = score

    return top_label, top_score, {
        "sentiment_negative": label_scores.get("negative", 0.0),
        "sentiment_neutral": label_scores.get("neutral", 0.0),
        "sentiment_positive": label_scores.get("positive", 0.0),
    }


def emotion_scores_from_result(result: Sequence[Dict[str, float]]) -> Tuple[str, float, Dict[str, float]]:
    label_scores: Dict[str, float] = {}
    top_label = ""
    top_score = -1.0

    for entry in result:
        label = normalize_label(str(entry.get("label", "")))
        score = float(entry.get("score", 0.0))
        label_scores[label] = score
        if score > top_score:
            top_label = label
            top_score = score

    return top_label, top_score, {f"emotion_{label}": score for label, score in sorted(label_scores.items())}


def score_comments(
    comment_inputs: Sequence[Dict[str, object]],
    sentiment_model_name: str,
    emotion_model_name: str,
    toxicity_model_name: str,
    batch_size: int,
) -> Tuple[List[Dict[str, object]], List[str]]:
    Detoxify, pipeline = require_dependencies()

    sentiment_pipe = pipeline(
        "text-classification",
        model=sentiment_model_name,
        truncation=True,
        top_k=None,
    )
    emotion_pipe = pipeline(
        "text-classification",
        model=emotion_model_name,
        truncation=True,
        top_k=None,
    )
    toxicity_model = Detoxify(toxicity_model_name)

    scored_rows: List[Dict[str, object]] = []
    emotion_fieldnames: set[str] = set()
    toxicity_fieldnames: List[str] = []

    for batch in chunked(comment_inputs, batch_size):
        texts = [str(item["comment_text"]) for item in batch]
        sentiment_results = sentiment_pipe(texts, batch_size=batch_size, truncation=True, max_length=512)
        emotion_results = emotion_pipe(texts, batch_size=batch_size, truncation=True, max_length=512)
        toxicity_results = toxicity_model.predict(texts)
        if not toxicity_fieldnames:
            toxicity_fieldnames = sorted(toxicity_results.keys())

        for index, item in enumerate(batch):
            sentiment_label, sentiment_score, sentiment_values = sentiment_scores_from_result(sentiment_results[index])
            emotion_label, emotion_score, emotion_values = emotion_scores_from_result(emotion_results[index])
            emotion_fieldnames.update(emotion_values.keys())

            row = dict(item)
            row.update(
                {
                    "sentiment_label": sentiment_label,
                    "sentiment_score": sentiment_score,
                    **sentiment_values,
                    "emotion_label": emotion_label,
                    "emotion_score": emotion_score,
                    **emotion_values,
                }
            )
            for name in toxicity_fieldnames:
                row[name] = float(toxicity_results[name][index])
            scored_rows.append(row)

    return scored_rows, toxicity_fieldnames + sorted(emotion_fieldnames)


class RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.sums: Dict[str, float] = defaultdict(float)

    def add(self, row: Dict[str, object], keys: Iterable[str]) -> None:
        self.count += 1
        for key in keys:
            value = row.get(key)
            if value is None or value == "":
                continue
            self.sums[key] += float(value)

    def mean(self, key: str) -> float:
        if self.count == 0:
            return 0.0
        return self.sums.get(key, 0.0) / self.count


def aggregate_video_rows(scored_rows: Sequence[Dict[str, object]], metric_columns: Sequence[str]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], RunningStats] = {}
    metadata: Dict[Tuple[str, str], Dict[str, object]] = {}

    for row in scored_rows:
        key = (str(row["video_id"]), str(row["humor_bucket"]))
        if key not in grouped:
            grouped[key] = RunningStats()
            metadata[key] = {
                "video_id": row["video_id"],
                "humor_bucket": row["humor_bucket"],
                "dataset_row_id": row["dataset_row_id"],
            }
        grouped[key].add(row, metric_columns)

    video_rows: List[Dict[str, object]] = []
    for key in sorted(grouped):
        stats = grouped[key]
        row = dict(metadata[key])
        row["comments_scored"] = stats.count
        for column in metric_columns:
            row[column] = stats.mean(column)
        video_rows.append(row)

    return video_rows


def bucket_summary_from_comments(
    scored_rows: Sequence[Dict[str, object]],
    dataset_counts: Counter,
    video_rows: Sequence[Dict[str, object]],
    metric_columns: Sequence[str],
) -> List[Dict[str, object]]:
    comment_stats: Dict[str, RunningStats] = {bucket: RunningStats() for bucket in BUCKETS}
    video_counts = Counter(row["humor_bucket"] for row in video_rows)
    comment_counts = Counter(row["humor_bucket"] for row in scored_rows)

    for row in scored_rows:
        bucket = str(row["humor_bucket"])
        if bucket in comment_stats:
            comment_stats[bucket].add(row, metric_columns)

    summary_rows: List[Dict[str, object]] = []
    for bucket in BUCKETS:
        stats = comment_stats[bucket]
        row: Dict[str, object] = {
            "humor_bucket": bucket,
            "dataset_videos": dataset_counts.get(bucket, 0),
            "videos_with_comments": video_counts.get(bucket, 0),
            "videos_without_comments": dataset_counts.get(bucket, 0) - video_counts.get(bucket, 0),
            "comments_scored": comment_counts.get(bucket, 0),
            "avg_comments_per_scored_video": (
                comment_counts.get(bucket, 0) / video_counts.get(bucket, 0) if video_counts.get(bucket, 0) else 0.0
            ),
        }
        for column in metric_columns:
            row[column] = stats.mean(column)
        summary_rows.append(row)

    return summary_rows


def bucket_summary_from_videos(
    video_rows: Sequence[Dict[str, object]], dataset_counts: Counter, metric_columns: Sequence[str]
) -> List[Dict[str, object]]:
    stats_by_bucket: Dict[str, RunningStats] = {bucket: RunningStats() for bucket in BUCKETS}

    for row in video_rows:
        bucket = str(row["humor_bucket"])
        if bucket in stats_by_bucket:
            stats_by_bucket[bucket].add(row, metric_columns)

    summary_rows: List[Dict[str, object]] = []
    for bucket in BUCKETS:
        stats = stats_by_bucket[bucket]
        row: Dict[str, object] = {
            "humor_bucket": bucket,
            "dataset_videos": dataset_counts.get(bucket, 0),
            "videos_with_comments": stats.count,
            "videos_without_comments": dataset_counts.get(bucket, 0) - stats.count,
        }
        for column in metric_columns:
            row[f"video_mean_{column}"] = stats.mean(column)
        summary_rows.append(row)

    return summary_rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def main() -> None:
    args = parse_args()

    dataset_rows, dataset_counts = load_dataset(Path(args.dataset), include_unknown=args.include_unknown)
    comment_inputs = list(iter_scored_comment_inputs(dataset_rows, Path(args.comments_dir), args.max_comments_per_video))

    if not comment_inputs:
        raise SystemExit("No comments matched the dataset rows and filtering options.")

    scored_rows, dynamic_metric_columns = score_comments(
        comment_inputs=comment_inputs,
        sentiment_model_name=args.sentiment_model,
        emotion_model_name=args.emotion_model,
        toxicity_model_name=args.toxicity_model,
        batch_size=args.batch_size,
    )

    metric_columns = [
        "sentiment_negative",
        "sentiment_neutral",
        "sentiment_positive",
    ] + [name for name in dynamic_metric_columns if name not in COMMENT_FIELDS]

    video_rows = aggregate_video_rows(scored_rows, metric_columns)
    bucket_comment_rows = bucket_summary_from_comments(scored_rows, dataset_counts, video_rows, metric_columns)
    bucket_video_rows = bucket_summary_from_videos(video_rows, dataset_counts, metric_columns)

    output_dir = Path(args.output_dir)
    comment_path = output_dir / "comment_scores_by_humor_bucket.csv"
    video_path = output_dir / "video_scores_by_humor_bucket.csv"
    bucket_comment_path = output_dir / "humor_bucket_comment_summary.csv"
    bucket_video_path = output_dir / "humor_bucket_video_summary.csv"

    comment_fieldnames = COMMENT_FIELDS + [name for name in metric_columns if name not in COMMENT_FIELDS]
    video_fieldnames = ["video_id", "dataset_row_id", "humor_bucket", "comments_scored"] + list(metric_columns)
    bucket_comment_fieldnames = [
        "humor_bucket",
        "dataset_videos",
        "videos_with_comments",
        "videos_without_comments",
        "comments_scored",
        "avg_comments_per_scored_video",
    ] + list(metric_columns)
    bucket_video_fieldnames = [
        "humor_bucket",
        "dataset_videos",
        "videos_with_comments",
        "videos_without_comments",
    ] + [f"video_mean_{column}" for column in metric_columns]

    write_csv(comment_path, scored_rows, comment_fieldnames)
    write_csv(video_path, video_rows, video_fieldnames)
    write_csv(bucket_comment_path, bucket_comment_rows, bucket_comment_fieldnames)
    write_csv(bucket_video_path, bucket_video_rows, bucket_video_fieldnames)

    print(f"Dataset rows loaded: {len(dataset_rows)}")
    print(f"Bucket counts in dataset: {dict(dataset_counts)}")
    print(f"Comments scored: {len(scored_rows)}")
    print(f"Wrote: {comment_path}")
    print(f"Wrote: {video_path}")
    print(f"Wrote: {bucket_comment_path}")
    print(f"Wrote: {bucket_video_path}")


if __name__ == "__main__":
    try:
        main()
    except MissingDependencyError as exc:
        raise SystemExit(str(exc)) from exc
