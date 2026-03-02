import csv
import os
import random
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMMENT_SCORES = os.path.join(BASE_DIR, "comment_level_scores.csv")
OUTPUT_VIDEO_SCORES = os.path.join(BASE_DIR, "video_level_scores_neutral50_screened.csv")


def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_video_scores_with_neutral_screening(
    comment_scores_path: str,
    output_path: str,
    neutral_keep_frac: float = 0.5,
    random_seed: int = 42,
):
    rows_by_video = defaultdict(list)

    with open(comment_scores_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"video_id", "toxicity", "sentiment_signed"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Missing columns in {comment_scores_path}: {sorted(missing)}")

        for row in reader:
            video_id = (row.get("video_id") or "").strip()
            if not video_id:
                continue
            rows_by_video[video_id].append(
                {
                    "toxicity": _to_float(row.get("toxicity")),
                    "sentiment_signed": _to_float(row.get("sentiment_signed")),
                }
            )

    rng = random.Random(random_seed)
    output_rows = []

    for video_id, rows in rows_by_video.items():
        neutral = [r for r in rows if r["sentiment_signed"] == 0.0]
        non_neutral = [r for r in rows if r["sentiment_signed"] != 0.0]

        keep_n = int(round(len(neutral) * neutral_keep_frac))
        if keep_n > len(neutral):
            keep_n = len(neutral)

        sampled_neutral = rng.sample(neutral, keep_n) if keep_n else []
        kept = non_neutral + sampled_neutral

        if not kept:
            continue

        n = len(kept)
        toxicity_values = [r["toxicity"] for r in kept]
        sentiment_values = [r["sentiment_signed"] for r in kept]

        output_rows.append(
            {
                "video_id": video_id,
                "n_comments_used": n,
                "toxicity_mean": sum(toxicity_values) / n,
                "toxicity_max": max(toxicity_values),
                "sentiment_signed_mean": sum(sentiment_values) / n,
                "sentiment_pos_frac": sum(1 for x in sentiment_values if x > 0) / n,
                "sentiment_neu_frac": sum(1 for x in sentiment_values if x == 0) / n,
                "sentiment_neg_frac": sum(1 for x in sentiment_values if x < 0) / n,
            }
        )

    output_rows.sort(key=lambda r: r["toxicity_mean"], reverse=True)

    fieldnames = [
        "video_id",
        "n_comments_used",
        "toxicity_mean",
        "toxicity_max",
        "sentiment_signed_mean",
        "sentiment_pos_frac",
        "sentiment_neu_frac",
        "sentiment_neg_frac",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    return len(output_rows)


if __name__ == "__main__":
    videos = build_video_scores_with_neutral_screening(
        comment_scores_path=COMMENT_SCORES,
        output_path=OUTPUT_VIDEO_SCORES,
        neutral_keep_frac=0.5,
    )
    print(f"WROTE: {OUTPUT_VIDEO_SCORES}")
    print(f"VIDEOS: {videos}")
