import os, json, glob
import pandas as pd
from detoxify import Detoxify
from transformers import pipeline

COMMENTS_DIR = "/home/kiwi-pandas/Documents/humor-harm-api/comments"
VIDEO_OUT = os.path.join(COMMENTS_DIR, "video_level_scores.csv")
COMMENT_OUT = os.path.join(COMMENTS_DIR, "comment_level_scores.csv")

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def load_comments(path):
    """
    Returns list of dicts, each dict is one comment with at least 'text'.
    Supports:
      1) JSON list: [ {...}, {...} ]
      2) JSON dict: { ... } (single comment)
      3) JSONL: one JSON object per line
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    if not raw:
        return []

    # JSONL
    if raw.count("\n") > 0 and raw.lstrip().startswith("{") and raw.rstrip().endswith("}"):
        comments = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                comments.append(json.loads(line))
            except Exception:
                pass
        if comments:
            return comments

    # Normal JSON
    obj = json.loads(raw)
    if isinstance(obj, list):
        return [c for c in obj if isinstance(c, dict)]
    if isinstance(obj, dict):
        return [obj]
    return []


def signed_sent(label, score):
    label = (label or "").lower()
    if "pos" in label:
        return float(score)
    if "neg" in label:
        return -float(score)
    return 0.0


def main():
    detox = Detoxify("original")
    sent = pipeline("sentiment-analysis", model=SENTIMENT_MODEL)

    comment_rows = []
    video_rows = []

    for path in sorted(glob.glob(os.path.join(COMMENTS_DIR, "*.json"))):
        video_id = os.path.splitext(os.path.basename(path))[0]

        comments = load_comments(path)
        texts = [c.get("text", "").strip() for c in comments]
        texts = [t for t in texts if t]

        if not texts:
            print("NO COMMENTS:", video_id)
            continue

        tox = detox.predict(texts)  # dict of lists
        sent_out = sent(texts, truncation=True, max_length=512, batch_size=16)

        tox_list = []
        sent_signed_list = []

        for i, text in enumerate(texts):
            toxicity = float(tox["toxicity"][i])
            s_label = sent_out[i]["label"]
            s_score = float(sent_out[i]["score"])
            s_signed = signed_sent(s_label, s_score)

            tox_list.append(toxicity)
            sent_signed_list.append(s_signed)

            row = {
                "video_id": video_id,
                "comment_index": i,
                "comment_text": text,
                "toxicity": toxicity,
                "sentiment_label": s_label,
                "sentiment_score": s_score,
                "sentiment_signed": s_signed,
            }

            # include other detoxify fields too
            for k in tox:
                if k != "toxicity":
                    row[k] = float(tox[k][i])

            comment_rows.append(row)

        # video aggregates
        n = len(texts)
        video_rows.append({
            "video_id": video_id,
            "n_comments_used": n,
            "toxicity_mean": sum(tox_list) / n,
            "toxicity_max": max(tox_list),
            "sentiment_signed_mean": sum(sent_signed_list) / n,
            "sentiment_pos_frac": sum(1 for x in sent_signed_list if x > 0) / n,
            "sentiment_neu_frac": sum(1 for x in sent_signed_list if x == 0) / n,
            "sentiment_neg_frac": sum(1 for x in sent_signed_list if x < 0) / n,
        })

        print("DONE:", video_id, "comments:", n)

    if comment_rows:
        pd.DataFrame(comment_rows).to_csv(COMMENT_OUT, index=False)
        print("WROTE:", COMMENT_OUT)

    if video_rows:
        dfv = pd.DataFrame(video_rows).sort_values("toxicity_mean", ascending=False)
        dfv.to_csv(VIDEO_OUT, index=False)
        print("WROTE:", VIDEO_OUT)


if __name__ == "__main__":
    main()