import argparse
import os
import re
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = str(REPO_ROOT / "unified_dataset.csv")
DEFAULT_OUT = str(REPO_ROOT / "scripts" / "topic_analysis" / "bertopic_topics.csv")
DEFAULT_TOPIC_INFO = str(REPO_ROOT / "scripts" / "topic_analysis" / "bertopic_topic_info.csv")

DEFAULT_ID_COL = "video_id"
DEFAULT_DESC_COL = "description"
DEFAULT_TR_COL = "transcript_text"


def clean_text(s: str) -> str:
    """Light cleaning that keeps meaning but removes obvious junk."""
    if s is None:
        return ""
    s = str(s).lower()

    # urls
    s = re.sub(r"http\S+|www\.\S+", " ", s)

    # remove some common boilerplate
    s = re.sub(r"\b(like and subscribe|subscribe|follow me|follow|link in bio)\b", " ", s)

    # remove timestamps like 00:12 or 1:02
    s = re.sub(r"\b\d{1,2}:\d{2}\b", " ", s)

    # keep letters and spaces
    s = re.sub(r"[^a-z\s]", " ", s)

    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_docs(df: pd.DataFrame, desc_col: str, tr_col: str) -> pd.Series:
    """Combine description + transcript and clean."""
    raw = df[desc_col].fillna("").astype(str) + " " + df[tr_col].fillna("").astype(str)
    cleaned = raw.map(clean_text)
    return cleaned


def safe_topic_probabilities(topics: np.ndarray, probs) -> list:
    """Pick the probability of the assigned topic for each doc."""
    if probs is None:
        return [None] * len(topics)

    out = []
    for t, p in zip(topics, probs):
        try:
            t_int = int(t)
        except Exception:
            out.append(None)
            continue

        if t_int == -1:
            out.append(float(np.max(p)) if len(p) else None)
        else:
            out.append(float(p[t_int]) if t_int < len(p) else None)
    return out


def pick_col(df: pd.DataFrame, preferred: str, fallbacks: list) -> str:
    if preferred in df.columns:
        return preferred
    for c in fallbacks:
        if c in df.columns:
            return c
    return ""


def train_and_assign_topics(
    df: pd.DataFrame,
    id_col: str,
    desc_col: str,
    tr_col: str,
    min_words_train: int,
    min_words_assign: int,
    min_cluster_size: int,
    min_samples: int,
    n_neighbors: int,
    n_components: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train BERTopic on longer docs only, then assign topics to all docs.
    This avoids collapse caused by extremely short transcripts.
    """

    docs = build_docs(df, desc_col, tr_col)
    word_counts = docs.str.split().str.len().fillna(0)

    # train only on docs with enough text
    train_mask = word_counts >= min_words_train
    df_train = df[train_mask].copy()
    docs_train = docs[train_mask].tolist()

    # later assign topics to docs that have at least some text
    assign_mask = word_counts >= min_words_assign
    df_assign = df[assign_mask].copy()
    docs_assign = docs[assign_mask].tolist()

    print("Total rows:", len(df))
    print("Train docs kept:", len(df_train), "min_words_train:", min_words_train)
    print("Assign docs kept:", len(df_assign), "min_words_assign:", min_words_assign)

    if len(df_train) < 50:
        raise ValueError(
            "Too few training docs after filtering. Lower --min-words-train."
        )

    # Delayed heavy imports so --help and argument parsing work even without deps installed.
    try:
        from bertopic import BERTopic
        from bertopic.representation import KeyBERTInspired
        import hdbscan
        from sentence_transformers import SentenceTransformer
        from umap import UMAP
    except ImportError as exc:
        missing_pkg = str(exc).split("'")[1] if "'" in str(exc) else str(exc)
        raise ImportError(
            "Missing BERTopic dependency. Install requirements before running this script. "
            f"Original error: {missing_pkg}"
        ) from exc

    # Embeddings
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # UMAP reduces embedding dimensionality before clustering
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    # HDBSCAN creates clusters and can mark outliers as -1
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    # Vectorizer is used for topic words after clusters are formed
    # Keep it safe if the number of topics is small
    vectorizer_model = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=1.0,
    )

    # Representation model improves topic names
    representation_model = KeyBERTInspired()

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        calculate_probabilities=True,
        verbose=True,
    )

    # Fit on longer docs
    topics_train, probs_train = topic_model.fit_transform(docs_train)

    # Assign topics to the broader set
    topics_assign, probs_assign = topic_model.transform(docs_assign)

    topic_prob_assign = safe_topic_probabilities(topics_assign, probs_assign)

    # Map topic id to name
    topic_info = topic_model.get_topic_info().copy()
    name_map = {}
    if "Name" in topic_info.columns:
        for _, r in topic_info.iterrows():
            name_map[int(r["Topic"])] = str(r["Name"])
    else:
        for _, r in topic_info.iterrows():
            tid = int(r["Topic"])
            if tid == -1:
                name_map[tid] = "outlier"
            else:
                words = [w for w, _ in topic_model.get_topic(tid)[:5]]
                name_map[tid] = "_".join(words)

    # Build output for all videos
    out = pd.DataFrame(
        {
            "video_id": df[id_col].astype(str),
            "description": df[desc_col].fillna("").astype(str),
            "transcript_text": df[tr_col].fillna("").astype(str),
            "topic_id": [-999] * len(df),  # placeholder
            "topic_probability": [None] * len(df),
            "topic_name": ["no_text"] * len(df),
            "topic_training_used": train_mask.values,
            "topic_assigned_used": assign_mask.values,
            "cleaned_word_count": word_counts.values,
        }
    )

    # Fill assigned rows
    out.loc[assign_mask, "topic_id"] = topics_assign
    out.loc[assign_mask, "topic_probability"] = topic_prob_assign
    out.loc[assign_mask, "topic_name"] = [name_map.get(int(t), "unknown") for t in topics_assign]

    # For rows with no text, keep topic_id = -999 and name no_text

    # Print quick sanity
    vc = out.loc[out["topic_id"] != -999, "topic_id"].value_counts()
    print("\nTopic counts among assigned docs (top 20)")
    print(vc.head(20))
    print("\nUnique topics excluding outlier (-1), among assigned docs:",
          out[(out["topic_id"] != -999) & (out["topic_id"] != -1)]["topic_id"].nunique())

    # Also show training topic spread
    train_vc = pd.Series(topics_train).value_counts()
    print("\nTraining cluster counts (top 20)")
    print(train_vc.head(20))
    print("\nUnique training topics excluding outlier (-1):",
          pd.Series(topics_train)[pd.Series(topics_train) != -1].nunique())

    return out, topic_info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--output", default=DEFAULT_OUT)
    ap.add_argument("--topic-info-output", default=DEFAULT_TOPIC_INFO)

    ap.add_argument("--id-col", default=DEFAULT_ID_COL)
    ap.add_argument("--description-col", default=DEFAULT_DESC_COL)
    ap.add_argument("--transcript-col", default=DEFAULT_TR_COL)

    # Filters for short form data
    ap.add_argument("--min-words-train", type=int, default=15)
    ap.add_argument("--min-words-assign", type=int, default=3)

    # Clustering knobs
    ap.add_argument("--min-cluster-size", type=int, default=15)
    ap.add_argument("--min-samples", type=int, default=3)

    # UMAP knobs
    ap.add_argument("--n-neighbors", type=int, default=12)
    ap.add_argument("--n-components", type=int, default=5)

    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"Input not found: {args.input}")
        sys.exit(1)

    df = pd.read_csv(args.input)

    id_col = pick_col(df, args.id_col, ["id", "videoid", "video_id"])
    desc_col = pick_col(df, args.description_col, ["desc", "video_description", "description"])
    tr_col = pick_col(df, args.transcript_col, ["transcript", "transcript_text", "caption", "captions"])

    if id_col == "" or desc_col == "" or tr_col == "":
        print("Could not find required columns.")
        print("Columns in file:", list(df.columns))
        sys.exit(1)

    out, topic_info = train_and_assign_topics(
        df=df,
        id_col=id_col,
        desc_col=desc_col,
        tr_col=tr_col,
        min_words_train=args.min_words_train,
        min_words_assign=args.min_words_assign,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        n_neighbors=args.n_neighbors,
        n_components=args.n_components,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.topic_info_output).parent.mkdir(parents=True, exist_ok=True)

    out.to_csv(args.output, index=False)
    topic_info.to_csv(args.topic_info_output, index=False)

    print("\nWrote", args.output)
    print("Wrote", args.topic_info_output)


if __name__ == "__main__":
    main()
