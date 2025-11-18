import os
import re
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ===== config =====
CSV_PATH = "youtube_shorts_cc_2023_2025.csv"
OUT_DIR = Path("figures")
SAMPLE_ROWS_TO_SHOW = 5
TOP_N_HASHTAGS = 20
sns.set(style="whitegrid", palette="pastel")
# ==================

HASHTAG_RE = re.compile(r"#\w+")


def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # Convert types
    for col in ["viewCount", "likeCount", "commentCount", "durationSeconds"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse date
    if "datePublished" in df.columns:
        df["datePublished"] = pd.to_datetime(df["datePublished"], errors="coerce", utc=True)
        df["month"] = df["datePublished"].dt.to_period("M").astype(str)

    # Transcript features
    if "transcript" in df.columns:
        df["hasTranscript"] = df["transcript"].fillna("").str.strip().ne("")
        df["transcriptLength"] = df["transcript"].fillna("").str.len()

    # Hashtags
    def extract_hashtags_row(row):
        tags = set()
        for field in ["hashtags", "title", "description"]:
            val = row.get(field, "")
            if isinstance(val, str):
                tags.update(HASHTAG_RE.findall(val))
        return sorted(tags)

    df["hashtags_list"] = df.apply(extract_hashtags_row, axis=1)
    df["hashtagCount"] = df["hashtags_list"].apply(len)

    # Duration bins
    bins = [0, 15, 30, 45, 60, np.inf]
    labels = ["0–15", "16–30", "31–45", "46–60", "60+"]
    df["durationBin"] = pd.cut(df["durationSeconds"], bins=bins, labels=labels, include_lowest=True)

    # Topic fallback
    if "topic" not in df.columns:
        df["topic"] = "Unknown"

    return df


def ensure_outdir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(name):
    plt.tight_layout()
    ensure_outdir()
    plt.savefig(OUT_DIR / f"{name}.png", bbox_inches="tight", dpi=150)
    plt.show()


def plot_videos_per_topic(df):
    plt.figure(figsize=(7, 4))
    sns.countplot(data=df, y="topic", order=df["topic"].value_counts().index)
    plt.title("Videos per Topic")
    plt.xlabel("Count")
    plt.ylabel("Topic")
    save_fig("videos_per_topic")


def plot_views_hist(df):
    plt.figure(figsize=(6, 4))
    sns.histplot(df["viewCount"], bins=25, log_scale=(True, False), kde=True)
    plt.title("View Count Distribution (log scale)")
    plt.xlabel("Views")
    plt.ylabel("Frequency")
    save_fig("views_hist")


def plot_duration_hist(df):
    plt.figure(figsize=(6, 4))
    sns.histplot(df["durationSeconds"], bins=20, kde=True)
    plt.title("Duration Distribution (Seconds)")
    plt.xlabel("Duration (s)")
    plt.ylabel("Count")
    save_fig("duration_hist")


def plot_views_by_topic(df):
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="topic", y="viewCount", showfliers=False)
    plt.yscale("log")
    plt.title("View Counts by Topic (log scale)")
    plt.xlabel("Topic")
    plt.ylabel("Views")
    plt.xticks(rotation=20)
    save_fig("views_by_topic")


def plot_views_vs_duration(df):
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x="durationSeconds", y="viewCount", hue="topic")
    plt.yscale("log")
    plt.title("Views vs Duration (log scale)")
    plt.xlabel("Duration (s)")
    plt.ylabel("Views")
    plt.legend(title="Topic")
    save_fig("views_vs_duration")


def plot_duration_bins(df):
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="durationBin", hue="topic")
    plt.title("Duration Bin Distribution by Topic")
    plt.xlabel("Duration Bin (seconds)")
    plt.ylabel("Count")
    plt.legend(title="Topic")
    save_fig("duration_bins_by_topic")


def plot_monthly_trends(df):
    if "month" not in df.columns:
        return
    monthly = df.groupby(["month", "topic"])["videoId"].count().reset_index(name="count")
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=monthly, x="month", y="count", hue="topic", marker="o")
    plt.xticks(rotation=60)
    plt.title("Monthly Video Counts by Topic")
    plt.xlabel("Month")
    plt.ylabel("Count")
    save_fig("monthly_counts_by_topic")


def plot_top_hashtags(df, n=20):
    tags = Counter()
    for lst in df["hashtags_list"]:
        tags.update(lst)
    if not tags:
        return
    common = tags.most_common(n)
    tag_df = pd.DataFrame(common, columns=["hashtag", "count"])
    plt.figure(figsize=(6, 6))
    sns.barplot(data=tag_df, y="hashtag", x="count", palette="viridis")
    plt.title(f"Top {n} Hashtags")
    plt.xlabel("Frequency")
    plt.ylabel("Hashtag")
    save_fig("top_hashtags")


def main():
    if not os.path.exists(CSV_PATH):
        raise SystemExit(f"CSV not found: {CSV_PATH}")

    df = load_and_clean(CSV_PATH)
    print(df.head(SAMPLE_ROWS_TO_SHOW).to_string(index=False))
    print("\nSummary stats:")
    print(df[["viewCount", "durationSeconds", "likeCount", "commentCount"]].describe().to_string())

    plot_videos_per_topic(df)
    plot_views_hist(df)
    plot_duration_hist(df)
    plot_views_by_topic(df)
    plot_views_vs_duration(df)
    plot_duration_bins(df)
    plot_monthly_trends(df)
    plot_top_hashtags(df, TOP_N_HASHTAGS)


if __name__ == "__main__":
    main()
