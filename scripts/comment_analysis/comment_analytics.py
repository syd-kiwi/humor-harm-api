import os
import pandas as pd

COMMENTS_DIR = "/home/kiwi-pandas/Documents/humor-harm-api/scripts/comment_analysis"
VIDEO_SCORES = os.path.join(COMMENTS_DIR, "video_level_scores.csv")

df = pd.read_csv(VIDEO_SCORES)

# Basic sanity
needed = {"video_id", "toxicity_mean", "toxicity_max", "sentiment_signed_mean",
          "sentiment_pos_frac", "sentiment_neg_frac", "sentiment_neu_frac", "n_comments_used"}
missing = needed - set(df.columns)
if missing:
    raise SystemExit(f"Missing columns in {VIDEO_SCORES}: {sorted(missing)}")

# Most / least toxic (by mean)
most_toxic = df.sort_values("toxicity_mean", ascending=False).iloc[0]
least_toxic = df.sort_values("toxicity_mean", ascending=True).iloc[0]

# Also show by max toxicity (often useful)
most_toxic_max = df.sort_values("toxicity_max", ascending=False).iloc[0]
least_toxic_max = df.sort_values("toxicity_max", ascending=True).iloc[0]

# Overall averages across videos (unweighted)
overall = {
    "n_videos": len(df),
    "avg_toxicity_mean": df["toxicity_mean"].mean(),
    "avg_toxicity_max": df["toxicity_max"].mean(),
    "avg_sentiment_signed_mean": df["sentiment_signed_mean"].mean(),
    "avg_pos_frac": df["sentiment_pos_frac"].mean(),
    "avg_neu_frac": df["sentiment_neu_frac"].mean(),
    "avg_neg_frac": df["sentiment_neg_frac"].mean(),
}

# Weighted by number of comments (better if some videos have fewer comments)
w = df["n_comments_used"].clip(lower=1)
overall_weighted = {
    "w_avg_toxicity_mean": (df["toxicity_mean"] * w).sum() / w.sum(),
    "w_avg_toxicity_max": (df["toxicity_max"] * w).sum() / w.sum(),
    "w_avg_sentiment_signed_mean": (df["sentiment_signed_mean"] * w).sum() / w.sum(),
    "w_avg_pos_frac": (df["sentiment_pos_frac"] * w).sum() / w.sum(),
    "w_avg_neu_frac": (df["sentiment_neu_frac"] * w).sum() / w.sum(),
    "w_avg_neg_frac": (df["sentiment_neg_frac"] * w).sum() / w.sum(),
}

# Average sentiment for "positive videos" vs "negative videos"
pos_videos = df[df["sentiment_signed_mean"] > 0]
neg_videos = df[df["sentiment_signed_mean"] < 0]

pos_stats = {
    "n_pos_videos": len(pos_videos),
    "avg_sentiment_signed_mean_pos_videos": pos_videos["sentiment_signed_mean"].mean() if len(pos_videos) else None,
    "avg_pos_frac_pos_videos": pos_videos["sentiment_pos_frac"].mean() if len(pos_videos) else None,
    "avg_neg_frac_pos_videos": pos_videos["sentiment_neg_frac"].mean() if len(pos_videos) else None,
}

neg_stats = {
    "n_neg_videos": len(neg_videos),
    "avg_sentiment_signed_mean_neg_videos": neg_videos["sentiment_signed_mean"].mean() if len(neg_videos) else None,
    "avg_pos_frac_neg_videos": neg_videos["sentiment_pos_frac"].mean() if len(neg_videos) else None,
    "avg_neg_frac_neg_videos": neg_videos["sentiment_neg_frac"].mean() if len(neg_videos) else None,
}

def print_video_row(title, row):
    print(title)
    print(f"  video_id: {row['video_id']}")
    print(f"  n_comments_used: {row['n_comments_used']}")
    print(f"  toxicity_mean: {row['toxicity_mean']:.4f}")
    print(f"  toxicity_max: {row['toxicity_max']:.4f}")
    print(f"  sentiment_signed_mean: {row['sentiment_signed_mean']:.4f}")
    print(f"  pos/neu/neg frac: {row['sentiment_pos_frac']:.3f} / {row['sentiment_neu_frac']:.3f} / {row['sentiment_neg_frac']:.3f}")
    print()

print_video_row("MOST TOXIC (by toxicity_mean)", most_toxic)
print_video_row("LEAST TOXIC (by toxicity_mean)", least_toxic)

print_video_row("MOST TOXIC (by toxicity_max)", most_toxic_max)
print_video_row("LEAST TOXIC (by toxicity_max)", least_toxic_max)

print("OVERALL (unweighted across videos)")
for k, v in overall.items():
    print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
print()

print("OVERALL (weighted by n_comments_used)")
for k, v in overall_weighted.items():
    print(f"  {k}: {v:.6f}")
print()

print("POSITIVE VIDEOS (sentiment_signed_mean > 0)")
for k, v in pos_stats.items():
    print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
print()

print("NEGATIVE VIDEOS (sentiment_signed_mean < 0)")
for k, v in neg_stats.items():
    print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
print()


import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

'''
Graphs
'''

df = pd.read_csv(VIDEO_SCORES).dropna(subset=["n_comments_used", "toxicity_mean", "sentiment_signed_mean"])

sns.set_theme(style="whitegrid")

# 1) Distributions (3 histograms)
plt.figure(figsize=(10, 6))
fig, axes = plt.subplots(3, 1, figsize=(10, 9))

sns.histplot(df["n_comments_used"], bins=25, kde=True, ax=axes[0])
axes[0].set_title("Distribution of n_comments_used")
axes[0].set_xlabel("n_comments_used")

sns.histplot(df["toxicity_mean"], bins=25, kde=True, ax=axes[1])
axes[1].set_title("Distribution of toxicity_mean")
axes[1].set_xlabel("toxicity_mean")

sns.histplot(df["sentiment_signed_mean"], bins=25, kde=True, ax=axes[2])
axes[2].axvline(0, linewidth=1)
axes[2].set_title("Distribution of sentiment_signed_mean")
axes[2].set_xlabel("sentiment_signed_mean")

plt.tight_layout()
plt.savefig(os.path.join(COMMENTS_DIR, "sb_distributions_n_tox_sent.png"), dpi=200)
plt.close()

# 2) Scatter: toxicity_mean vs sentiment_signed_mean (size = n_comments_used)
plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=df,
    x="toxicity_mean",
    y="sentiment_signed_mean",
    alpha=0.6,
)
plt.axhline(0, linewidth=1)
plt.title("toxicity_mean vs sentiment_signed_mean")
plt.tight_layout()
plt.savefig(os.path.join(COMMENTS_DIR, "sb_scatter_tox_vs_sent.png"), dpi=200)
plt.close()

print("Saved plots to:", COMMENTS_DIR)