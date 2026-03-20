import os
import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

UNIFIED_CSV = "/home/kiwi-pandas/Documents/humor-harm-api/unified_dataset.csv"
EMOTION_CSV = "/home/kiwi-pandas/Documents/humor-harm-api/scripts/comment_analysis/comment_emotion/comment_emotion_scores.csv"
TOXICITY_CSV = "/home/kiwi-pandas/Documents/humor-harm-api/scripts/comment_analysis/comment_sentiment_tox_scores.csv"

# -----------------------------
# helpers
# -----------------------------
def pick_col(cols, candidates):
    cols_lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    for cand in candidates:
        for c in cols:
            if cand.lower() in c.lower():
                return c
    return None

def to_num(series):
    s = series.astype("string").str.replace(",", "", regex=False).str.replace(" ", "", regex=False)
    s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "null": pd.NA})
    return pd.to_numeric(s, errors="coerce")

def mode_or_na(x):
    m = x.mode(dropna=True)
    return m.iloc[0] if not m.empty else pd.NA

# -----------------------------
# load unified
# -----------------------------
if not os.path.exists(UNIFIED_CSV):
    raise FileNotFoundError(f"Could not find {UNIFIED_CSV}")

df = pd.read_csv(UNIFIED_CSV, dtype=str)

df_vid_col = pick_col(df.columns, ["video_id", "video", "yt_id", "youtube_id", "id"])
if df_vid_col is None:
    raise ValueError(f"Could not find video id column in unified dataset. Columns: {list(df.columns)}")

if "humor_type" not in df.columns:
    raise ValueError(f"Could not find humor_type column in unified dataset. Columns: {list(df.columns)}")

df["video_id"] = df[df_vid_col].astype(str).str.strip()
df["humor_type"] = df["humor_type"].astype("string").str.strip()

# -----------------------------
# load and aggregate emotion
# -----------------------------
if not os.path.exists(EMOTION_CSV):
    raise FileNotFoundError(f"Could not find {EMOTION_CSV}")

edf = pd.read_csv(EMOTION_CSV, dtype=str)

e_vid_col = pick_col(edf.columns, ["video_id", "video", "yt_id", "youtube_id", "id"])
e_label_col = pick_col(edf.columns, ["emotion_label", "emotion", "label"])
e_conf_col = pick_col(edf.columns, ["confidence", "score", "emotion_confidence", "mean_confidence"])

if e_vid_col is None or e_label_col is None:
    raise ValueError(f"Need video id and emotion label columns in emotion file. Columns: {list(edf.columns)}")

edf["video_id"] = edf[e_vid_col].astype(str).str.strip()
edf["emotion_label"] = edf[e_label_col].astype("string").str.strip()

if e_conf_col is not None:
    edf["emotion_confidence"] = to_num(edf[e_conf_col])
else:
    edf["emotion_confidence"] = np.nan

edf = edf.dropna(subset=["video_id", "emotion_label"])
edf = edf[edf["emotion_label"] != ""]

emotion_video = (
    edf.groupby("video_id", as_index=False)
       .agg(
           comment_emotion_mode=("emotion_label", mode_or_na),
           n_comments_used_emotion=("emotion_label", "count"),
           comment_emotion_unique_count=("emotion_label", "nunique"),
           comment_emotion_conf_mean=("emotion_confidence", "mean"),
       )
)

# -----------------------------
# load and aggregate toxicity
# -----------------------------
if not os.path.exists(TOXICITY_CSV):
    raise FileNotFoundError(f"Could not find {TOXICITY_CSV}")

tdf = pd.read_csv(TOXICITY_CSV, dtype=str)

t_vid_col = pick_col(tdf.columns, ["video_id", "video", "yt_id", "youtube_id", "id"])
t_tox_col = pick_col(tdf.columns, ["toxicity", "toxic"])

if t_vid_col is None or t_tox_col is None:
    raise ValueError(f"Need video id and toxicity columns in toxicity file. Columns: {list(tdf.columns)}")

tdf["video_id"] = tdf[t_vid_col].astype(str).str.strip()
tdf["toxicity"] = to_num(tdf[t_tox_col])

tox_video = (
    tdf.dropna(subset=["video_id", "toxicity"])
       .groupby("video_id", as_index=False)
       .agg(
           comment_toxicity_mean=("toxicity", "mean"),
           comment_toxicity_median=("toxicity", "median"),
           comment_toxicity_max=("toxicity", "max"),
           n_comments_used_toxicity=("toxicity", "count"),
       )
)

# -----------------------------
# merge into unified
# -----------------------------
df = df.merge(emotion_video, on="video_id", how="left")
df = df.merge(tox_video, on="video_id", how="left")

# -----------------------------
# subset for Dark vs Regular
# -----------------------------
subset = df[df["humor_type"].isin(["Dark Humor", "Regular Humor"])].copy()

print("=" * 80)
print("BASIC COUNTS")
print("=" * 80)
print("Total rows in dataset:", len(df))
print("Rows with Dark Humor or Regular Humor:", len(subset))
print("Rows with merged emotion:", df["comment_emotion_mode"].notna().sum())
print("Rows with merged toxicity:", df["comment_toxicity_mean"].notna().sum())
print()

# -----------------------------
# overall toxicity summary
# -----------------------------
print("=" * 80)
print("OVERALL TOXICITY SUMMARY")
print("=" * 80)
print(df["comment_toxicity_mean"].describe())
print()

# -----------------------------
# toxicity by humor type
# -----------------------------
print("=" * 80)
print("TOXICITY BY HUMOR TYPE")
print("=" * 80)
tox_by_humor = (
    subset.groupby("humor_type")["comment_toxicity_mean"]
    .agg(["count", "mean", "median", "std", "min", "max"])
    .round(4)
)
print(tox_by_humor)
print()

# -----------------------------
# toxicity quartiles by humor type
# -----------------------------
print("=" * 80)
print("TOXICITY QUARTILES BY HUMOR TYPE")
print("=" * 80)
tox_quartiles = (
    subset.groupby("humor_type")["comment_toxicity_mean"]
    .quantile([0.25, 0.5, 0.75])
    .unstack()
    .rename(columns={0.25: "lower_25", 0.5: "median", 0.75: "upper_75"})
    .round(4)
)
print(tox_quartiles)
print()

# -----------------------------
# overall emotion counts
# -----------------------------
print("=" * 80)
print("OVERALL EMOTION COUNTS")
print("=" * 80)
emotion_counts = df["comment_emotion_mode"].value_counts(dropna=False)
print(emotion_counts)
print()

# -----------------------------
# overall emotion percentages
# -----------------------------
print("=" * 80)
print("OVERALL EMOTION PERCENTAGES")
print("=" * 80)
emotion_pct = (df["comment_emotion_mode"].value_counts(normalize=True, dropna=False) * 100).round(2)
print(emotion_pct)
print()

# -----------------------------
# emotion counts by humor type
# -----------------------------
print("=" * 80)
print("EMOTION COUNTS BY HUMOR TYPE")
print("=" * 80)
emotion_by_humor_counts = pd.crosstab(subset["humor_type"], subset["comment_emotion_mode"])
print(emotion_by_humor_counts)
print()

# -----------------------------
# emotion percentages by humor type
# -----------------------------
print("=" * 80)
print("EMOTION PERCENTAGES BY HUMOR TYPE")
print("=" * 80)
emotion_by_humor_pct = (
    pd.crosstab(subset["humor_type"], subset["comment_emotion_mode"], normalize="index") * 100
).round(2)
print(emotion_by_humor_pct)
print()

# -----------------------------
# save merged dataset + outputs
# -----------------------------
OUT_DIR = "/home/kiwi-pandas/Documents/humor-harm-api/scripts/comment_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

MERGED_OUT = f"{OUT_DIR}/unified_with_comment_emotion_toxicity.csv"
df.to_csv(MERGED_OUT, index=False)

tox_by_humor.to_csv(f"{OUT_DIR}/toxicity_by_humor_type.csv")
tox_quartiles.to_csv(f"{OUT_DIR}/toxicity_quartiles_by_humor_type.csv")
emotion_counts.to_csv(f"{OUT_DIR}/emotion_counts_overall.csv", header=["count"])
emotion_pct.to_csv(f"{OUT_DIR}/emotion_percentages_overall.csv", header=["percent"])
emotion_by_humor_counts.to_csv(f"{OUT_DIR}/emotion_counts_by_humor_type.csv")
emotion_by_humor_pct.to_csv(f"{OUT_DIR}/emotion_percentages_by_humor_type.csv")

print("Wrote merged dataset:", MERGED_OUT)
print("Saved summary CSV files to:", OUT_DIR)