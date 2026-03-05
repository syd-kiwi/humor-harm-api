import os, glob, re
import pandas as pd
import numpy as np

# -------------------------
# Paths (edit these)
# -------------------------
LLM_DIR = "/home/kiwi-pandas/Documents/humor-harm-api/scripts/llm_eval/outputs"
DATASET_CSV = "/home/kiwi-pandas/Documents/humor-harm-api/unified_dataset.csv"
OUT_LONG = "/home/kiwi-pandas/Documents/humor-harm-api/scripts/llm_eval/llm_vs_human_by_model_long.csv"
OUT_SUMMARY= "/home/kiwi-pandas/Documents/humor-harm-api/scripts/llm_eval/llm_vs_human_by_model_summary.csv"

MODEL_LIST = [
    "deepseek-ai/DeepSeek-V3.1",
    "gemini-3.1-pro-preview",
    "gemini-2.5-flash",
    "gpt-5-mini-2025-08-07",
    "claude-sonnet-4-6",
]

FIELDS = [
    "humor_presence","joke_topic","rhetorical_device","stand_up",
    "humor_type","target_category","dark_intensity","note"
]

TEXT_COLS = ["humor_presence","humor_type","stand_up","note"]
LIST_COLS = ["joke_topic","rhetorical_device","target_category"]
NUM_COLS  = ["dark_intensity"]

# -------------------------
# HELPERS
# -------------------------
def get_model_from_filename(fname: str):
    s = str(fname).lower()
    for m in MODEL_LIST:
        if m.lower() in s:
            return m
    return np.nan

def extract_video_id(qid):
    if pd.isna(qid):
        return np.nan
    s = str(qid).strip()
    last = s.split("/")[-1]
    last = re.sub(r"\.(mp4|mov|mkv|webm)$", "", last, flags=re.IGNORECASE)
    last = re.sub(r"[^A-Za-z0-9_\-]", "", last)
    return last if last else np.nan

def norm_text(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    s = s.replace("none of these", "none")
    s = " ".join(s.split())
    return s

def norm_listlike(x):
    if pd.isna(x):
        return []
    if isinstance(x, list):
        items = x
    else:
        s = str(x).strip()
        if s.startswith("[") and s.endswith("]"):
            s2 = s[1:-1].strip()
            if not s2:
                return []
            items = [i.strip().strip("'\"") for i in s2.split(",")]
        else:
            items = [i.strip() for i in s.split(",")]
    items = [norm_text(i) for i in items if norm_text(i)]
    return sorted(list(dict.fromkeys(items)))

def parse_rawresponse(raw):
    out = {k: np.nan for k in FIELDS}
    if pd.isna(raw):
        return out
    s = str(raw).strip()
    if '""' in s:
        s = s.replace('""', '"')
    if not (s.startswith("{") and s.endswith("}")):
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if m:
            s = m.group(0)
    try:
        obj = json.loads(s)
    except Exception:
        return out
    for k in FIELDS:
        if k in obj:
            out[k] = obj[k]
    return out

def safe_accuracy(a, b):
    if a is None or b is None:
        return np.nan, 0
    m = a.notna() & b.notna()
    if m.sum() == 0:
        return np.nan, 0
    return float((a[m] == b[m]).mean()), int(m.sum())

def safe_exact_set_match(a_lists, b_lists):
    if a_lists is None or b_lists is None:
        return np.nan, 0
    ok = []
    for a, b in zip(a_lists, b_lists):
        if not isinstance(a, list) or not isinstance(b, list):
            continue
        if (len(a) == 0) and (len(b) == 0):
            continue
        ok.append(a == b)
    if len(ok) == 0:
        return np.nan, 0
    return float(np.mean(ok)), int(len(ok))

def safe_mae(x, y):
    if x is None or y is None:
        return np.nan, 0
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    m = x.notna() & y.notna()
    if m.sum() == 0:
        return np.nan, 0
    return float((x[m] - y[m]).abs().mean()), int(m.sum())

# -------------------------
# LOAD HUMAN
# -------------------------
ds = pd.read_csv(DATASET_CSV)

if "video_id" in ds.columns:
    ds["video_id_key"] = ds["video_id"].astype(str).str.strip()
else:
    ds["video_id_key"] = ds["id"].astype(str).str.strip()

ds = ds.drop_duplicates(subset=["video_id_key"], keep="first").copy()
print("Human unique videos:", ds["video_id_key"].nunique())

# normalize human
for c in TEXT_COLS:
    if c in ds.columns:
        ds[c + "_norm"] = ds[c].apply(norm_text)
for c in LIST_COLS:
    if c in ds.columns:
        ds[c + "_norm"] = ds[c].apply(norm_listlike)
for c in NUM_COLS:
    if c in ds.columns:
        ds[c + "_num"] = pd.to_numeric(ds[c], errors="coerce")

# -------------------------
# LOAD LLM FILES
# -------------------------
llm_files = sorted(glob.glob(os.path.join(LLM_DIR, "*.csv")))
print("LLM files found:", len(llm_files))
if not llm_files:
    raise FileNotFoundError(f"No CSV files found in {LLM_DIR}")

parts = []
for f in llm_files:
    tmp = pd.read_csv(f)
    tmp["source_file"] = os.path.basename(f)
    tmp["model"] = get_model_from_filename(tmp["source_file"].iloc[0])
    parts.append(tmp)

llm = pd.concat(parts, ignore_index=True)
llm = llm[llm["model"].notna()].copy()

if "QID" not in llm.columns:
    raise KeyError("LLM CSVs must have QID.")

llm["video_id_key"] = llm["QID"].apply(extract_video_id)

# parse RawResponse if parsed columns are missing
missing = [c for c in FIELDS if c not in llm.columns]
if missing:
    if "RawResponse" not in llm.columns:
        raise KeyError("LLM CSVs missing parsed columns and RawResponse.")
    parsed = llm["RawResponse"].apply(parse_rawresponse).apply(pd.Series)
    for c in FIELDS:
        if c not in llm.columns:
            llm[c] = parsed[c]

# normalize llm
for c in TEXT_COLS:
    llm[c + "_norm"] = llm[c].apply(norm_text)
for c in LIST_COLS:
    llm[c + "_norm"] = llm[c].apply(norm_listlike)
for c in NUM_COLS:
    llm[c + "_num"] = pd.to_numeric(llm[c], errors="coerce")

# keep first row per video per model
llm = llm.drop_duplicates(subset=["model", "video_id_key"], keep="first").copy()
print("LLM unique (model,video):", len(llm))
print("Coverage by model:\n", llm.groupby("model")["video_id_key"].nunique().to_string())

# -------------------------
# MERGE LONG (1550 humans, repeated per model)
# -------------------------
human_keep = ["video_id_key"]
for c in ["id","video_id","url","title","channel","upload_date","duration","view_count","like_count","comment_count","searched_keyword"]:
    if c in ds.columns:
        human_keep.append(c)
for c in FIELDS:
    if c in ds.columns:
        human_keep.append(c)
for c in [x + "_norm" for x in TEXT_COLS + LIST_COLS]:
    if c in ds.columns:
        human_keep.append(c)
for c in [x + "_num" for x in NUM_COLS]:
    if c in ds.columns:
        human_keep.append(c)

llm_keep = ["video_id_key","model","source_file","QID"]
for c in ["ParseIssues","RawResponse"]:
    if c in llm.columns:
        llm_keep.append(c)
for c in FIELDS:
    if c in llm.columns:
        llm_keep.append(c)
for c in [x + "_norm" for x in TEXT_COLS + LIST_COLS]:
    if c in llm.columns:
        llm_keep.append(c)
for c in [x + "_num" for x in NUM_COLS]:
    if c in llm.columns:
        llm_keep.append(c)

merged = ds[human_keep].merge(llm[llm_keep], on="video_id_key", how="left", suffixes=("_human","_llm"))

# rename normalized cols so summary is stable (prevents your NoneType crash)
rename_map = {}
for c in TEXT_COLS + LIST_COLS:
    if (c + "_norm_human") not in merged.columns and (c + "_norm") in merged.columns:
        rename_map[c + "_norm"] = c + "_norm_human"
    if (c + "_norm_llm") not in merged.columns and (c + "_norm_y") in merged.columns:
        rename_map[c + "_norm_y"] = c + "_norm_llm"
for c in NUM_COLS:
    if (c + "_num_human") not in merged.columns and (c + "_num") in merged.columns:
        rename_map[c + "_num"] = c + "_num_human"
    if (c + "_num_llm") not in merged.columns and (c + "_num_y") in merged.columns:
        rename_map[c + "_num_y"] = c + "_num_llm"

if rename_map:
    merged = merged.rename(columns=rename_map)

# compute match columns (optional but useful)
for c in ["humor_presence","humor_type","stand_up"]:
    ha = c + "_norm_human"
    la = c + "_norm_llm"
    if ha in merged.columns and la in merged.columns:
        m = merged[ha].notna() & merged[la].notna()
        merged[c + "_match"] = np.where(m, (merged[ha] == merged[la]).astype(int), np.nan)

for c in ["joke_topic","rhetorical_device","target_category"]:
    ha = c + "_norm_human"
    la = c + "_norm_llm"
    if ha in merged.columns and la in merged.columns:
        out = []
        for a, b in zip(merged[ha], merged[la]):
            if not isinstance(a, list) or not isinstance(b, list):
                out.append(np.nan)
            else:
                if (len(a) == 0) and (len(b) == 0):
                    out.append(np.nan)
                else:
                    out.append(int(a == b))
        merged[c + "_match"] = out

if "dark_intensity_num_human" in merged.columns and "dark_intensity_num_llm" in merged.columns:
    merged["dark_intensity_abs_diff"] = (merged["dark_intensity_num_llm"] - merged["dark_intensity_num_human"]).abs()

merged.to_csv(OUT_LONG, index=False)
print("Wrote:", OUT_LONG)
print("Human unique videos (merged):", merged["video_id_key"].nunique())

# -------------------------
# SUMMARY BY MODEL (no crashes even if some cols missing)
# -------------------------
rows = []
for model, g in merged.groupby("model", dropna=False):
    if pd.isna(model):
        continue

    hp_acc, hp_n = safe_accuracy(
        g["humor_presence_norm_human"] if "humor_presence_norm_human" in g.columns else None,
        g["humor_presence_norm_llm"] if "humor_presence_norm_llm" in g.columns else None,
    )
    ht_acc, ht_n = safe_accuracy(
        g["humor_type_norm_human"] if "humor_type_norm_human" in g.columns else None,
        g["humor_type_norm_llm"] if "humor_type_norm_llm" in g.columns else None,
    )
    su_acc, su_n = safe_accuracy(
        g["stand_up_norm_human"] if "stand_up_norm_human" in g.columns else None,
        g["stand_up_norm_llm"] if "stand_up_norm_llm" in g.columns else None,
    )
    jt_esm, jt_n = safe_exact_set_match(
        g["joke_topic_norm_human"] if "joke_topic_norm_human" in g.columns else None,
        g["joke_topic_norm_llm"] if "joke_topic_norm_llm" in g.columns else None,
    )
    rd_esm, rd_n = safe_exact_set_match(
        g["rhetorical_device_norm_human"] if "rhetorical_device_norm_human" in g.columns else None,
        g["rhetorical_device_norm_llm"] if "rhetorical_device_norm_llm" in g.columns else None,
    )
    di_mae, di_n = safe_mae(
        g["dark_intensity_num_human"] if "dark_intensity_num_human" in g.columns else None,
        g["dark_intensity_num_llm"] if "dark_intensity_num_llm" in g.columns else None,
    )

    match_cols = [c for c in g.columns if c.endswith("_match")]
    if match_cols:
        mat = g[match_cols].apply(pd.to_numeric, errors="coerce")
        possible = mat.notna().sum(axis=1)
        count = mat.sum(axis=1, skipna=True)
        mr = np.where(possible > 0, count / possible, np.nan)
        overall = float(np.nanmean(mr)) if np.isfinite(mr).any() else np.nan
        overall_n = int(np.sum(~np.isnan(mr)))
    else:
        overall, overall_n = np.nan, 0

    rows.append({
        "model": model,
        "videos_with_llm": int(g["video_id_key"].nunique()),
        "accuracy_humor_presence": hp_acc, "n_humor_presence": hp_n,
        "accuracy_humor_type": ht_acc, "n_humor_type": ht_n,
        "accuracy_stand_up": su_acc, "n_stand_up": su_n,
        "exact_set_match_joke_topic": jt_esm, "n_joke_topic": jt_n,
        "exact_set_match_rhetorical_device": rd_esm, "n_rhetorical_device": rd_n,
        "mae_dark_intensity": di_mae, "n_dark_intensity": di_n,
        "overall_mean_match_rate": overall, "n_match_rate": overall_n
    })

summary = pd.DataFrame(rows).sort_values("model").reset_index(drop=True)

# round for readability
for c in summary.columns:
    if c not in ["model","videos_with_llm","n_humor_presence","n_humor_type","n_stand_up","n_joke_topic","n_rhetorical_device","n_dark_intensity","n_match_rate"]:
        summary[c] = pd.to_numeric(summary[c], errors="coerce").round(4)

summary.to_csv(OUT_SUMMARY, index=False)
print("Wrote:", OUT_SUMMARY)
print(summary.to_string(index=False))

# -------------------------
# ACCURACY BLOCK
# -------------------------
acc_cols = [
    "model","videos_with_llm",
    "accuracy_humor_presence","accuracy_humor_type","accuracy_stand_up",
    "exact_set_match_joke_topic","exact_set_match_rhetorical_device",
    "mae_dark_intensity","overall_mean_match_rate"
]
acc_view = summary[acc_cols].copy()

print("\n=== Accuracy by model (sorted by overall) ===")
print(acc_view.sort_values("overall_mean_match_rate", ascending=False).to_string(index=False))