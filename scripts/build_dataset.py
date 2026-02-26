import os
import re
import pandas as pd
import glob

ANNO_CSV = "/home/kiwi-pandas/Documents/humor-harm-api/annotation_dashboard/02-25.csv"
TASKS_CSV = "/home/kiwi-pandas/Documents/humor-harm-api/scripts/tasks_id_inner_id.csv"
OUT_CSV = "/home/kiwi-pandas/Documents/humor-harm-api/annotation_dashboard/02-25_minimal.csv"

S3_PREFIX = "s3://humor-harm-dataset/dataset-s3/"

def extract_video_id_from_s3(video_value: str):
    if video_value is None:
        return None
    s = str(video_value).strip().strip('"').strip("'")
    if not s or s.lower() == "nan":
        return None

    if s.startswith(S3_PREFIX):
        s = s[len(S3_PREFIX):]

    s = os.path.basename(s)
    s = re.sub(r"\.mp4$", "", s, flags=re.IGNORECASE)
    return s.strip() or None

# ---- read anno ----
df = pd.read_csv(ANNO_CSV, dtype=str)

'''
"annotation_id","annotator","created_at","dark_intensity","humor_presence","humor_type","id","joke_topic","lead_time","note","rhetorical_device","stand_up","target_category","transcript_text","updated_at","video"
'''

required = ["id","transcript_text","video","dark_intensity","humor_presence","humor_type","joke_topic","rhetorical_device","stand_up","target_category","note"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise SystemExit(f"Missing columns in {ANNO_CSV}: {missing}\nFound: {list(df.columns)}")

out = df[["id","transcript_text","video","dark_intensity","humor_presence","humor_type","joke_topic","rhetorical_device","stand_up","target_category","note"]].copy()
out["id"] = out["id"].astype(str).str.strip()
out["video_id"] = out["video"].apply(extract_video_id_from_s3)
out = out.drop(columns=["video"])

# One row per id: if duplicates exist, keep the first
# (If you want newest, we can sort by updated_at before this)
out = out.drop_duplicates(subset=["id"], keep="first")

print("anno rows after dedupe:", len(out))
print("anno unique ids:", out["id"].nunique())
print("anno unique video_id:", out["video_id"].nunique(dropna=True))
print("anno missing video_id:", out["video_id"].isna().sum())

# ---- read tasks (master ids) ----
tasks = pd.read_csv(TASKS_CSV, dtype=str)

if "id" not in tasks.columns:
    raise SystemExit(f"tasks file must have 'id' column. Found: {list(tasks.columns)}")

base = tasks[["id"]].copy()
base["id"] = base["id"].astype(str).str.strip()
base = base.drop_duplicates(subset=["id"], keep="first")

print("tasks rows:", len(base))
print("tasks unique ids:", base["id"].nunique())

# ---- left join to keep ALL task ids ----
merged = base.merge(out, on="id", how="left")

print("merged rows:", len(merged))
print("merged unique ids:", merged["id"].nunique())
print("missing transcript_text:", merged["transcript_text"].isna().sum())
print("missing video_id:", merged["video_id"].isna().sum())

merged.to_csv(OUT_CSV, index=False)
print("\nWROTE:", OUT_CSV)

# -------------------------------
# Add video descriptions (by video_id)
# -------------------------------

DESC_CSV = "/home/kiwi-pandas/Documents/humor-harm-api/descriptions/video_descriptions.csv"

desc = pd.read_csv(DESC_CSV, dtype=str)

if "video_id" not in desc.columns:
    raise SystemExit(f"video_descriptions.csv must contain 'video_id'. Found: {list(desc.columns)}")

# normalize video_id
desc["video_id"] = desc["video_id"].astype(str).str.strip()

# if duplicates exist in descriptions, keep first
desc = desc.drop_duplicates(subset=["video_id"], keep="first")

print("description overlap:", merged["video_id"].isin(desc["video_id"]).sum())

merged = merged.merge(desc, on="video_id", how="left")

print("rows after description merge:", len(merged))

# -------------------------------
# Add searched_keyword from keyword_searches folder
# -------------------------------

KEYWORD_DIR = "/home/kiwi-pandas/Documents/humor-harm-api/keyword_searches"

keyword_files = []
for ext in ["csv", "jsonl", "json", "txt"]:
    keyword_files.extend(glob.glob(os.path.join(KEYWORD_DIR, f"**/*.{ext}"), recursive=True))

kw_rows = []

def keyword_from_filename(path):
    name = os.path.basename(path)
    name = re.sub(r"\.(csv|jsonl|json|txt)$", "", name, flags=re.IGNORECASE)
    return name.strip()

for fp in sorted(keyword_files):
    keyword = keyword_from_filename(fp)
    ext = os.path.splitext(fp)[1].lower()

    try:
        if ext == ".csv":
            df = pd.read_csv(fp, dtype=str)
            if "video_id" not in df.columns:
                continue

            for vid in df["video_id"].dropna():
                kw_rows.append({"video_id": vid.strip(), "searched_keyword": keyword})

        elif ext == ".jsonl":
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        vid = obj.get("video_id")
                        if vid:
                            kw_rows.append({"video_id": vid.strip(), "searched_keyword": keyword})
                    except:
                        continue
    except Exception as e:
        print("skip file:", fp, e)

kw_df = pd.DataFrame(kw_rows)

if not kw_df.empty:
    kw_map = (
        kw_df.groupby("video_id")["searched_keyword"]
        .apply(lambda x: "; ".join(sorted(set(x))))
        .reset_index()
    )
else:
    kw_map = pd.DataFrame(columns=["video_id", "searched_keyword"])

print("keyword overlap:", merged["video_id"].isin(kw_map["video_id"]).sum())

merged = merged.merge(kw_map, on="video_id", how="left")

print("rows after keyword merge:", len(merged))

merged = merged.drop_duplicates(subset=["id"], keep="first")

print("FINAL rows:", len(merged))
print("unique ids:", merged["id"].nunique())

merged.to_csv("/home/kiwi-pandas/Documents/humor-harm-api/unified_dataset.csv", index=False)
print("WROTE unified_dataset.csv")

