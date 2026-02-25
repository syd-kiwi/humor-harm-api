import subprocess
import time
import os
import json
import csv

INPUT_PATH = "/home/kiwi-pandas/Documents/humor-harm-api/scripts/pull_yt_data/uploaded-videos.txt"
OUT_DIR = "/home/kiwi-pandas/Documents/humor-harm-api/descriptions"
OUT_CSV = os.path.join(OUT_DIR, "video_descriptions.csv")

os.makedirs(OUT_DIR, exist_ok=True)

FAIL_LOG = os.path.join(OUT_DIR, "failures.txt")

FIELDS = [
    "video_id",
    "url",
    "title",
    "description",
    "tags",
    "channel",
    "channel_id",
    "uploader",
    "uploader_id",
    "upload_date",
    "duration",
    "view_count",
    "like_count",
    "comment_count",
    "categories",
    "language",
]

def run_yt_dlp_json(url: str) -> dict:
    cmd = [
        "yt-dlp",
        "-J",
        "--no-warnings",
        "--no-playlist",
        url,
    ]
    r = subprocess.run(cmd, text=True, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or f"yt-dlp failed with code {r.returncode}")
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Could not parse yt-dlp JSON: {e}\nFirst 500 chars:\n{r.stdout[:500]}")

def normalize_list(val):
    # Store list fields safely in CSV
    if val is None:
        return ""
    if isinstance(val, list):
        return "|".join(str(x) for x in val)
    return str(val)

seen = set()
rows = []

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.split()
        mp4 = next((p for p in reversed(parts) if p.endswith(".mp4")), None)
        if not mp4:
            continue

        video_id = mp4[:-4]
        if video_id in seen:
            continue
        seen.add(video_id)

        url = f"https://www.youtube.com/watch?v={video_id}"
        print("RUN:", url)

        try:
            info = run_yt_dlp_json(url)

            row = {
                "video_id": video_id,
                "url": url,
                "title": info.get("title") or "",
                "description": info.get("description") or "",
                "tags": normalize_list(info.get("tags")),
                "channel": info.get("channel") or "",
                "channel_id": info.get("channel_id") or "",
                "uploader": info.get("uploader") or "",
                "uploader_id": info.get("uploader_id") or "",
                "upload_date": info.get("upload_date") or "",
                "duration": info.get("duration") or "",
                "view_count": info.get("view_count") or "",
                "like_count": info.get("like_count") or "",
                "comment_count": info.get("comment_count") or "",
                "categories": normalize_list(info.get("categories")),
                "language": info.get("language") or "",
            }

            rows.append(row)
            print("OK:", video_id)

        except Exception as e:
            print("FAIL:", video_id, "|", str(e)[:200])
            with open(FAIL_LOG, "a", encoding="utf-8") as log:
                log.write(f"{video_id}\t{url}\n{e}\n---\n")

        time.sleep(1)

# Write CSV
with open(OUT_CSV, "w", encoding="utf-8", newline="") as out:
    writer = csv.DictWriter(out, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)

print("WROTE:", OUT_CSV)
print("ROWS:", len(rows))
