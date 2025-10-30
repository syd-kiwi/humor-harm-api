import csv
import os
import subprocess
from pathlib import Path

# ==== user settings ====
CSV_PATH = "youtube_shorts_cc_2023_2025.csv"   # path to your output CSV
OUTPUT_DIR = "downloads"                        # base folder for downloads
USE_TOPIC_SUBFOLDERS = True                     # put files into topic subfolders if column exists
ONLY_FIRST_N = None                             # set to an int to limit downloads, or None for all
# ========================

def yt_url(video_id: str) -> str:
    # watch URL works for Shorts and avoids occasional Shorts redirection issues
    return f"https://www.youtube.com/watch?v={video_id.strip()}"

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def run_download(url: str, out_dir: Path):
    """
    Runs:
      yt-dlp -v -f "bv*[height<=720][ext=mp4]+ba*[ext=m4a]" -N 4 \
             --merge-output-format mp4 -o "%(id)s.%(ext)s" -P <out_dir> <url>
    """
    cmd = [
        "yt-dlp",
        "-v",
        "-f", "bv*[height<=720][ext=mp4]+ba*[ext=m4a]",
        "-N", "4",
        "--no-playlist",
        "--continue",
        "--no-overwrites",
        "--merge-output-format", "mp4",
        "-o", "%(id)s.%(ext)s",
        "-P", str(out_dir),
        url,
    ]
    # print("Running:", " ".join(cmd))  # uncomment for debug
    subprocess.run(cmd, check=False)

def main():
    if not os.path.exists(CSV_PATH):
        raise SystemExit(f"CSV not found: {CSV_PATH}")

    base_out = Path(OUTPUT_DIR)
    ensure_dir(base_out)

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            vid = (row.get("videoId") or "").strip()
            if not vid:
                continue

            # choose subfolder by topic if present
            if USE_TOPIC_SUBFOLDERS and "topic" in row and row["topic"].strip():
                out_dir = base_out / row["topic"].strip().replace("/", "_")
            else:
                out_dir = base_out
            ensure_dir(out_dir)

            # skip if a file with this id already exists in the target folder
            existing = list(out_dir.glob(f"{vid}.*"))
            if existing:
                # already downloaded and merged
                continue

            run_download(yt_url(vid), out_dir)

            count += 1
            if ONLY_FIRST_N is not None and count >= ONLY_FIRST_N:
                break

    print("Done.")

if __name__ == "__main__":
    main()
