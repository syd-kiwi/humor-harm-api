import subprocess
import time
import os

INPUT_PATH = "/home/kiwi-pandas/Documents/humor-harm-api/scripts/pull_yt_data/uploaded-videos.txt"
OUT_DIR = "/home/kiwi-pandas/Documents/humor-harm-api/comments"

os.makedirs(OUT_DIR, exist_ok=True)

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.split()
        mp4 = next((p for p in reversed(parts) if p.endswith(".mp4")), None)
        if not mp4:
            continue

        video_id = mp4[:-4]
        out_file = os.path.join(OUT_DIR, f"{video_id}.json")

        # already have it
        if os.path.exists(out_file):
            print("HAVE:", out_file)
            continue

        # choose correct mode
        if video_id.startswith("-"):
            cmd = [
                "youtube-comment-downloader",
                f"--youtubeid={video_id}",
                f"--output={out_file}",
                "--limit", "30",
            ]
        else:
            cmd = [
                "youtube-comment-downloader",
                "--url", f"https://www.youtube.com/watch?v={video_id}",
                "--output", out_file,
                "--limit", "30",
            ]

        print("RUN:", " ".join(cmd))

        r = subprocess.run(cmd, text=True, capture_output=True)
        if r.returncode != 0:
            print("FAIL:", video_id)
            with open(os.path.join(OUT_DIR, "failures.txt"), "a", encoding="utf-8") as log:
                log.write(f"{video_id}\tcode={r.returncode}\n{r.stderr}\n---\n")
        else:
            print("OK:", video_id)

        time.sleep(1)
