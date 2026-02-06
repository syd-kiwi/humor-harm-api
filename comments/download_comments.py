import subprocess

INPUT_PATH = "/home/kiwi-pandas/Documents/humor-harm-api/scripts/uploaded-videos.txt"

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.split()
        mp4 = next((p for p in reversed(parts) if p.endswith(".mp4")), None)
        if not mp4:
            continue

        video_id = mp4[:-4]  # remove ".mp4"

        if video_id.startswith("-"):
            print("SKIP:", video_id)
            continue

        out_file = f"{video_id}.json"
        cmd = [
            "youtube-comment-downloader",
            "--url", f"https://www.youtube.com/watch?v={video_id}",
            "--output", out_file,
            "--limit", "30",
        ]

        print("RUN:", " ".join(cmd))
        subprocess.run(cmd, check=False)
