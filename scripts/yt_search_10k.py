import os
import pandas as pd
import subprocess
import time

folder = "keyword_searches"
download_dir = "/media/kiwi-pandas/Extreme SSD/downloads"
webm_dir = os.path.join(download_dir, "webm")  # folder where .webm files are stored
os.makedirs(download_dir, exist_ok=True)

high_views_list = []

for file in os.listdir(folder):
    if file.endswith(".csv"):
        path = os.path.join(folder, file)
        df = pd.read_csv(path)

        if len(df) < 18:
            print(f"{file} has only {len(df)} rows")

        filtered = df[df["view_count"] > 10000]
        if not filtered.empty:
            filtered["source_file"] = file
            high_views_list.append(filtered)

if high_views_list:
    high_views_df = pd.concat(high_views_list, ignore_index=True)
else:
    high_views_df = pd.DataFrame()

# get already downloaded IDs (without extension)
downloaded = {
    os.path.splitext(f)[0] for f in os.listdir(webm_dir)
    if f.endswith(".webm") or f.endswith(".mp4")
}

for vid in high_views_df["video_id"]:
    if vid in downloaded:
        print(f"Skipping {vid} (already downloaded)")
        continue

    url = f"https://www.youtube.com/shorts/{vid}"
    print(f"Downloading {url}...")

    subprocess.run([
        "yt-dlp",
        "-f", "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "-o", f"{download_dir}/%(id)s.%(ext)s",
        url
    ])

    time.sleep(3)

print(f"\n✅ All new videos downloaded to: {download_dir}")

