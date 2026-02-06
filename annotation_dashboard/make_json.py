import os
import json

# Path to your downloaded videos
VIDEO_DIR = "/media/kiwi-pandas/Extreme SSD/downloads/mp4/"

# Output JSON
OUTPUT_JSON = "labelstudio_videos.json"

tasks = []
task_id = 1

for filename in sorted(os.listdir(VIDEO_DIR)):
    if filename.lower().endswith(".mp4"):
        full_path = os.path.join(VIDEO_DIR, filename)

        tasks.append({
            "id": task_id,
            "data": {
                "video_url": full_path
            }
        })

        task_id += 1

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(tasks, f, indent=2)

print(f"Created {OUTPUT_JSON} with {len(tasks)} video tasks.")
