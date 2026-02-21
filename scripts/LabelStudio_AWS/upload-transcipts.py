import os
import subprocess
import requests
import whisper
from tqdm import tqdm

LABEL_STUDIO_URL = "https://humor-harm-dataset.space"
PROJECT_ID = 1

S3_BUCKET = "humor-harm-dataset"
S3_PREFIX = "dataset-s3"

WHISPER_MODEL = "small"
WORKDIR = "/tmp/humor_harm_whisper_mp4"

def run_cmd(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{p.stderr}")
    return p.stdout

def list_s3_mp4_names(bucket: str, prefix: str) -> list[str]:
    out = run_cmd(["aws", "s3", "ls", f"s3://{bucket}/{prefix}/"])
    names = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 4:
            name = parts[3]
            if name.lower().endswith(".mp4"):
                names.append(name)
    return sorted(names)

def s3_download_mp4(bucket: str, prefix: str, name: str, dst: str) -> None:
    run_cmd(["aws", "s3", "cp", f"s3://{bucket}/{prefix}/{name}", dst])

def get_ls_tasks(api_token: str) -> list[dict]:
    headers = {"Authorization": f"Token {api_token}"}
    r = requests.get(
        f"{LABEL_STUDIO_URL}/api/projects/{PROJECT_ID}/tasks?page_size=5000",
        headers=headers,
        timeout=60,
    )
    r.raise_for_status()
    return r.json()

def patch_ls_task(api_token: str, task_id: int, new_data: dict) -> None:
    headers = {"Authorization": f"Token {api_token}", "Content-Type": "application/json"}
    r = requests.patch(
        f"{LABEL_STUDIO_URL}/api/tasks/{task_id}",
        headers=headers,
        json={"data": new_data},
        timeout=60,
    )
    r.raise_for_status()

def base_from_s3_video_uri(video_uri: str) -> str:
    # s3://bucket/prefix/ID.mp4  -> ID
    name = os.path.basename(video_uri)
    return os.path.splitext(name)[0]

def main():
    api_token = os.environ.get("LS_TOKEN")
    if not api_token:
        raise SystemExit("Set LS_TOKEN first: export LS_TOKEN='...'\n")

    os.makedirs(WORKDIR, exist_ok=True)

    print("Listing mp4s in S3...")
    mp4_names = list_s3_mp4_names(S3_BUCKET, S3_PREFIX)
    print(f"Found {len(mp4_names)} mp4s in s3://{S3_BUCKET}/{S3_PREFIX}/")

    print("Loading Label Studio tasks...")
    tasks = get_ls_tasks(api_token)
    print(f"Found {len(tasks)} tasks in project {PROJECT_ID}")

    # Map base -> (task_id, data)
    base_to_task = {}
    for t in tasks:
        data = t.get("data", {})
        video = data.get("video", "")
        if isinstance(video, str) and video.startswith("s3://") and video.lower().endswith(".mp4"):
            base = base_from_s3_video_uri(video)
            base_to_task[base] = (t["id"], data)

    print(f"Matched {len(base_to_task)} S3 video tasks in Label Studio")

    print("Loading Whisper model...")
    model = whisper.load_model(WHISPER_MODEL)

    updated = 0
    skipped_already_has = 0
    skipped_no_task = 0

    for name in tqdm(mp4_names, desc="Whisper + patch"):
        base = os.path.splitext(name)[0]

        if base not in base_to_task:
            skipped_no_task += 1
            continue

        task_id, data = base_to_task[base]

        # Do not overwrite if already present
        if data.get("transcript_text"):
            skipped_already_has += 1
            continue

        local_mp4 = os.path.join(WORKDIR, name)

        # Download mp4
        if not os.path.exists(local_mp4):
            s3_download_mp4(S3_BUCKET, S3_PREFIX, name, local_mp4)

        # Transcribe
        result = model.transcribe(local_mp4, language="en", task="transcribe")
        transcript = (result.get("text") or "").strip()

        # Patch task: add transcript_text, DO NOT TOUCH video
        new_data = dict(data)
        new_data["transcript_text"] = transcript

        patch_ls_task(api_token, task_id, new_data)
        updated += 1

        # Cleanup mp4 to save disk
        try:
            os.remove(local_mp4)
        except OSError:
            pass

    print("\nDone")
    print(f"Updated tasks: {updated}")
    print(f"Skipped (already had transcript_text): {skipped_already_has}")
    print(f"Skipped (no matching LS task for S3 mp4): {skipped_no_task}")

if __name__ == "__main__":
    main()
