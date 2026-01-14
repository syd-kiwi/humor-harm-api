#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path

LOCAL_DIR_DEFAULT = "/media/kiwi-pandas/Extreme SSD/downloads/mp4"
S3_DEST_DEFAULT = "s3://humor-harm-dataset/dataset-s3/"
OUT_DEFAULT = "to_upload.txt"



def load_uploaded_names(uploaded_file: Path) -> set[str]:
    uploaded = set()
    for line in uploaded_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s:
            continue
        low = s.lower()
        if "lastmodified" in low and "size" in low and "name" in low:
            continue
        if low.startswith("total"):
            continue
        parts = s.split()
        if parts:
            uploaded.add(parts[-1])  # filename is last column
    return uploaded


def aws_cp(local_path: Path, s3_dest: str, dry_run: bool) -> None:
    cmd = ["aws", "s3", "cp", str(local_path), s3_dest]
    if dry_run:
        print("DRY RUN:", " ".join(cmd))
        return
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uploaded", required=True, help="Path to uploaded-videos.txt")
    ap.add_argument("--n", type=int, required=True, help="How many files to upload")
    ap.add_argument("--local-dir", default=LOCAL_DIR_DEFAULT, help="Local mp4 folder")
    ap.add_argument("--s3-dest", default=S3_DEST_DEFAULT, help="S3 destination prefix")
    ap.add_argument("--out", default=OUT_DEFAULT, help="Write picked filenames here")
    ap.add_argument("--dry-run", action="store_true", help="Print commands only")
    args = ap.parse_args()

    uploaded_file = Path(args.uploaded).expanduser()
    local_dir = Path(args.local_dir)

    if not uploaded_file.exists():
        raise SystemExit(f"Missing uploaded file: {uploaded_file}")
    if not local_dir.exists():
        raise SystemExit(f"Missing local dir: {local_dir}")
    if args.n <= 0:
        raise SystemExit("--n must be > 0")

    s3_dest = args.s3_dest
    if not s3_dest.endswith("/"):
        s3_dest += "/"

    uploaded_names = load_uploaded_names(uploaded_file)

    local_files = sorted([p for p in local_dir.glob("*.mp4") if p.is_file()], key=lambda p: p.name.lower())
    not_uploaded = [p for p in local_files if p.name not in uploaded_names]
    picked = not_uploaded[: args.n]

    out_path = Path(args.out).expanduser()
    out_path.write_text("\n".join([p.name for p in picked]) + ("\n" if picked else ""), encoding="utf-8")

    print(f"Uploaded listed: {len(uploaded_names)}")
    print(f"Local mp4s found: {len(local_files)}")
    print(f"Not uploaded: {len(not_uploaded)}")
    print(f"Picked: {len(picked)}")
    print(f"Wrote: {out_path}")

    if not picked:
        print("Nothing to upload.")
        return

    # Upload
    for i, p in enumerate(picked, start=1):
        print(f"[{i}/{len(picked)}] Uploading {p.name}")
        aws_cp(p, s3_dest, args.dry_run)

    print("Done.")


if __name__ == "__main__":
    main()
