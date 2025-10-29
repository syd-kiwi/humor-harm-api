#!/usr/bin/env python3
import os, csv, sys, tempfile, subprocess, re
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# === Config ===
INPUT_CSV = "/home/kiwi-pandas/Documents/humor-harm-api/youtube_shorts_metadata.csv"  # your uploaded file
OUTPUT_CSV = "transcripts.csv"
MAX_VIDEOS = None              # None or an int
REQUIRE_CC = True              # keep True if you really need CC; unknowns won't be skipped
MAX_SECONDS = 60               # None to disable; unknowns won't be skipped
AUDIO_FORMAT = "m4a"
WHISPER_MODEL = "base"
LANG_HINT = None

# === Helpers ===
def have_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False

def pick_url_id(df: pd.DataFrame):
    lc = {c.lower(): c for c in df.columns}
    url_cols = ["url","video_url","link","video link","video-link","shorts_url","short_url","watch_url","webpage_url"]
    id_cols  = ["videoId","id","youtube_id","yt_id"]
    url_col = next((lc[c] for c in (c.lower() for c in url_cols) if c in lc), None)
    id_col  = next((lc[c] for c in (c.lower() for c in id_cols)  if c in lc), None)
    return url_col, id_col

YT_RE = re.compile(r"(https?://)?(www\.)?(youtube\.com|youtu\.be)/")

def extract_first_youtube_url_from_row(row_dict):
    for v in row_dict.values():
        if isinstance(v, str) and YT_RE.search(v):
            return v.strip()
    return None

def to_full_url(url_val, vid_val):
    if isinstance(url_val, str) and url_val.strip():
        return url_val.strip()
    if isinstance(vid_val, str) and vid_val.strip():
        vid = vid_val.strip()
        # watch form works for both shorts and normal
        return f"https://www.youtube.com/watch?v={vid}"
    return None

def ytdlp_json(url):
    """Return info dict from yt-dlp --dump-json (no download)."""
    cmd = [sys.executable, "-m", "yt_dlp", "--no-warnings", "--dump-single-json", "--no-playlist", url]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        return {}
    import json
    try:
        return json.loads(p.stdout)
    except Exception:
        return {}

def license_allows_cc(lic):
    if lic is None:
        return None  # unknown
    s = str(lic).lower()
    # yt-dlp reports "Creative Commons Attribution license (reuse allowed)" or similar
    return ("creative" in s and "common" in s) or s in {"cc","cc-by","creative_commons"}

def passes_duration_limit(dur):
    if MAX_SECONDS is None:
        return True
    if dur is None:
        return True  # unknown duration: don't skip
    try:
        return float(dur) <= float(MAX_SECONDS)
    except Exception:
        return True  # unknown/invalid: don't skip

def download_audio_to_temp(url):
    tmp = tempfile.NamedTemporaryFile(prefix="yt_", suffix=f".{AUDIO_FORMAT}", delete=False)
    tmp_path = tmp.name
    tmp.close()
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", "bestaudio/best",
        "-x", "--audio-format", AUDIO_FORMAT,
        "-o", tmp_path,
        "--no-playlist",
        url,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        try: os.remove(tmp_path)
        except: pass
        raise RuntimeError(f"yt-dlp failed: {p.stderr.strip()}")
    return tmp_path

def run_whisper(file_path, language_hint=None, model_name=WHISPER_MODEL):
    import whisper
    model = whisper.load_model(model_name)
    kwargs = {}
    if language_hint:
        kwargs["language"] = language_hint
    return model.transcribe(file_path, **kwargs)

# === Main ===
def main():
    if not have_ffmpeg():
        print("ERROR: ffmpeg not found in PATH.", file=sys.stderr); sys.exit(1)
    if not Path(INPUT_CSV).exists():
        print(f"ERROR: {INPUT_CSV} not found.", file=sys.stderr); sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    url_col, id_col = pick_url_id(df)

    out_rows = []
    skipped_no_url = skipped_cc = skipped_dur = 0

    it = df.itertuples(index=False)
    total_rows = len(df)
    processed = 0

    for row in tqdm(it, total=total_rows, desc="Transcribing"):
        if MAX_VIDEOS is not None and processed >= MAX_VIDEOS:
            break

        row_dict = row._asdict() if hasattr(row, "_asdict") else dict(zip(df.columns, row))
        url = to_full_url(row_dict.get(url_col), row_dict.get(id_col))

        if not url:
            # last-chance: scan any column for a youtube link
            url = extract_first_youtube_url_from_row(row_dict)
        if not url:
            skipped_no_url += 1
            processed += 1
            continue

        # probe metadata first (license + duration)
        meta = ytdlp_json(url)
        lic = meta.get("license")
        dur = meta.get("duration")

        # license check
        if REQUIRE_CC:
            cc_ok = license_allows_cc(lic)
            if cc_ok is False:  # explicit non-CC
                skipped_cc += 1
                processed += 1
                continue
            # cc_ok True or None (unknown) are allowed

        # duration check
        if not passes_duration_limit(dur):
            skipped_dur += 1
            processed += 1
            continue

        try:
            audio_fp = download_audio_to_temp(url)
            result = run_whisper(audio_fp, language_hint=LANG_HINT, model_name=WHISPER_MODEL)
            text = (result.get("text") or "").strip()
            language = result.get("language")
            out = {
                "video_url": url,
                "transcript": text,
                "language": language,
                "duration": dur,
                "license": lic,
                "title": meta.get("title"),
                "channel": meta.get("uploader"),
                "channel_id": meta.get("channel_id"),
                "video_id": meta.get("id"),
            }
            out_rows.append(out)
        except Exception as e:
            out_rows.append({
                "video_url": url,
                "transcript": "",
                "language": None,
                "duration": dur,
                "license": lic,
                "error": str(e)[:1000],
            })
        finally:
            try:
                if 'audio_fp' in locals() and audio_fp and os.path.exists(audio_fp):
                    os.remove(audio_fp)
            except:
                pass

        processed += 1

    if out_rows:
        fieldnames = sorted({k for r in out_rows for k in r.keys()})
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in out_rows:
                w.writerow(r)
        print(f"Saved {len(out_rows)} rows to {OUTPUT_CSV}")
        print(f"Skipped: no_url={skipped_no_url}, non_cc={skipped_cc}, over_duration={skipped_dur}")
    else:
        print("No rows to write.")
        print(f"Skipped: no_url={skipped_no_url}, non_cc={skipped_cc}, over_duration={skipped_dur}")
        print("Tip: set REQUIRE_CC=False or MAX_SECONDS=None to test end-to-end.")
        
if __name__ == "__main__":
    main()
