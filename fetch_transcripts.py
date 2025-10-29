#!/usr/bin/env python3
"""
fetch_transcripts.py

Fetch English transcripts for a list of YouTube URLs, skipping failures.
Prefers human-made subtitles via yt-dlp; falls back to auto-captions;
finally tries youtube_transcript_api as a last resort.

Output CSV columns: duration, error, language, license, transcript, video_url

USAGE EXAMPLES
--------------
# From a plain text file (one URL per line)
python fetch_transcripts.py --urls input_urls.txt --out transcripts_clean.csv

# From a CSV with a 'video_url' column
python fetch_transcripts.py --csv input.csv --url-col video_url --out transcripts_clean.csv

# Force auto-captions if no human captions are available
python fetch_transcripts.py --urls input_urls.txt --allow-auto --out transcripts_clean.csv

# Specify preferred language(s) by priority (comma separated)
python fetch_transcripts.py --urls input_urls.txt --lang en,en-US

DEPENDENCIES
------------
pip install yt-dlp youtube-transcript-api pandas
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Lazy imports inside functions for environments without the libs
def _import_yt_dlp():
    try:
        from yt_dlp import YoutubeDL
        return YoutubeDL
    except Exception as e:
        raise RuntimeError("yt-dlp is not installed. Install with: pip install yt-dlp") from e

def _import_yta():
    try:
        from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
        return YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
    except Exception as e:
        raise RuntimeError("youtube-transcript_api is not installed. Install with: pip install youtube-transcript-api") from e

def parse_args():
    p = argparse.ArgumentParser(description="Fetch transcripts for YouTube URLs and skip failures.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--urls", type=str, help="Path to a .txt file with one URL per line.")
    src.add_argument("--csv", type=str, help="Path to a CSV file that contains a column with URLs.")
    p.add_argument("--url-col", type=str, default="video_url", help="Column name containing URLs when using --csv.")
    p.add_argument("--out", type=str, default="transcripts_clean.csv", help="Output CSV path.")
    p.add_argument("--lang", type=str, default="en", help="Preferred languages by priority, comma separated (e.g., 'en,en-US').")
    p.add_argument("--allow-auto", action="store_true", help="Allow auto captions if human captions not available.")
    p.add_argument("--max-duration", type=int, default=65, help="Skip videos longer than this many seconds (default 65).")
    p.add_argument("--verbose", action="store_true", help="Print detailed progress.")
    return p.parse_args()

def read_urls_from_txt(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def read_urls_from_csv(path: Path, col: str) -> List[str]:
    df = pd.read_csv(path)
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in {path}. Columns are: {list(df.columns)}")
    return [u for u in df[col].astype(str).tolist() if u and u.startswith("http")]

def pick_lang_track(tracks: Dict[str, List[Dict]], langs: List[str]) -> Tuple[Optional[str], Optional[Dict]]:
    """
    Given yt-dlp tracks dict (either 'subtitles' or 'automatic_captions'), pick the first matching language.
    Returns (picked_lang, track_info_dict) or (None, None).
    """
    if not tracks:
        return None, None
    for lng in langs:
        for key in tracks.keys():
            # yt-dlp may use keys like 'en', 'en-US', 'en-GB'
            if key.lower() == lng.lower():
                # Prefer vtt if present, else first format
                formats = tracks[key]
                vtt = next((f for f in formats if f.get("ext") == "vtt"), None)
                return key, (vtt or (formats[0] if formats else None))
    # Fallback: pick any available language with vtt, else any
    for key, formats in tracks.items():
        vtt = next((f for f in formats if f.get("ext") == "vtt"), None)
        if vtt:
            return key, vtt
        if formats:
            return key, formats[0]
    return None, None

def download_caption_text(url: str, allow_auto: bool, preferred_langs: List[str], verbose: bool=False) -> Tuple[Optional[str], Optional[str], Optional[int], Optional[str], Optional[str]]:
    """
    Try to fetch caption text and metadata via yt-dlp first (subtitles then auto captions).
    If that fails or unavailable, fall back to youtube_transcript_api.
    Returns: (transcript_text, language, duration, license, error)
    """
    YoutubeDL = _import_yt_dlp()
    ydl_opts = {
        "skip_download": True,
        "quiet": not verbose,
        "nocheckcertificate": True,
        "writesubtitles": True,
        "writeautomaticsub": allow_auto,
        "subtitlesformat": "vtt",
    }

    transcript_text = None
    picked_lang = None
    duration = None
    license_name = None

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            duration = info.get("duration")
            license_name = info.get("license") or info.get("license_url") or info.get("license_name")

            # Skip too long videos if required
            # (Helps when you're targeting Shorts and want to avoid long-form.)
            if duration is not None and args.max_duration and duration > args.max_duration:
                return None, None, duration, license_name, f"Skipped: duration {duration}s > {args.max_duration}s"

            subtitles = info.get("subtitles") or {}
            autocs = info.get("automatic_captions") or {}

            # 1) Try human-made subtitles
            picked_lang, track = pick_lang_track(subtitles, preferred_langs)
            if not track and allow_auto:
                # 2) Try auto-captions
                picked_lang, track = pick_lang_track(autocs, preferred_langs)

            if track and "url" in track:
                # Download the VTT file content and strip timestamps for plain text
                import urllib.request
                with urllib.request.urlopen(track["url"]) as resp:
                    vtt_data = resp.read().decode("utf-8", errors="replace")
                transcript_text = vtt_to_plaintext(vtt_data)
                if transcript_text.strip():
                    return transcript_text, picked_lang, duration, license_name, None
    except Exception as e:
        if verbose:
            print(f"[yt-dlp] Failed for {url}: {e}", file=sys.stderr)

    # 3) Fallback: youtube_transcript_api (structured segments)
    try:
        YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable = _import_yta()
        vid = youtube_id_from_url(url)
        segs = YouTubeTranscriptApi.get_transcript(vid, languages=preferred_langs + ["en"])
        transcript_text = " ".join([s["text"] for s in segs if s.get("text")])
        if transcript_text.strip():
            # Language is not guaranteed; best-effort
            return transcript_text, picked_lang or "en", duration, license_name, None
        else:
            return None, None, duration, license_name, "Empty transcript from youtube_transcript_api"
    except Exception as e:
        return None, None, duration, license_name, f"Fallback failed: {type(e).__name__}: {e}"

def vtt_to_plaintext(vtt: str) -> str:
    """
    Very simple VTT to plain text: drops header, timestamps, and cues.
    """
    lines = []
    for line in vtt.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("WEBVTT") or "-->" in line or line.isdigit():
            continue
        # Remove position/line attrs if present after timestamp lines
        if line.lower().startswith("note") or line.lower().startswith("style"):
            continue
        lines.append(line)
    return " ".join(lines)

def youtube_id_from_url(url: str) -> str:
    """
    Extract YouTube video id from common URL patterns.
    """
    import re
    patterns = [
        r"v=([A-Za-z0-9_-]{11})",
        r"youtu\.be/([A-Za-z0-9_-]{11})",
        r"shorts/([A-Za-z0-9_-]{11})",
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    raise ValueError(f"Could not extract video id from URL: {url}")

def main(args):
    # Collect URLs
    if args.urls:
        urls = read_urls_from_txt(Path(args.urls))
    else:
        urls = read_urls_from_csv(Path(args.csv), args.url_col)

    preferred_langs = [s.strip() for s in args.lang.split(",") if s.strip()]

    rows = []
    for url in urls:
        if args.verbose:
            print(f"Processing: {url}")
        text, lang, duration, license_name, err = download_caption_text(
            url=url,
            allow_auto=args.allow_auto,
            preferred_langs=preferred_langs,
            verbose=args.verbose
        )
        rows.append({
            "duration": duration if duration is not None else "",
            "error": "" if err is None else err,
            "language": "" if lang is None else lang,
            "license": "" if license_name is None else str(license_name),
            "transcript": "" if text is None else text,
            "video_url": url
        })

    # Keep only rows with non-empty transcript
    cleaned = [r for r in rows if r["transcript"]]
    if args.verbose:
        print(f"Total URLs: {len(rows)} | With transcript: {len(cleaned)}")

    # Write output
    out_path = Path(args.out)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["duration", "error", "language", "license", "transcript", "video_url"])
        writer.writeheader()
        writer.writerows(cleaned)

    # Also write a separate 'all rows' CSV so you can see failures alongside successes
    out_all = out_path.with_name(out_path.stem + "_all.csv")
    with out_all.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["duration", "error", "language", "license", "transcript", "video_url"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote cleaned transcripts: {out_path}")
    print(f"Wrote all attempts (incl. failures): {out_all}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
