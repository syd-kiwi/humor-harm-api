#!/usr/bin/env python3
"""
YouTube Shorts metadata + Captions (Transcripts) Collector
Includes public video URL for annotation tools (e.g., Label Studio)
------------------------------------------------------------------

- Searches for candidate Shorts (<= 60s + #shorts unless --include-no-tag)
- Requires captions per API (contentDetails.caption == "true")
- Fetches transcript text via youtube-transcript-api (preferred languages)
- Outputs CSV with: videoId, videoUrl, title, description, view/like/comment counts, transcriptText, etc.

Setup:
    pip install google-api-python-client packaging youtube-transcript-api
    export YT_API_KEY="YOUR_KEY"

Usage:
    python youtube_shorts_metadata_captions.py --queries "comedy" "pranks" --max-results 200 --region US --languages en es
"""

import os
import csv
import time
import math
import argparse
from typing import List, Dict, Any

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

# ----------------------------- CLI ---------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Collect YouTube Shorts metadata (captions-only) with transcripts and video URL")
    p.add_argument("--api-key", type=str, default=os.getenv("YT_API_KEY", ""), help="YouTube Data API v3 key")
    p.add_argument("--queries", nargs="+", required=True, help="Search topics, e.g., comedy pranks memes")
    p.add_argument("--published-after", type=str, default="2023-10-01T00:00:00Z", help="Start date (ISO 8601). Default Oct 1 2023")
    p.add_argument("--published-before", type=str, default="2024-10-01T00:00:00Z", help="End date (ISO 8601). Default Oct 1 2024")
    p.add_argument("--region", type=str, default="US", help="Region code, e.g., US, GB, IN")
    p.add_argument("--relevance-language", type=str, default=None, help="Language hint, e.g., en")
    p.add_argument("--max-results", type=int, default=1000, help="Target total results across all queries (search step)")
    p.add_argument("--per-page", type=int, default=50, help="Results per page (max 50)")
    p.add_argument("--sleep", type=float, default=0.2, help="Pause (seconds) between API calls")
    p.add_argument("--out", type=str, default="youtube_shorts_metadata_captions.csv", help="Output CSV path")
    p.add_argument("--include-no-tag", action="store_true", help="Also include videos without #shorts tag if duration <= 60s")
    p.add_argument("--languages", nargs="+", default=["en"], help="Preferred transcript languages in order, e.g., en es fr")
    return p.parse_args()

# --------------------------- Helpers --------------------------------

def duration_to_seconds(iso_duration: str) -> int:
    total, num = 0, ""
    for c in iso_duration:
        if c.isdigit():
            num += c
            continue
        if c == "H" and num:
            total += int(num) * 3600; num = ""
        elif c == "M" and num:
            total += int(num) * 60; num = ""
        elif c == "S" and num:
            total += int(num); num = ""
    return total

def get_transcript_text(video_id: str, langs: List[str]) -> str:
    # Try preferred languages first; fall back to any available transcript
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=langs)
        text = " ".join([seg.get("text", "") for seg in transcript]).strip()
        return " ".join(text.split())
    except (TranscriptsDisabled, NoTranscriptFound):
        try:
            available = YouTubeTranscriptApi.list_transcripts(video_id)
            # Prefer user-requested languages
            for code in langs:
                if code in [t.language_code for t in available]:
                    t = available.find_transcript([code])
                    text = " ".join([seg.get("text", "") for seg in t.fetch()]).strip()
                    return " ".join(text.split())
            # Otherwise fetch the first available transcript (may be auto/translated)
            for t in available:
                try:
                    text = " ".join([seg.get("text", "") for seg in t.fetch()]).strip()
                    return " ".join(text.split())
                except Exception:
                    continue
        except Exception:
            pass
    except Exception:
        pass
    return ""

# ---------------------------- API calls ------------------------------

def search_shorts(service, query: str, opts) -> List[str]:
    ids: List[str] = []
    next_page = None
    fetched = 0
    target = max(1, math.ceil(opts.max_results / max(1, len(opts.queries))))

    while True:
        try:
            params = dict(
                part="id,snippet",
                q=query + " #shorts",
                type="video",
                maxResults=min(opts.per_page, 50),
                order="date",
                regionCode=opts.region,
                videoDuration="short",  # < 4 min
                publishedAfter=opts.published_after,
                publishedBefore=opts.published_before,
                relevanceLanguage=opts.relevance_language,
            )
            if next_page:
                params["pageToken"] = next_page

            data = service.search().list(**params).execute()
        except HttpError as e:
            print("HTTP error during search:", e)
            break

        for item in data.get("items", []):
            vid = item["id"]["videoId"]
            ids.append(vid)
            fetched += 1
            if fetched >= target:
                break

        if fetched >= target:
            break

        next_page = data.get("nextPageToken")
        if not next_page:
            break

        time.sleep(opts.sleep)

    return ids

def fetch_details(service, video_ids: List[str], include_no_tag: bool, langs: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        try:
            resp = service.videos().list(
                part="snippet,contentDetails,statistics,status",
                id=",".join(batch)
            ).execute()
        except HttpError as e:
            print("HTTP error during videos.list:", e)
            continue

        for v in resp.get("items", []):
            snippet = v.get("snippet", {}) or {}
            details = v.get("contentDetails", {}) or {}
            stats = v.get("statistics", {}) or {}
            status = v.get("status", {}) or {}

            title = snippet.get("title", "") or ""
            desc = snippet.get("description", "") or ""
            duration_iso = details.get("duration", "PT0S")
            seconds = duration_to_seconds(duration_iso)

            # Shorts heuristic
            combined_text = (title + " " + desc).lower()
            has_shorts_tag = "#shorts" in combined_text
            is_short_by_time = seconds <= 60
            if not (is_short_by_time and (has_shorts_tag or include_no_tag)):
                continue

            vid = v.get("id")
            transcript_text = get_transcript_text(vid, langs)
            if not transcript_text:
                continue  # captions-only requirement

            row = {
                "videoId": vid,
                "videoUrl": f"https://www.youtube.com/watch?v={vid}",  # Added for Label Studio
                "title": title,
                "description": desc,
                "channelId": snippet.get("channelId"),
                "channelTitle": snippet.get("channelTitle"),
                "publishedAt": snippet.get("publishedAt"),
                "durationSeconds": seconds,
                "viewCount": stats.get("viewCount"),
                "likeCount": stats.get("likeCount"),
                "commentCount": stats.get("commentCount"),
                "definition": details.get("definition"),
                "licensedContent": details.get("licensedContent"),
                "license": status.get("license"),
                "madeForKids": status.get("madeForKids"),
                "tagsCount": len(snippet.get("tags", [])) if snippet.get("tags") else 0,
                "hasShortsTag": has_shorts_tag,
                "isShortByTime": is_short_by_time,
                "transcriptText": transcript_text,
            }
            rows.append(row)

        time.sleep(0.1)

    return rows

def write_csv(path: str, rows: List[Dict[str, Any]]):
    if not rows:
        print("No rows to write")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {len(rows)} rows to {path}")

# ----------------------------- Main ---------------------------------

def main():
    opts = parse_args()
    if not opts.api_key:
        raise SystemExit("Missing API key. Provide --api-key or set YT_API_KEY")

    service = build("youtube", "v3", developerKey=opts.api_key)

    all_ids: List[str] = []
    for q in opts.queries:
        ids = search_shorts(service, q, opts)
        print(f"Query '{q}' returned {len(ids)} candidate IDs")
        all_ids.extend(ids)

    # Deduplicate preserving order
    seen, deduped = set(), []
    for vid in all_ids:
        if vid not in seen:
            deduped.append(vid); seen.add(vid)
    print(f"After dedupe, {len(deduped)} IDs")

    rows = fetch_details(service, deduped, include_no_tag=opts.include_no_tag, langs=opts.languages)
    write_csv(opts.out, rows)

if __name__ == "__main__":
    main()
