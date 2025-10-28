#!/usr/bin/env python3
"""
YouTube Shorts metadata collector using the official YouTube Data API v3.

Features
- Searches for videos with the #Shorts tag and duration <= 60 seconds
- Pulls details in batches and writes to CSV
- Allows topic keywords and date filters
- Safe for ToS if used only for metadata and public thumbnails

Usage
  1. Create a Google Cloud project and enable YouTube Data API v3
  2. Create an API key and set YT_API_KEY environment variable or put it in --api-key
  3. Run:
     python youtube_shorts_scraper.py --queries "comedy" "pranks" --max-results 500 --region US
"""

import os
import csv
import time
import math
import argparse
from datetime import datetime, timezone
from typing import List, Dict, Any

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

ISO_8601_ZERO = "1970-01-01T00:00:00Z"

def parse_args():
    p = argparse.ArgumentParser(description="Collect YouTube Shorts metadata into CSV")
    p.add_argument("--api-key", type=str, default=os.getenv("YT_API_KEY", ""), help="YouTube Data API v3 key")
    p.add_argument("--queries", nargs="+", required=True, help="Search queries or topics. Example: comedy pranks memes")
    p.add_argument("--published-after", type=str, default="2023-10-01T00:00:00Z", help="Start date: include videos posted after Oct 1, 2023")
    p.add_argument("--published-before", type=str, default="2024-10-01T00:00:00Z", help="End date: include videos posted before Oct 1, 2024")
    p.add_argument("--region", type=str, default="US", help="Region code. Example US, GB, IN")
    p.add_argument("--relevance-language", type=str, default=None, help="Language code. Example en")
    p.add_argument("--max-results", type=int, default=1000, help="Target total results across all queries")
    p.add_argument("--per-page", type=int, default=50, help="API page size up to 50")
    p.add_argument("--sleep", type=float, default=0.2, help="Seconds to sleep between API calls")
    p.add_argument("--out", type=str, default="youtube_shorts_metadata.csv", help="Output CSV path")
    p.add_argument("--include-no-tag", action="store_true", help="Also include videos without #shorts tag if duration <= 60s")
    return p.parse_args()

def iso_date_or_none(s: str):
    if not s:
        return None
    if len(s) == 10:
        return s + "T00:00:00Z"
    return s

def duration_to_seconds(iso_duration: str) -> int:
    # ISO 8601 duration like PT53S or PT1M2S
    total = 0
    num = ""
    in_time = False
    for c in iso_duration:
        if c.isdigit():
            num += c
            continue
        if c == "T":
            in_time = True
            continue
        if c in ("H","M","S"):
            if not num:
                continue
            val = int(num)
            if c == "H":
                total += val * 3600
            elif c == "M":
                total += val * 60
            elif c == "S":
                total += val
            num = ""
    return total

def search_shorts(service, query: str, opts) -> List[str]:
    """Return a list of video IDs that look like Shorts for a given query."""
    ids = []
    next_page = None
    fetched = 0
    target = math.ceil(opts.max_results / max(1, len(opts.queries)))
    published_after = iso_date_or_none(opts.published_after)
    published_before = iso_date_or_none(opts.published_before) or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    while True:
        try:
            req = service.search().list(
                part="id,snippet",
                q=query + " #shorts",
                type="video",
                maxResults=min(opts.per_page, 50),
                order="date",
                regionCode=opts.region,
                videoDuration="short",  # <4 minutes
                publishedAfter=published_after,
                publishedBefore=published_before,
                relevanceLanguage=opts.relevance_language
            )
            if next_page:
                req = req.execute
                data = req(pageToken=next_page)
            else:
                data = req.execute()
        except TypeError:
            # Handle slightly different calling styles in googleapiclient
            req = service.search().list(
                part="id,snippet",
                q=query + " #shorts",
                type="video",
                maxResults=min(opts.per_page, 50),
                order="date",
                regionCode=opts.region,
                videoDuration="short",
                publishedAfter=published_after,
                publishedBefore=published_before,
                relevanceLanguage=opts.relevance_language,
                pageToken=next_page or None
            )
            data = req.execute()
        except HttpError as e:
            print("HTTP error during search:", e)
            break

        for item in data.get("items", []):
            vid = item["id"]["videoId"]
            ids.append(vid)
            fetched += 1
            if fetched >= target:
                break

        next_page = data.get("nextPageToken")
        if fetched >= target or not next_page:
            break
        time.sleep(opts.sleep)
    return ids

def fetch_details(service, video_ids: List[str]) -> List[Dict[str, Any]]:
    rows = []
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
            snippet = v.get("snippet", {})
            details = v.get("contentDetails", {})
            stats = v.get("statistics", {})
            status = v.get("status", {})

            title = snippet.get("title", "")
            desc = snippet.get("description", "") or ""
            duration_iso = details.get("duration", "PT0S")
            seconds = duration_to_seconds(duration_iso)

            # Heuristic flags
            has_shorts_tag = "#shorts" in (title + " " + desc).lower()
            is_short_by_time = seconds <= 60
            is_short = is_short_by_time and (has_shorts_tag or INCLUDE_NO_TAG_GLOBAL)

            if not is_short:
                continue

            rows.append({
                "videoId": v.get("id"),
                "title": title,
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
                "thumbnailDefault": snippet.get("thumbnails", {}).get("default", {}).get("url"),
                "thumbnailMedium": snippet.get("thumbnails", {}).get("medium", {}).get("url"),
                "thumbnailHigh": snippet.get("thumbnails", {}).get("high", {}).get("url"),
            })
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

def main():
    opts = parse_args()
    if not opts.api_key:
        raise SystemExit("Missing API key. Provide --api-key or set YT_API_KEY")

    global INCLUDE_NO_TAG_GLOBAL
    INCLUDE_NO_TAG_GLOBAL = opts.include_no_tag

    service = build("youtube", "v3", developerKey=opts.api_key)

    all_ids = []
    for q in opts.queries:
        ids = search_shorts(service, q, opts)
        print(f"Query '{q}' returned {len(ids)} candidate IDs")
        all_ids.extend(ids)

    # Deduplicate
    all_ids = list(dict.fromkeys(all_ids))
    print(f"After dedupe, {len(all_ids)} IDs")

    rows = fetch_details(service, all_ids)
    write_csv(opts.out, rows)

if __name__ == "__main__":
    main()
