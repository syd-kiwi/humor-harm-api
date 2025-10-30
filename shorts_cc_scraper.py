import os
import csv
import time
import re
import requests
import isodate
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    CouldNotRetrieveTranscript,
)

# ===============================================================
#  CONFIGURATION
# ===============================================================
API_KEY = os.getenv("YT_API_KEY")
OUTPUT_CSV = "youtube_shorts_humor_dataset2.csv"

# ---- Base topic keywords ----
BASE_TOPIC_KEYWORDS = {
    "Health and Safety": [
        "COVID-19", "vaccine", "mask mandate", "mental health", "self-harm",
        "pandemic", "health misinformation", "lockdown", "isolation", "body image"
    ],
    "Politics and Society": [
        "US election", "voting", "political polarization", "campaign", "censorship",
        "freedom of speech", "work from home", "remote work burnout", "civil rights",
        "government corruption"
    ],
    "Conflict and Global Events": [
        "Russia Ukraine", "war", "refugees", "migration crisis", "propaganda",
        "sanctions", "natural disaster", "climate refugee", "peace negotiations",
        "armed conflict"
    ],
    "Environmental and Ethical Issues": [
        "climate change", "global warming", "sustainability", "animal rights",
        "eco-activism", "wealth inequality", "clean energy", "deforestation",
        "carbon footprint", "environmental justice"
    ],
}

# ---- Quota-friendly settings ----
MAX_SEARCH_CALLS_TOTAL = 20          # each search.list costs 100 units
MAX_PER_TOPIC = 25                   # target per humor/non-humor subset
MIN_VIEWS = 100_000
PUBLISHED_AFTER  = "2023-10-01T00:00:00Z"
PUBLISHED_BEFORE = "2025-10-31T23:59:59Z"
SLEEP_BETWEEN_CALLS = 0.35
# ===============================================================

SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
HASHTAG_RE = re.compile(r"#\w+")
search_calls_used = 0


# ===============================================================
#  HELPER FUNCTIONS
# ===============================================================
def expand_humor_queries(base_topics: dict) -> dict:
    """Duplicate each topic into humor/non_humor, adding humor suffixes automatically."""
    humor_suffixes = ["funny", "humor", "comedy", "meme", "satire", "parody"]
    topic_queries = {}
    for topic, keywords in base_topics.items():
        humor_queries = [f"{kw} {tag}" for kw in keywords for tag in humor_suffixes]
        topic_queries[topic] = {
            "humor": humor_queries,
            "non_humor": keywords
        }
    return topic_queries


TOPIC_QUERIES = expand_humor_queries(BASE_TOPIC_KEYWORDS)


def log_err(resp):
    try:
        body = resp.json()
        reason = body.get("error", {}).get("errors", [{}])[0].get("reason", "")
        msg = body.get("error", {}).get("message", "")
        print(f"[YouTube ERROR {resp.status_code}] reason={reason} msg={msg}")
    except Exception:
        print(f"[YouTube ERROR {resp.status_code}] {resp.text[:300]}")


def yt_get(url, params, cost=0):
    """GET with quota counter."""
    global search_calls_used
    if cost > 0:
        if search_calls_used >= MAX_SEARCH_CALLS_TOTAL:
            return None
        search_calls_used += 1
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        log_err(r)
        return None
    return r.json()


def iso8601_to_seconds(dur_str: str) -> int:
    try:
        return int(isodate.parse_duration(dur_str).total_seconds())
    except Exception:
        return 0


def extract_hashtags(title: str, desc: str) -> str:
    tags = set()
    for t in (title, desc):
        if isinstance(t, str):
            tags.update(HASHTAG_RE.findall(t))
    return " ".join(sorted(tags)) if tags else ""


def get_transcript_text(video_id: str) -> tuple[str, bool]:
    """Return (transcript_text, is_generated). Prefers auto-generated EN."""
    try:
        tlist = YouTubeTranscriptApi.list_transcripts(video_id)
        # Auto EN
        for tr in tlist:
            if tr.language_code.startswith("en") and tr.is_generated:
                lines = tr.fetch()
                return " ".join(ch.get("text", "") for ch in lines if ch.get("text")), True
        # Manual EN
        for tr in tlist:
            if tr.language_code.startswith("en") and not tr.is_generated:
                lines = tr.fetch()
                return " ".join(ch.get("text", "") for ch in lines if ch.get("text")), False
        # Translate to EN
        for tr in tlist:
            try:
                tr_en = tr.translate("en")
                lines = tr_en.fetch()
                return " ".join(ch.get("text", "") for ch in lines if ch.get("text")), tr.is_generated
            except Exception:
                continue
    except (TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript):
        pass
    except Exception:
        pass
    return "", False


# ===============================================================
#  CORE YOUTUBE SCRAPER (LOW QUOTA)
# ===============================================================
def one_search_page(keyword: str):
    """Fetch exactly one search page (50 IDs). Costs 100 quota units."""
    params = {
        "key": API_KEY,
        "q": keyword,
        "type": "video",
        "part": "id",
        "maxResults": 50,
        "videoDuration": "short",
        "videoCaption": "closedCaption",  # must have captions
        "relevanceLanguage": "en",
        "publishedAfter": PUBLISHED_AFTER,
        "publishedBefore": PUBLISHED_BEFORE,
        "order": "viewCount",
        "safeSearch": "none",
    }
    data = yt_get(SEARCH_URL, params, cost=100)
    if not data:
        return []
    return [it["id"]["videoId"] for it in data.get("items", []) if "videoId" in it.get("id", {})]


def fetch_video_stats(ids: list[str]):
    """Batch fetch video metadata."""
    all_stats = {}
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        params = {
            "key": API_KEY,
            "id": ",".join(chunk),
            "part": "snippet,contentDetails,statistics,status",
        }
        r = requests.get(VIDEOS_URL, params=params, timeout=30)
        if r.status_code != 200:
            log_err(r)
            continue
        data = r.json()
        for v in data.get("items", []):
            vid = v.get("id")
            snip = v.get("snippet", {})
            details = v.get("contentDetails", {})
            stats = v.get("statistics", {})
            all_stats[vid] = (snip, details, stats)
        time.sleep(0.15)
    return all_stats


def collect_for_topic(topic: str, keywords: list[str]):
    """Fetch 1 page per keyword until we hit quota or target count."""
    kept = []
    seen_ids = set()
    for kw in keywords:
        if len(kept) >= MAX_PER_TOPIC or search_calls_used >= MAX_SEARCH_CALLS_TOTAL:
            break
        ids = one_search_page(kw)
        print(f"[{topic}] '{kw}' -> {len(ids)} ids (calls used: {search_calls_used}/{MAX_SEARCH_CALLS_TOTAL})")
        ids = [vid for vid in ids if vid not in seen_ids]
        seen_ids.update(ids)
        # Filter by transcript first (free)
        pass_ids, transcripts_map, gen_flag_map = [], {}, {}
        for vid in ids:
            text, is_gen = get_transcript_text(vid)
            if text.strip():
                transcripts_map[vid] = text
                gen_flag_map[vid] = is_gen
                pass_ids.append(vid)
        if not pass_ids:
            continue
        stats_map = fetch_video_stats(pass_ids)
        for vid in pass_ids:
            if vid not in stats_map:
                continue
            snip, details, stats = stats_map[vid]
            dur_s = iso8601_to_seconds(details.get("duration", "PT0S"))
            if dur_s == 0 or dur_s > 60:
                continue
            views = int(stats.get("viewCount", 0) or 0)
            if views < MIN_VIEWS:
                continue
            title = snip.get("title", "")
            desc = snip.get("description", "")
            row = {
                "videoId": vid,
                "title": title,
                "description": desc,
                "channelId": snip.get("channelId", ""),
                "datePublished": snip.get("publishedAt", ""),
                "durationSeconds": dur_s,
                "viewCount": views,
                "likeCount": int(stats.get("likeCount", 0) or 0) if "likeCount" in stats else "",
                "commentCount": int(stats.get("commentCount", 0) or 0) if "commentCount" in stats else "",
                "hashtags": extract_hashtags(title, desc),
                "transcript": transcripts_map[vid],
                "transcriptGenerated": gen_flag_map[vid],
                "topic": topic,
            }
            kept.append(row)
            if len(kept) >= MAX_PER_TOPIC:
                break
        print(f"[{topic}] kept so far: {len(kept)}/{MAX_PER_TOPIC}")
        time.sleep(SLEEP_BETWEEN_CALLS)
    return kept


# ===============================================================
#  MAIN
# ===============================================================
def main():
    if not API_KEY:
        raise SystemExit("❌ Set YOUTUBE_API_KEY first.")
    all_rows = []
    for topic, groups in TOPIC_QUERIES.items():
        print(f"\n=== Topic: {topic} ===")
        for humor_type, keywords in groups.items():
            if search_calls_used >= MAX_SEARCH_CALLS_TOTAL:
                print("Reached quota limit; stopping.")
                break
            label = "humor" if humor_type == "humor" else "non_humor"
            print(f"\n-- {topic} | {label.upper()} --")
            rows = collect_for_topic(topic, keywords)
            for r in rows:
                r["humorType"] = label
            all_rows.extend(rows)
            print(f"[{topic} | {label}] kept {len(rows)}")
    fieldnames = [
        "videoId","title","description","channelId","datePublished",
        "durationSeconds","viewCount","likeCount","commentCount",
        "hashtags","transcript","transcriptGenerated","topic","humorType"
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"\n📁 Done. Wrote {len(all_rows)} rows to {OUTPUT_CSV}")
    print(f"Search calls used: {search_calls_used}/{MAX_SEARCH_CALLS_TOTAL} (~{search_calls_used*100} quota units)")

if __name__ == "__main__":
    main()
