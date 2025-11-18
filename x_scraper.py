import os, argparse, json
from datetime import datetime
import pandas as pd
import tweepy

# ---- Load keys (JSON or KEY=VAL file) ----
def load_keys(path="X_Tokens_Keys"):
    with open(path, "r") as f:
        raw = f.read().strip()
    try:
        return json.loads(raw)  # JSON form
    except json.JSONDecodeError:
        keys = {}
        for line in raw.splitlines():
            if "=" in line:
                k, v = line.strip().split("=", 1)
                keys[k.strip()] = v.strip()
        return keys

TOPIC_KEYWORDS = {
    "Health and Safety": [
        "COVID-19","vaccine","mask mandate","mental health","self-harm",
        "pandemic","health misinformation","lockdown","isolation","body image"
    ],
    "Politics and Society": [
        "US election","voting","political polarization","campaign","censorship",
        "freedom of speech","work from home","remote work burnout",
        "civil rights","government corruption"
    ],
    "Conflict and Global Events": [
        "Russia Ukraine","war","refugees","migration crisis","propaganda",
        "sanctions","natural disaster","climate refugee","peace negotiations",
        "armed conflict"
    ],
    "Environmental and Ethical Issues": [
        "climate change","global warming","sustainability","animal rights",
        "eco-activism","wealth inequality","clean energy","deforestation",
        "carbon footprint","environmental justice"
    ],
}

HUMOR_TERMS = ["comedy", "funny", "humor", "satire", "joke", "jokes"]

def build_queries(humor: bool, lang="en"):
    queries = []
    for topic, kws in TOPIC_KEYWORDS.items():
        kw_or = " OR ".join([f'"{k}"' if " " in k else k for k in kws])
        base = f"({kw_or}) lang:{lang} -is:retweet -is:reply"
        if humor:
            hum = " OR ".join(HUMOR_TERMS)
            q = f"{base} ({hum})"
        else:
            q = base
        queries.append((topic, q))
    return queries

def pick_mp4_variant(variants):
    if not variants:
        return ""
    mp4s = [v for v in variants if isinstance(v, dict) and v.get("content_type") == "video/mp4" and "url" in v]
    if not mp4s:
        return ""
    # Prefer highest bitrate if present
    mp4s.sort(key=lambda v: v.get("bit_rate", -1), reverse=True)
    return mp4s[0]["url"]

def flush_rows(rows, outfile):
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not os.path.exists(outfile) or os.path.getsize(outfile) == 0
    df.to_csv(outfile, mode="a", index=False, header=header)
    rows.clear()

def main():
    ap = argparse.ArgumentParser(prog="x_scraper_video")
    ap.add_argument("--start", required=True, help="ISO 8601, e.g., 2025-11-01T00:00:00Z")
    ap.add_argument("--end", required=True, help="ISO 8601, e.g., 2025-11-04T00:00:00Z")
    ap.add_argument("--limit", type=int, default=800, help="Max tweets per topic to scan")
    ap.add_argument("--outfile", default="x_video_short.csv")
    ap.add_argument("--humor", choices=["yes","no"], default="no")
    ap.add_argument("--max_duration_sec", type=int, default=60, help="Keep videos <= this length")
    args = ap.parse_args()

    keys = load_keys("X_Tokens_Keys")
    client = tweepy.Client(
        bearer_token=keys["X_BEARER_TOKEN"],
        consumer_key=keys.get("X_API_KEY"),
        consumer_secret=keys.get("X_API_SECRET"),
        access_token=keys.get("X_ACCESS_TOKEN"),
        access_token_secret=keys.get("X_ACCESS_SECRET"),
        wait_on_rate_limit=True,
    )

    queries = build_queries(humor=(args.humor == "yes"))
    max_ms = args.max_duration_sec * 1000

    all_rows_buffer = []
    for topic, query in queries:
        print(f"[{topic}] searching…")
        collected = 0
        next_token = None
        while True:
            resp = client.search_recent_tweets(
                query=query,
                max_results=100,
                start_time=args.start,
                end_time=args.end,
                tweet_fields=[
                    "created_at","lang","public_metrics","source","possibly_sensitive",
                    "in_reply_to_user_id","conversation_id","entities","attachments"
                ],
                user_fields=["username","name","created_at","public_metrics","verified","verified_type"],
                media_fields=[
                    "media_key","type","duration_ms","preview_image_url",
                    "public_metrics","width","height","variants"
                ],
                expansions=["author_id","attachments.media_keys"],
                next_token=next_token
            )

            if not resp or not getattr(resp, "data", None):
                if not next_token:
                    print(f"[{topic}] no results in this page.")
                break

            users = {u.id: u for u in (resp.includes.get("users") if resp.includes else [])}
            medias = {m.media_key: m for m in (resp.includes.get("media") if resp.includes else [])}

            for t in resp.data:
                # Filter: must have at least one video with duration <= max_ms
                media_urls = []
                has_short_video = False
                if getattr(t, "attachments", None) and t.attachments.get("media_keys"):
                    for mk in t.attachments["media_keys"]:
                        m = medias.get(mk)
                        if not m:
                            continue
                        if m.type == "video":
                            dur = getattr(m, "duration_ms", None)
                            if dur is not None and dur <= max_ms:
                                has_short_video = True
                                media_urls.append(pick_mp4_variant(getattr(m, "variants", None)) or getattr(m, "preview_image_url", "") or "")
                        # ignore animated_gif (no reliable duration)
                if not has_short_video:
                    continue

                pm = t.public_metrics or {}
                u = users.get(getattr(t, "author_id", None))
                all_rows_buffer.append({
                    "topic": topic,
                    "tweet_id": t.id,
                    "created_at": t.created_at.isoformat() if t.created_at else None,
                    "author_id": getattr(t, "author_id", None),
                    "author_username": getattr(u, "username", None) if u else None,
                    "author_name": getattr(u, "name", None) if u else None,
                    "lang": t.lang,
                    "text": t.text,
                    "likes": pm.get("like_count", 0),
                    "retweets": pm.get("retweet_count", 0),
                    "replies": pm.get("reply_count", 0),
                    "quotes": pm.get("quote_count", 0),
                    "possibly_sensitive": getattr(t, "possibly_sensitive", None),
                    "source": getattr(t, "source", None),
                    "url": f"https://x.com/{getattr(u,'username','user')}/status/{t.id}" if u else f"https://x.com/i/web/status/{t.id}",
                    "video_urls": "|".join([u for u in media_urls if u]),
                    "max_duration_sec": args.max_duration_sec,
                })
                collected += 1

                if collected % 200 == 0:
                    flush_rows(all_rows_buffer, args.outfile)
                    print(f"[{topic}] collected {collected} (flushed)")

                if collected >= args.limit:
                    break

            if collected >= args.limit:
                break
            next_token = (resp.meta or {}).get("next_token")
            if not next_token:
                break

        flush_rows(all_rows_buffer, args.outfile)
        print(f"[{topic}] done. wrote so far -> {args.outfile}")

    # Final de-dupe
    if os.path.exists(args.outfile):
        df = pd.read_csv(args.outfile)
        df = df.drop_duplicates(subset=["tweet_id"])
        df.to_csv(args.outfile, index=False)
        print(f"[final] {len(df)} rows in {args.outfile}")
    else:
        print("[final] No rows written. Try widening dates or lowering filters.")

if __name__ == "__main__":
    main()

