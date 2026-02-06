import os, time, csv, itertools, requests, re
from datetime import datetime

ACCESS_TOKEN = os.environ.get("IG_TOKEN") or "YOUR_LONG_LIVED_USER_TOKEN"
IG_USER_ID   = os.environ.get("IG_USER_ID") or "YOUR_IG_USER_ID"
OUT_CSV      = "instagram_reels_search.csv"

# Topics and keywords (from your plan)
TOPICS = {
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
        "sanctions","natural disaster","climate refugee","peace negotiations","armed conflict"
    ],
    "Environmental and Ethical Issues": [
        "climate change","global warming","sustainability","animal rights","eco-activism",
        "wealth inequality","clean energy","deforestation","carbon footprint","environmental justice"
    ]
}

# Humor boosters for "humor" mode
HUMOR_TAGS = ["funny", "comedy", "humor"]

# Build hashtag candidates from a keyword
def hashtag_variants(kw):
    base = kw.lower()
    base = re.sub(r"[^a-z0-9 ]+", "", base)
    words = base.split()
    if not words: 
        return []
    cand = set()
    joiners = ["", "_"]  # try both joined and underscored
    for j in joiners:
        cand.add("#" + j.join(words))
    # common US election alias
    if base == "us election":
        cand.update({"#uselection", "#election2024", "#election2025"})
    # russia ukraine alias
    if base == "russia ukraine":
        cand.update({"#russiaukraine", "#ukrainewar"})
    # climate change alias
    if base == "climate change":
        cand.update({"#climatechange"})
    return list(cand)

GRAPH = "https://graph.facebook.com/v20.0"

def ig_get(path, **params):
    params["access_token"] = ACCESS_TOKEN
    r = requests.get(f"{GRAPH}/{path}", params=params, timeout=30)
    if r.status_code == 429:
        # simple backoff
        time.sleep(5)
        r = requests.get(f"{GRAPH}/{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def get_hashtag_id(tag):
    # tag should be like "#funny" or "funny"; API needs bare word
    word = tag.lstrip("#")
    data = ig_get("ig_hashtag_search", user_id=IG_USER_ID, q=word)
    arr = data.get("data", [])
    return arr[0]["id"] if arr else None

def fetch_recent_media(hashtag_id, limit=50):
    fields = "id,caption,media_type,media_product_type,permalink,thumbnail_url,timestamp"
    data = ig_get(f"{hashtag_id}/recent_media", user_id=IG_USER_ID, fields=fields, limit=limit)
    items = data.get("data", [])
    return items

def row_from_item(item, topic, keyword, humor_flag):
    return {
        "ig_media_id": item.get("id",""),
        "caption": (item.get("caption") or "").replace("\n"," ").strip(),
        "permalink": item.get("permalink",""),
        "timestamp": item.get("timestamp",""),
        "topic": topic,
        "keyword": keyword,
        "humor_flag": humor_flag,
        "media_type": item.get("media_type",""),
        "media_product_type": item.get("media_product_type",""),
        "is_reel": str(item.get("media_product_type","") == "REELS")
    }

def ensure_csv(path):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "ig_media_id","caption","permalink","timestamp",
                "topic","keyword","humor_flag","media_type","media_product_type","is_reel"
            ])
            w.writeheader()

def save_rows(path, rows):
    if not rows:
        return 0
    ensure_csv(path)
    seen = set()
    # dedupe by ig_media_id already on disk
    with open(path, "r", encoding="utf-8") as f:
        for line in f.read().splitlines()[1:]:
            if line:
                seen.add(line.split(",", 1)[0])
    new_rows = [r for r in rows if r["ig_media_id"] not in seen]
    if not new_rows:
        return 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=new_rows[0].keys())
        for r in new_rows:
            w.writerow(r)
    return len(new_rows)

def collect(max_calls=2000, per_hashtag_limit=25, sleep_sec=0.4):
    calls = 0
    total = 0
    for topic, keywords in TOPICS.items():
        for kw in keywords:
            # two modes: non humor and humor
            modes = [("non_humor", []), ("humor", HUMOR_TAGS)]
            for humor_flag, boosters in modes:
                tags = hashtag_variants(kw)
                if boosters:
                    tags += ["#" + b for b in boosters]
                # keep top 3 tags to conserve calls
                tags = tags[:3]
                for tag in tags:
                    if calls + 2 > max_calls:
                        print("Reached call budget.")
                        return total
                    hid = get_hashtag_id(tag); calls += 1
                    if not hid:
                        continue
                    items = fetch_recent_media(hid, limit=per_hashtag_limit); calls += 1
                    # filter reels
                    reels = [it for it in items if it.get("media_product_type") == "REELS"]
                    rows = [row_from_item(it, topic, kw, humor_flag) for it in reels]
                    total += save_rows(OUT_CSV, rows)
                    time.sleep(sleep_sec)
    return total

if __name__ == "__main__":
    added = collect(max_calls=2000, per_hashtag_limit=25, sleep_sec=0.4)
    print(f"Added {added} rows to {OUT_CSV}")
