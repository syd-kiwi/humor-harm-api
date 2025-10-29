#!/usr/bin/env python3
import os, csv, argparse, math, time
from datetime import datetime, timedelta, timezone
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

COLUMNS = ["videoId","videoUrl","title","description","channelId","channelTitle","publishedAt",
           "durationSeconds","viewCount","likeCount","commentCount","license","definition",
           "madeForKids","tagsCount","hasShortsTag","isShortByTime","transcriptText"]

def parse_args():
    p = argparse.ArgumentParser(description="Simple Shorts collector")
    p.add_argument("--api-key", default=os.getenv("YT_API_KEY",""))
    p.add_argument("--queries", nargs="+", required=True)
    p.add_argument("--limit-rows", type=int, default=10)
    p.add_argument("--cc-only", action="store_true")
    p.add_argument("--out", default="shorts_simple.csv")
    p.add_argument("--region", default="US")
    return p.parse_args()

def last_year_iso():
    now = datetime.now(timezone.utc)
    return (now - timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%SZ"), now.strftime("%Y-%m-%dT%H:%M:%SZ")

def dur_to_sec(iso):
    total=n=""; s=0
    for c in iso:
        if c.isdigit(): n+=c
        elif c=="H" and n: s+=int(n)*3600; n=""
        elif c=="M" and n: s+=int(n)*60;   n=""
        elif c=="S" and n: s+=int(n);      n=""
    return s

def try_transcript(vid):
    try:
        t = YouTubeTranscriptApi.get_transcript(vid)
        return " ".join(seg.get("text","") for seg in t).strip()
    except (TranscriptsDisabled, NoTranscriptFound):
        try:
            for tr in YouTubeTranscriptApi.list_transcripts(vid):
                try:
                    text = " ".join(seg.get("text","") for seg in tr.fetch()).strip()
                    if text: return text
                except: pass
        except: pass
    except: pass
    return ""

def search_ids(svc, q, after, before, region, target):
    ids=[]; tok=None
    while True:
        try:
            params=dict(part="id,snippet", q=q+" #shorts", type="video", maxResults=50,
                        order="date", regionCode=region, videoDuration="short",
                        publishedAfter=after, publishedBefore=before)
            if tok: params["pageToken"]=tok
            data=svc.search().list(**params).execute()
        except HttpError as e:
            print("Search error:", e); break
        for it in data.get("items",[]):
            ids.append(it["id"]["videoId"])
            if len(ids)>=target: return ids
        tok=data.get("nextPageToken")
        if not tok: return ids
        time.sleep(0.2)

def write_csv(base, rows):
    if not rows: print("No rows to write"); return
    ts=datetime.now().strftime("%Y%m%d_%H%M%S"); path=f"{os.path.splitext(base)[0]}_{ts}.csv"
    with open(path,"w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f, fieldnames=COLUMNS); w.writeheader(); [w.writerow(r) for r in rows]
    print(f"Wrote {len(rows)} rows to {path}")

def main():
    a=parse_args()
    if not a.api_key: raise SystemExit("Missing API key. Set YT_API_KEY or pass --api-key")
    after,before = last_year_iso()
    svc = build("youtube","v3",developerKey=a.api_key)

    # collect candidate ids
    all_ids=[]
    per=max(1, math.ceil(a.limit_rows*5 / max(1,len(a.queries))))  # grab a bit extra
    for q in a.queries:
        all_ids += search_ids(svc,q,after,before,a.region,per)

    # details and filtering
    rows=[]
    for i in range(0,len(all_ids),50):
        try:
            resp=svc.videos().list(part="snippet,contentDetails,statistics,status",
                                   id=",".join(all_ids[i:i+50])).execute()
        except HttpError as e:
            print("Details error:", e); continue
        for v in resp.get("items",[]):
            sn=v.get("snippet",{}); cd=v.get("contentDetails",{}); st=v.get("statistics",{}); ss=v.get("status",{})
            secs=dur_to_sec(cd.get("duration","PT0S"))
            if secs>65: continue
            if a.cc_only and str(ss.get("license","")).lower()!="creativecommon": continue
            title=sn.get("title",""); desc=sn.get("description","")
            row=dict(
                videoId=v["id"],
                videoUrl=f"https://www.youtube.com/watch?v={v['id']}",
                title=title, description=desc,
                channelId=sn.get("channelId"), channelTitle=sn.get("channelTitle"),
                publishedAt=sn.get("publishedAt"),
                durationSeconds=secs,
                viewCount=st.get("viewCount"), likeCount=st.get("likeCount"), commentCount=st.get("commentCount"),
                license=ss.get("license"), definition=cd.get("definition"),
                madeForKids=ss.get("madeForKids"),
                tagsCount=len(sn.get("tags",[])) if sn.get("tags") else 0,
                hasShortsTag="#shorts" in (title+" "+desc).lower(),
                isShortByTime=True,
                transcriptText=try_transcript(v["id"])
            )
            rows.append(row)
            if len(rows)>=a.limit_rows: write_csv(a.out, rows); return
        time.sleep(0.2)
    write_csv(a.out, rows)

if __name__=="__main__": main()
