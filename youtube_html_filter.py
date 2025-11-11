#!/usr/bin/env python3
"""
Extracts video IDs from a saved YouTube HTML file and keeps only short (<2 min)
and popular (>100K views) videos.

Reads a saved YouTube page (HTML source in a .txt file),

Usage:
  Edit URL and OUTPUT_CSV below before running.
"""

import re
import csv
import requests

URL = "https://www.youtube.com/results?search_query=COVID-19+funny&sp=CAASCAgFEAEYASgB"
r = requests.get(URL)
with open("page_source.txt", "w", encoding="utf-8") as f:
    f.write(r.text)
print("[✔] Saved HTML source to page_source.txt")


# --- Edit paths ---
INPUT_FILE = "page_source.txt"    # your saved YouTube HTML
OUTPUT_CSV = "filtered_videos.csv"  # output file

block_pat = re.compile(
    r':\s*\[\s*\{\s*"text"\s*:\s*"(.*?)"\s*\}\s*\]\s*,\s*"accessibility"\s*:\s*\{\s*"accessibilityData"\s*:\s*\{\s*"label"\s*:\s*"(.*?)"\s*\}\s*\}\s*\}\s*,\s*"longBylineText"',
    re.IGNORECASE | re.DOTALL
)

# views = number next to "views"
views_pat = re.compile(r'(?P<views>\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?\s*[KM]?)\s+views', re.IGNORECASE)
# duration = number before "seconds"

secs_pat_ms = re.compile(
    r'(?P<mins>\d+)\s*minutes?\s*(?:,|\s+and\s+)\s*(?P<secs>\d+)\s*seconds?\b',
    re.IGNORECASE
)
secs_pat_s = re.compile(
    r'(?P<secs>\d+)\s*seconds?\b',
    re.IGNORECASE
)
secs_pat_m = re.compile(
    r'(?P<mins>\d+)\s*minutes?(?!\s*ago)\b',
    re.IGNORECASE
)
# videoId (search just after the block)
longbyline_pat = re.compile(r',"longBylineText":', re.IGNORECASE)
vid_pat = re.compile(r'"videoId"\s*:\s*"([A-Za-z0-9_-]{11})"', re.IGNORECASE)

with open(INPUT_FILE, "r", encoding="utf-8", errors="replace") as f:
    html = f.read()

rows = []

for m in block_pat.finditer(html):
    label = m.group(2)

    # title = start of label up to " by "
    title = label.split(" by ", 1)[0].replace(r"\u0026", "&")

    # videoId: look right after the matched block (forward window)
    vid = None
    byline_match = longbyline_pat.search(html, m.start(), m.end() + 2000)
    if byline_match:
        start = byline_match.end()
        end = min(len(html), start + 1500)
        vid_match = vid_pat.search(html, start, end)
        if vid_match:
            vid = vid_match.group(1)

    # views
    views = None
    vm = views_pat.search(label)
    if vm:
        raw = vm.group("views").strip().replace(",", "")
        up = raw.upper()
        try:
            if up.endswith("M"):
                views = int(float(up[:-1]) * 1_000_000)
            elif up.endswith("K"):
                views = int(float(up[:-1]) * 1_000)
            else:
                views = int(raw)
        except:
            views = None

    # duration (seconds)
    dur = None
    m = secs_pat_ms.search(label) or secs_pat_s.search(label) or secs_pat_m.search(label)
    if m:
        mins = int(m.groupdict().get("mins") or 0)
        secs = int(m.groupdict().get("secs") or 0)
        dur = mins * 60 + secs

    rows.append([vid, title, views, dur])

# --- Print all ---
print(f"{'VIDEO ID':<15} | {'DURATION(s)':<10} | {'VIEWS':<10} | TITLE")
print("-" * 80)
for vid, title, views, dur in rows:
    print(f"{(vid or 'N/A'):<15} | {(dur or 'N/A')!s:<10} | {(views or 'N/A')!s:<10} | {title or '(no title)'}")

# --- Save CSV ---
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["video_id", "title", "view_count", "duration_seconds"])
    w.writerows(rows)

print(f"\nSaved {len(rows)} rows -> {OUTPUT_CSV}")