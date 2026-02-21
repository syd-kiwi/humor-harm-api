
#!/usr/bin/env python3
"""
Extracts video IDs from a saved YouTube HTML file and keeps only short (<2 min)
and popular (>100K views) videos.

Reads a saved YouTube page (HTML source in a .txt file),

Usage:
  Edit URL and OUTPUT_CSV below before running.
"""

import re
import os
import csv
import requests
from urllib.parse import urlparse, parse_qs
import time

#URL = "https://www.youtube.com/results?search_query=COVID-19+funny&sp=CAASCAgFEAEYASgB"
#URL = "https://www.youtube.com/results?search_query=humor+covid-19&sp=EgYIBRABGAE%253D"
#URL = "https://www.youtube.com/results?search_query=humor+vaccine&sp=EgYIBRABGAE%253D"
#URL = "https://www.youtube.com/results?search_query=humor+mask+mandate&sp=EgYIBRABGAE%253D"
#URL = "https://www.youtube.com/results?search_query=funny+mental+health&sp=EgYIBRABGAE%253D"
#URL = "https://www.youtube.com/results?search_query=funny+self+harm&sp=EgYIBRABGAE%253D"
#URL = "https://www.youtube.com/results?search_query=funny+pandemic&sp=EgYIBRABGAE%253D"
#URL = "https://www.youtube.com/results?search_query=funny+health+misinformation&sp=EgYIBRABGAE%253D"
#URL = "https://www.youtube.com/results?search_query=funny+lockdown&sp=EgYIBRABGAE%253D"
#URL = "https://www.youtube.com/results?search_query=funny+US+election&sp=EgYIBRABGAE%253D"
#URL = "https://www.youtube.com/results?search_query=funny+US+voting&sp=EgYIBRABGAE%253D"
#URL = "https://www.youtube.com/results?search_query=funny+us+campaign&sp=EgYIBRABGAE%253D"
#URL = "https://www.youtube.com/results?search_query=funny+us+campaign&sp=EgYIBRABGAE%253D"

headers = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
}

with open("yt_links.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader, None)  # skip header if present

    for row in reader:
        if not row:
            continue

        URL = row[0].strip()
        print(f"Processing: {URL}")

        parsed = urlparse(URL)
        q = parse_qs(parsed.query)
        params = {
            "search_query": q.get("search_query", [""])[0],
            "sp": q.get("sp", [""])[0],
            "hl": "en",
            "gl": "US"
        }

        headers = {
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                           "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
            "Accept-Language": "en-US,en;q=0.9",
        }
        cookies = {
            # avoid consent interstitials
            "CONSENT": "YES+",
            # also reinforce language/region
            "PREF": "hl=en&gl=US",
        }

        # compile patterns once per URL
        block_pat = re.compile(
            r':\s*\[\s*\{\s*"text"\s*:\s*"(.*?)"\s*\}\s*\]\s*,\s*"accessibility"\s*:\s*\{\s*"accessibilityData"\s*:\s*\{\s*"label"\s*:\s*"(.*?)"\s*\}\s*\}\s*\}\s*,\s*"longBylineText"',
            re.IGNORECASE | re.DOTALL
        )

        views_pat = re.compile(
            r'(?P<views>\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?\s*[KM]?)\s+views',
            re.IGNORECASE
        )

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

        longbyline_pat = re.compile(r',"longBylineText":', re.IGNORECASE)
        vid_pat = re.compile(r'"videoId"\s*:\s*"([A-Za-z0-9_-]{11})"', re.IGNORECASE)

        MAX_RETRIES = 3
        attempt = 0

        while True:
            # fetch the page
            r = requests.get(URL, params=params, headers=headers, cookies=cookies, timeout=20)
            html = r.text

            # save latest html to file
            with open("page_source.txt", "w", encoding="utf-8") as pf:
                pf.write(html)

            print("[✔] Saved HTML source to page_source.txt")

            rows = []

            # parse blocks
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
                    except Exception:
                        views = None

                # duration (seconds)
                dur = None
                mdur = secs_pat_ms.search(label) or secs_pat_s.search(label) or secs_pat_m.search(label)
                if mdur:
                    mins = int(mdur.groupdict().get("mins") or 0)
                    secs = int(mdur.groupdict().get("secs") or 0)
                    dur = mins * 60 + secs

                rows.append([vid, title, views, dur])

            if len(rows) >= 15:
                # enough rows, continue
                break

            attempt += 1
            if attempt >= MAX_RETRIES:
                print(f"[!] Only {len(rows)} rows found after {MAX_RETRIES} attempts. Using what we have.")
                break

            print(f"[!] Only {len(rows)} rows found. Retrying in 5 seconds... (Attempt {attempt}/{MAX_RETRIES})")
            time.sleep(5)

        # --- Print all ---
        print(f"{'VIDEO ID':<15} | {'DURATION(s)':<10} | {'VIEWS':<10} | TITLE")
        print("-" * 80)
        for vid, title, views, dur in rows:
            print(f"{(vid or 'N/A'):<15} | {(dur or 'N/A')!s:<10} | {(views or 'N/A')!s:<10} | {title or '(no title)'}")

        # --- Save CSV ---
        OUTPUT_CSV = f"keyword_searches/Keyword_{params['search_query'].replace(' ', '')}.csv"
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

        file_exists = os.path.isfile(OUTPUT_CSV)

        with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f_out:
            w = csv.writer(f_out)

            # Write header only if file is new
            if not file_exists:
                w.writerow(["video_id", "title", "view_count", "duration_seconds"])

            # Append rows
            w.writerows(rows)

        print(f"\nAppended {len(rows)} rows -> {OUTPUT_CSV}")
        time.sleep(5)