import os
import pandas as pd

csv_dir = "/home/kiwi-pandas/Documents/humor-harm-api/keyword_searches"
output_file = "csvs_with_less_than_20_rows.txt"

bad_csvs = []

for csv_file in os.listdir(csv_dir):
    if csv_file.endswith(".csv"):
        csv_path = os.path.join(csv_dir, csv_file)
        try:
            df = pd.read_csv(csv_path)
            if len(df) < 20:
                bad_csvs.append(csv_file)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

with open(output_file, "w") as f:
    for name in bad_csvs:
        f.write(name + "\n")

print(f"[✔] Saved {len(bad_csvs)} CSV filenames with <20 rows to {output_file}")
