import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # ---- config ----
    CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "/home/kiwi-pandas/Documents/humor-harm-api/annotation_dashboard/02-21a.csv"
    OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "/home/kiwi-pandas/Documents/humor-harm-api/figures"
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- load ----
    df = pd.read_csv(CSV_PATH)

    print("\n=== Shape ===")
    print(df.shape)

    print("\n=== Columns ===")
    print(list(df.columns))

    # ---- non-NA counts ----
    non_na = df.notna().sum().sort_values(ascending=False)
    pct_filled = (non_na / len(df) * 100).round(2)

    summary = pd.DataFrame({
        "non_na_count": non_na,
        "pct_filled": pct_filled
    })

    print("\n=== Non-NA counts per column ===")
    print(summary)

    summary_csv = os.path.join(OUT_DIR, "non_na_counts.csv")
    summary.to_csv(summary_csv, index=True)
    print(f"\nWrote: {summary_csv}")

    # ---- helper for saving figs ----
    def savefig(name: str):
        path = os.path.join(OUT_DIR, name)
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()
        print("Wrote:", path)

    # ---- visualization 1: missing values per column ----
    plt.figure(figsize=(10, 5))
    miss_counts = df.isna().sum().sort_values(ascending=False)
    sns.barplot(x=miss_counts.index, y=miss_counts.values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Missing (NA) count")
    plt.title("Missing values per column")
    savefig("missing_counts_by_column.png")

    # ---- visualization 2: fill rate per column ----
    plt.figure(figsize=(10, 5))
    fill_rates = pct_filled.sort_values(ascending=False)
    sns.barplot(x=fill_rates.index, y=fill_rates.values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Percent filled")
    plt.ylim(0, 100)
    plt.title("Fill rate per column")
    savefig("fill_rate_by_column.png")

    # ---- visualization 3: label distributions (top categories) ----
    cat_cols = [
        "dark_intensity",
        "humor_presence",
        "humor_type",
        "joke_topic",
        "rhetorical_device",
        "stand_up",
        "target_category",
        "annotator",
    ]

    for col in cat_cols:
        if col not in df.columns:
            continue

        s = df[col].dropna()

        # Skip if empty or too many unique values (to avoid unreadable plots)
        if s.empty:
            continue

        nunique = s.nunique(dropna=True)
        if nunique > 50:
            # still write a short summary file instead of a plot
            top = s.value_counts().head(20)
            top.to_csv(os.path.join(OUT_DIR, f"top_values_{col}.csv"))
            print(f"Skipped plot for {col} (too many unique: {nunique}). Wrote top_values_{col}.csv")
            continue

        top = s.value_counts().head(20)

        plt.figure(figsize=(10, 5))
        sns.barplot(x=top.index.astype(str), y=top.values)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Count")
        plt.title(f"Top values for {col}")
        savefig(f"top_values_{col}.png")

    # ---- visualization 4: numeric lead_time ----
    if "lead_time" in df.columns:
        lead = pd.to_numeric(df["lead_time"], errors="coerce").dropna()
        if not lead.empty:
            plt.figure(figsize=(8, 5))
            sns.histplot(lead, kde=True)
            plt.xlabel("lead_time")
            plt.title("Distribution of lead_time")
            savefig("lead_time_hist.png")

    # ---- optional: duplicates by video/id ----
    for key in ["video", "id", "annotation_id"]:
        if key in df.columns:
            dup_count = df.duplicated(subset=[key]).sum()
            print(f"\nDuplicate rows by {key}: {dup_count}")

    print("\nDone.")

if __name__ == "__main__":
    main()