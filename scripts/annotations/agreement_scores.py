import argparse
import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path

import pandas as pd


# -----------------------------
# Helpers
# -----------------------------

def pairwise_disagreement_rate(values):
    if len(values) < 2:
        return None
    pairs = list(combinations(values, 2))
    if not pairs:
        return None
    disagree = sum(1 for a, b in pairs if a != b)
    return disagree / len(pairs)


def irony_or_satire_present(cell):
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return None

    s = str(cell).strip()
    if not s:
        return None

    lowered = s.lower()

    # JSON case
    if lowered.startswith("{") or lowered.startswith("["):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                choices = obj.get("choices", [])
            elif isinstance(obj, list):
                choices = obj
            else:
                choices = []
            choices_l = [str(c).strip().lower() for c in choices if c]
            return "Yes" if ("irony" in choices_l or "satire" in choices_l) else "No"
        except Exception:
            pass

    return "Yes" if ("irony" in lowered or "satire" in lowered) else "No"


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--item_col", default="id")
    ap.add_argument("--annotator_col", default="annotator")
    ap.add_argument(
        "--labels",
        nargs="+",
        default=[
            "humor_presence",
            "joke_topic",
            "humor_type",
            "dark_intensity",
            "target_category",
            "stand_up",
            "irony_or_satire_present",
        ],
    )
    ap.add_argument("--top_k", type=int, default=20)
    args = ap.parse_args()

    df = pd.read_csv(args.csv_path)

    # Derived binary label
    if "rhetorical_device" in df.columns and "irony_or_satire_present" not in df.columns:
        df["irony_or_satire_present"] = df["rhetorical_device"].apply(irony_or_satire_present)

    needed = [args.item_col, args.annotator_col] + list(args.labels)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}")

    per_id_scores = []

    # Compute disagreement per id per field
    for label in args.labels:
        sub = df[[args.item_col, args.annotator_col, label]].dropna(subset=[label])

        for item_id, g in sub.groupby(args.item_col):
            g2 = g.drop_duplicates(subset=[args.annotator_col])
            vals = [str(v) for v in g2[label].tolist()]
            if len(vals) < 2:
                continue

            rate = pairwise_disagreement_rate(vals)
            if rate is not None:
                per_id_scores.append({
                    "id": item_id,
                    "field": label,
                    "disagreement_rate": rate,
                })

    per_id_df = pd.DataFrame(per_id_scores)

    # Aggregate across fields
    agg = (
        per_id_df
        .groupby("id")
        .agg(
            mean_disagreement=("disagreement_rate", "mean"),
            n_fields_used=("field", "count"),
        )
        .reset_index()
    )

    worst = (
        agg.sort_values(
            by=["mean_disagreement", "n_fields_used"],
            ascending=[False, False],
        )
        .head(args.top_k)
    )

    print("\nTop 20 worst ids overall (highest mean disagreement):\n")
    for _, row in worst.iterrows():
        print(
            f"id={row['id']}  "
            f"mean_disagreement={row['mean_disagreement']:.3f}  "
            f"fields_used={int(row['n_fields_used'])}"
        )

    out_path = Path(args.csv_path).with_name(
        f"{Path(args.csv_path).stem}_worst_ids_overall_top{args.top_k}.csv"
    )
    worst.to_csv(out_path, index=False)

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()