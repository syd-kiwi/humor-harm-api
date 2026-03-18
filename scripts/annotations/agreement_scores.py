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


def normalize_cell(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def cohen_kappa_score(labels_a, labels_b):
    if len(labels_a) != len(labels_b):
        raise ValueError("labels_a and labels_b must have the same length")
    if len(labels_a) == 0:
        return None

    agreement = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    observed = agreement / len(labels_a)

    a_counts = Counter(labels_a)
    b_counts = Counter(labels_b)
    all_labels = set(a_counts) | set(b_counts)
    expected = sum(
        (a_counts[label] / len(labels_a)) * (b_counts[label] / len(labels_b))
        for label in all_labels
    )

    if math.isclose(1.0 - expected, 0.0, abs_tol=1e-12):
        return 1.0 if math.isclose(observed, 1.0, abs_tol=1e-12) else None

    return (observed - expected) / (1.0 - expected)


def choose_metadata_columns(df, requested_cols):
    if requested_cols:
        return [col for col in requested_cols if col in df.columns]

    preferred = ["video_id", "url", "title"]
    return [col for col in preferred if col in df.columns]


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
    ap.add_argument(
        "--video_cols",
        nargs="*",
        default=None,
        help="Optional metadata columns to include when listing which videos/items were used. Defaults to video_id/url/title when present.",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv_path)
    df = pd.read_csv(csv_path)

    # Derived binary label
    if "rhetorical_device" in df.columns and "irony_or_satire_present" not in df.columns:
        df["irony_or_satire_present"] = df["rhetorical_device"].apply(irony_or_satire_present)

    metadata_cols = choose_metadata_columns(df, args.video_cols)
    needed = [args.item_col, args.annotator_col] + list(args.labels)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}")

    per_id_scores = []
    kappa_rows = []
    videos_used_rows = []

    # Compute disagreement per id per field
    for label in args.labels:
        label_cols = [args.item_col, args.annotator_col, label] + metadata_cols
        sub = df[label_cols].copy()
        sub[label] = sub[label].apply(normalize_cell)
        sub = sub.dropna(subset=[label])

        for item_id, g in sub.groupby(args.item_col):
            g2 = g.drop_duplicates(subset=[args.annotator_col])
            vals = g2[label].tolist()
            if len(vals) < 2:
                continue

            rate = pairwise_disagreement_rate(vals)
            if rate is not None:
                per_id_scores.append(
                    {
                        "id": item_id,
                        "field": label,
                        "disagreement_rate": rate,
                    }
                )

        # Pairwise Cohen's kappa across annotators for this label
        shared = sub[[args.item_col, args.annotator_col, label] + metadata_cols].drop_duplicates(
            subset=[args.item_col, args.annotator_col]
        )

        annotators = sorted(shared[args.annotator_col].astype(str).unique())
        for annotator_a, annotator_b in combinations(annotators, 2):
            a_rows = (
                shared[shared[args.annotator_col].astype(str) == annotator_a]
                .drop_duplicates(subset=[args.item_col])
                .set_index(args.item_col)
            )
            b_rows = (
                shared[shared[args.annotator_col].astype(str) == annotator_b]
                .drop_duplicates(subset=[args.item_col])
                .set_index(args.item_col)
            )
            shared_ids = sorted(set(a_rows.index) & set(b_rows.index))
            if not shared_ids:
                continue

            labels_a = [a_rows.at[item_id, label] for item_id in shared_ids]
            labels_b = [b_rows.at[item_id, label] for item_id in shared_ids]
            kappa = cohen_kappa_score(labels_a, labels_b)
            observed_agreement = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / len(shared_ids)

            kappa_rows.append(
                {
                    "field": label,
                    "annotator_a": annotator_a,
                    "annotator_b": annotator_b,
                    "n_items_used": len(shared_ids),
                    "observed_agreement": observed_agreement,
                    "cohen_kappa": kappa,
                }
            )

            for item_id in shared_ids:
                row = {
                    "field": label,
                    "annotator_a": annotator_a,
                    "annotator_b": annotator_b,
                    "id": item_id,
                }
                for col in metadata_cols:
                    row[col] = a_rows.at[item_id, col] if col in a_rows.columns else None
                videos_used_rows.append(row)

    if not per_id_scores:
        raise SystemExit("No items with 2 or more annotations found for the requested labels.")

    per_id_df = pd.DataFrame(per_id_scores)

    # Aggregate across fields
    agg = (
        per_id_df.groupby("id")
        .agg(
            mean_disagreement=("disagreement_rate", "mean"),
            n_fields_used=("field", "count"),
        )
        .reset_index()
    )

    worst = agg.sort_values(
        by=["mean_disagreement", "n_fields_used"],
        ascending=[False, False],
    ).head(args.top_k)

    print(f"\nTop {args.top_k} worst ids overall (highest mean disagreement):\n")
    for _, row in worst.iterrows():
        print(
            f"id={row['id']}  "
            f"mean_disagreement={row['mean_disagreement']:.3f}  "
            f"fields_used={int(row['n_fields_used'])}"
        )

    worst_out_path = csv_path.with_name(
        f"{csv_path.stem}_worst_ids_overall_top{args.top_k}.csv"
    )
    worst.to_csv(worst_out_path, index=False)

    kappa_df = pd.DataFrame(kappa_rows)
    videos_used_df = pd.DataFrame(videos_used_rows)

    if not kappa_df.empty:
        kappa_out_path = csv_path.with_name(f"{csv_path.stem}_cohen_kappa_scores.csv")
        kappa_df.to_csv(kappa_out_path, index=False)

        kappa_summary = (
            kappa_df.groupby("field")
            .agg(
                annotator_pairs=("cohen_kappa", "size"),
                mean_items_used=("n_items_used", "mean"),
                mean_observed_agreement=("observed_agreement", "mean"),
                mean_cohen_kappa=("cohen_kappa", "mean"),
                min_cohen_kappa=("cohen_kappa", "min"),
                max_cohen_kappa=("cohen_kappa", "max"),
            )
            .reset_index()
        )
        kappa_summary_out = csv_path.with_name(
            f"{csv_path.stem}_cohen_kappa_summary.csv"
        )
        kappa_summary.to_csv(kappa_summary_out, index=False)

        videos_out_path = csv_path.with_name(
            f"{csv_path.stem}_videos_used_for_kappa.csv"
        )
        videos_used_df.to_csv(videos_out_path, index=False)

        print(f"\nWrote {kappa_out_path}")
        print(f"Wrote {kappa_summary_out}")
        print(f"Wrote {videos_out_path}")

        print("\nCohen's kappa summary by field:\n")
        for _, row in kappa_summary.iterrows():
            mean_kappa = row["mean_cohen_kappa"]
            mean_items = row["mean_items_used"]
            print(
                f"field={row['field']}  "
                f"annotator_pairs={int(row['annotator_pairs'])}  "
                f"mean_items_used={mean_items:.1f}  "
                f"mean_cohen_kappa={mean_kappa:.3f}"
            )

        unique_items = videos_used_df[["field", "id"] + metadata_cols].drop_duplicates()
        print("\nVideos/items included in the Cohen's kappa calculations:\n")
        print(f"unique item-label combinations used: {len(unique_items)}")
        preview_cols = ["field", "id"] + metadata_cols
        preview = unique_items[preview_cols].head(20)
        if not preview.empty:
            print(preview.to_string(index=False))
    else:
        print(
            "\nNo Cohen's kappa scores were written because no annotator pairs shared any items "
            "for the requested labels."
        )

    print(f"\nWrote {worst_out_path}")


if __name__ == "__main__":
    main()
