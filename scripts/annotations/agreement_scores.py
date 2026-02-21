from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# -----------------------------
# Metrics
# -----------------------------

def cohen_kappa(pairs: List[Tuple[str, str]]) -> Optional[float]:
    if not pairs:
        return None

    n = len(pairs)
    po = sum(1 for a, b in pairs if a == b) / n

    a_counts = Counter(a for a, _ in pairs)
    b_counts = Counter(b for _, b in pairs)

    pe = 0.0
    for c in set(a_counts) | set(b_counts):
        pe += (a_counts.get(c, 0) / n) * (b_counts.get(c, 0) / n)

    if math.isclose(1.0 - pe, 0.0):
        return None

    return (po - pe) / (1.0 - pe)


def krippendorff_alpha_nominal(items: List[List[Optional[str]]]) -> Optional[float]:
    """
    Nominal Krippendorff alpha.
    items: list of items, each item is list of ratings (None allowed)
    """
    coincidence = defaultdict(lambda: defaultdict(float))
    categories = set()

    for vals in items:
        vals = [v for v in vals if v is not None and pd.notna(v)]
        m = len(vals)
        if m < 2:
            continue

        counts = Counter(vals)
        categories.update(counts.keys())
        denom = m - 1

        for c in counts:
            for k in counts:
                if c == k:
                    coincidence[c][k] += counts[c] * (counts[c] - 1) / denom
                else:
                    coincidence[c][k] += counts[c] * counts[k] / denom

    if not categories:
        return None

    n_c = {c: sum(coincidence[c].values()) for c in categories}
    n = sum(n_c.values())
    if n <= 0:
        return None

    do = 0.0
    for c in categories:
        for k in categories:
            if c != k:
                do += coincidence[c][k]
    do = do / n

    de = 0.0
    for c in categories:
        for k in categories:
            if c != k:
                de += n_c[c] * n_c[k]

    if n <= 1:
        return None

    de = de / (n * (n - 1))

    if math.isclose(de, 0.0):
        return None

    return 1.0 - (do / de)


# -----------------------------
# Option A: rhetorical_device -> binary presence
# -----------------------------

def _safe_lower(x: Any) -> str:
    return str(x).strip().lower()


def irony_or_satire_present(cell):
    """
    Option A:
    Yes if rhetorical_device indicates Irony or Satire is present.
    No otherwise.
    Returns None only when the cell is truly missing/blank.
    """
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return None

    s = str(cell).strip()
    if not s:
        return None

    lowered = s.lower()

    # If Label Studio exported JSON like {"choices":["Irony","Satire"]}
    if lowered.startswith("{") or lowered.startswith("["):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                choices = obj.get("choices", [])
            elif isinstance(obj, list):
                choices = obj
            else:
                choices = []

            choices_l = [str(c).strip().lower() for c in choices if c is not None]
            return "Yes" if ("irony" in choices_l or "satire" in choices_l) else "No"
        except Exception:
            # fall through to substring rule
            pass

    # Plain string export like "None of these" or "Irony" or "Satire"
    return "Yes" if ("irony" in lowered or "satire" in lowered) else "No"

# -----------------------------
# Agreement computation per field
# -----------------------------

def compute_for_label(
    df: pd.DataFrame,
    item_col: str,
    annotator_col: str,
    label_col: str,
) -> Dict[str, Any]:
    two_pairs: List[Tuple[str, str]] = []
    three_plus_items: List[List[str]] = []
    skipped = 0

    sub = df[[item_col, annotator_col, label_col]].copy()
    sub = sub.dropna(subset=[label_col])

    for _, g in sub.groupby(item_col):
        # ensure one label per annotator per item
        g2 = g.drop_duplicates(subset=[annotator_col])

        vals = [v for v in g2[label_col].tolist() if v is not None and pd.notna(v)]
        if len(vals) < 2:
            skipped += 1
        elif len(vals) == 2:
            two_pairs.append((str(vals[0]), str(vals[1])))
        else:
            three_plus_items.append([str(v) for v in vals])

    kappa = cohen_kappa(two_pairs)
    alpha = krippendorff_alpha_nominal(three_plus_items)

    return {
        "field": label_col,
        "two_annotator_items": len(two_pairs),
        "three_plus_annotator_items": len(three_plus_items),
        "skipped_items_less_than_2": skipped,
        "cohen_kappa": "" if kappa is None else round(kappa, 6),
        "krippendorff_alpha_nominal": "" if alpha is None else round(alpha, 6),
    }


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="Label Studio CSV export")
    ap.add_argument("--item_col", default="id")
    ap.add_argument("--annotator_col", default="annotator")
    ap.add_argument(
        "--labels",
        nargs="+",
        default=[
            "humor_presence",
            "humor_type",
            "dark_intensity",
            "target_category",
            "stand_up",
            "irony_or_satire_present",  # Option A
        ],
        help="Label columns to score (default includes Option A binary field).",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv_path)
    df = pd.read_csv(csv_path)

    # Add derived binary label for Option A
    if "rhetorical_device" in df.columns and "irony_or_satire_present" not in df.columns:
        df["irony_or_satire_present"] = df["rhetorical_device"].apply(irony_or_satire_present)

    # Validate columns
    needed = [args.item_col, args.annotator_col] + list(args.labels)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(
            f"Missing columns: {missing}\n"
            f"Available columns: {list(df.columns)}\n"
            f"Tip: if you want rhetorical presence, keep 'rhetorical_device' in the CSV."
        )

    rows: List[Dict[str, Any]] = []
    for label in args.labels:
        rows.append(compute_for_label(df, args.item_col, args.annotator_col, label))

    out_path = csv_path.with_name(f"{csv_path.stem}_agreement_scores.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
