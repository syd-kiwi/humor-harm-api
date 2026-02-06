#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Dict, List, Optional

import pandas as pd


def normalize_label_value(result: Dict[str, Any]) -> Optional[str]:
    value = result.get("value", {})
    if not isinstance(value, dict):
        return None

    if "choices" in value and isinstance(value["choices"], list):
        return "|".join(map(str, value["choices"]))

    if "text" in value:
        t = value["text"]
        if isinstance(t, list):
            return " ".join(map(str, t)).strip()
        return str(t).strip()

    if "rating" in value:
        return str(value["rating"])

    return None


def parse_labelstudio_export(json_path: str) -> pd.DataFrame:
    """
    Returns tidy df with columns:
    item_id, annotator_id, field, label
    """
    with open(json_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    rows = []
    for task in tasks:
        item_id = str(task.get("id", task.get("task_id", task.get("pk", ""))))

        for ann in task.get("annotations", []) or []:
            annotator = (
                ann.get("created_username")
                or ann.get("created_by")
                or ann.get("completed_by")
                or ann.get("user")
                or ann.get("id")
            )
            annotator_id = str(annotator)

            for res in ann.get("result", []) or []:
                field = res.get("from_name") or res.get("name")
                if not field:
                    continue

                label = normalize_label_value(res)
                if not label:
                    continue

                rows.append(
                    {
                        "item_id": item_id,
                        "annotator_id": annotator_id,
                        "field": str(field),
                        "label": label,
                    }
                )

    if not rows:
        raise ValueError(
            "Parsed 0 labels. Your export structure likely differs. "
            "Paste a small redacted snippet and I will adapt the parser."
        )

    return pd.DataFrame(rows)


def pairwise_percent_agreement(labels_by_item: Dict[str, Dict[str, Any]]) -> Optional[float]:
    matches = 0
    total = 0
    for _, ann_map in labels_by_item.items():
        ann_ids = list(ann_map.keys())
        if len(ann_ids) < 2:
            continue
        for a, b in combinations(ann_ids, 2):
            total += 1
            if ann_map[a] == ann_map[b]:
                matches += 1
    return None if total == 0 else matches / total


def cohen_kappa_for_two_lists(y1: List[Any], y2: List[Any]) -> Optional[float]:
    if len(y1) == 0 or len(y1) != len(y2):
        return None

    n = len(y1)
    po = sum(1 for i in range(n) if y1[i] == y2[i]) / n

    c1 = Counter(y1)
    c2 = Counter(y2)
    pe = 0.0
    for k in set(c1.keys()) | set(c2.keys()):
        pe += (c1.get(k, 0) / n) * (c2.get(k, 0) / n)

    denom = 1.0 - pe
    if denom == 0:
        return None
    return (po - pe) / denom


def avg_pairwise_cohen_kappa(labels_by_item: Dict[str, Dict[str, Any]]) -> Optional[float]:
    all_annotators = set()
    for ann_map in labels_by_item.values():
        all_annotators.update(ann_map.keys())
    all_annotators = sorted(all_annotators)

    kappas = []
    for a, b in combinations(all_annotators, 2):
        y1, y2 = [], []
        for _, ann_map in labels_by_item.items():
            if a in ann_map and b in ann_map:
                y1.append(ann_map[a])
                y2.append(ann_map[b])
        k = cohen_kappa_for_two_lists(y1, y2)
        if k is not None:
            kappas.append(k)

    return None if not kappas else sum(kappas) / len(kappas)


def krippendorff_alpha_nominal(labels_by_item: Dict[str, Dict[str, Any]]) -> Optional[float]:
    coincidence: Dict[Any, Dict[Any, float]] = defaultdict(lambda: defaultdict(float))
    categories = set()

    for _, ann_map in labels_by_item.items():
        vals = list(ann_map.values())
        if len(vals) < 2:
            continue
        freq = Counter(vals)
        cats = list(freq.keys())
        categories.update(cats)

        n_u = sum(freq.values())
        if n_u < 2:
            continue

        for c in cats:
            for k in cats:
                add = freq[c] * (freq[k] - (1 if c == k else 0)) / (n_u - 1)
                coincidence[c][k] += add

    categories = sorted(categories)
    if not categories:
        return None

    total = 0.0
    diag = 0.0
    marginals: Dict[Any, float] = defaultdict(float)

    for c in categories:
        for k in categories:
            v = coincidence[c].get(k, 0.0)
            total += v
            if c == k:
                diag += v
            marginals[c] += v

    if total == 0:
        return None

    Do = 1.0 - (diag / total)

    M = sum(marginals.values())
    if M <= 1:
        return None

    sum_same = 0.0
    for c in categories:
        m = marginals[c]
        sum_same += m * (m - 1.0)

    De = 1.0 - (sum_same / (M * (M - 1.0)))
    if De == 0:
        return None

    return 1.0 - (Do / De)


def labels_by_item_for_field(df: pd.DataFrame, field: str) -> Dict[str, Dict[str, Any]]:
    sub = df[df["field"] == field]
    out: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for _, row in sub.iterrows():
        out[str(row["item_id"])][str(row["annotator_id"])] = row["label"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Agreement metrics from Label Studio JSON export")
    ap.add_argument("--json", required=True, help="Path to export JSON")
    ap.add_argument("--out_json", default="agreement_scores.json", help="Output metrics JSON")
    ap.add_argument("--out_csv", default="", help="Optional: write tidy labels CSV")
    args = ap.parse_args()

    df = parse_labelstudio_export(args.json)

    if args.out_csv:
        df.to_csv(args.out_csv, index=False)

    fields = sorted(df["field"].dropna().unique().tolist())

    results: Dict[str, Any] = {"input_json": args.json, "fields": {}}

    for field in fields:
        lb = labels_by_item_for_field(df, field)
        n_items = len(lb)
        n_ratings = sum(len(v) for v in lb.values())
        annotators = sorted({a for v in lb.values() for a in v.keys()})

        pa = pairwise_percent_agreement(lb)
        ck = avg_pairwise_cohen_kappa(lb)
        ka = krippendorff_alpha_nominal(lb)

        results["fields"][field] = {
            "counts": {
                "num_items": n_items,
                "num_ratings": n_ratings,
                "num_annotators": len(annotators),
            },
            "metrics": {
                "percent_agreement_pairwise": None if pa is None else round(pa, 4),
                "cohen_kappa_avg_pairwise": None if ck is None else round(ck, 4),
                "krippendorff_alpha_nominal": None if ka is None else round(ka, 4),
            },
        }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {args.out_json}")
    print("Fields:")
    for field in fields:
        m = results["fields"][field]["metrics"]
        c = results["fields"][field]["counts"]
        print(
            f"{field}: items {c['num_items']} ratings {c['num_ratings']} "
            f"PA {m['percent_agreement_pairwise']} "
            f"Kappa {m['cohen_kappa_avg_pairwise']} "
            f"Alpha {m['krippendorff_alpha_nominal']}"
        )


if __name__ == "__main__":
    main()

