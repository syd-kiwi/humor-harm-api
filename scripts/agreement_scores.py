import argparse
import json
import itertools
import math
from collections import Counter, defaultdict

import numpy as np


def load_tasks(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_ratings(tasks):
    """
    Returns:
      ratings[task_id][annotation_id][field] = tuple(sorted(choices)) or None
    """
    ratings = {}
    for task in tasks:
        tid = task.get("id")
        ratings[tid] = {}

        for ann in task.get("annotations", []):
            ann_id = ann.get("completed_by") or ann.get("id")
            fields = {}

            for r in ann.get("result", []):
                field = r.get("from_name")
                choices = (r.get("value") or {}).get("choices")

                if not choices:
                    fields[field] = None
                else:
                    fields[field] = tuple(sorted(str(c) for c in choices))

            ratings[tid][ann_id] = fields

    return ratings


def cohen_kappa(labels1, labels2):
    n = len(labels1)
    if n == 0:
        return float("nan")

    po = sum(1 for a, b in zip(labels1, labels2) if a == b) / n
    c1 = Counter(labels1)
    c2 = Counter(labels2)

    pe = 0.0
    keys = set(c1) | set(c2)
    for k in keys:
        pe += (c1[k] / n) * (c2.get(k, 0) / n)

    denom = 1.0 - pe
    if denom == 0:
        return float("nan")
    return (po - pe) / denom


def pairwise_exact_agreement(ratings, field):
    agree = []
    for tid, annos in ratings.items():
        vals = []
        for _, fields in annos.items():
            v = fields.get(field)
            if v is not None:
                vals.append(v)

        for v1, v2 in itertools.combinations(vals, 2):
            agree.append(1 if v1 == v2 else 0)

    return float(np.mean(agree)) if agree else float("nan"), len(agree)


def avg_pairwise_kappa_single_choice(ratings, field):
    pair_map = defaultdict(lambda: ([], []))

    for _, annos in ratings.items():
        vals = {}
        for aid, fields in annos.items():
            v = fields.get(field)
            if v is None:
                continue
            if isinstance(v, tuple) and len(v) == 1:
                v = v[0]
            vals[aid] = v

        for a1, a2 in itertools.combinations(sorted(vals.keys()), 2):
            l1, l2 = pair_map[(a1, a2)]
            l1.append(vals[a1])
            l2.append(vals[a2])
            pair_map[(a1, a2)] = (l1, l2)

    weighted = []
    total_n = 0

    for _, (l1, l2) in pair_map.items():
        k = cohen_kappa(l1, l2)
        n = len(l1)
        if not math.isnan(k) and n > 0:
            weighted.append((k, n))
            total_n += n

    if total_n == 0:
        return float("nan"), 0

    k_avg = sum(k * n for k, n in weighted) / total_n
    return k_avg, total_n


def krippendorff_alpha_nominal_single_choice(ratings, field):
    units = []
    for _, annos in ratings.items():
        vals = []
        for _, fields in annos.items():
            v = fields.get(field)
            if v is None:
                continue
            if isinstance(v, tuple) and len(v) == 1:
                v = v[0]
            vals.append(v)
        if len(vals) >= 2:
            units.append(vals)

    if not units:
        return float("nan"), 0

    do_num = 0.0
    do_den = 0.0
    all_vals = []

    for vals in units:
        all_vals.extend(vals)
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                do_num += 0.0 if vals[i] == vals[j] else 1.0
                do_den += 1.0

    do = do_num / do_den if do_den else float("nan")

    n = len(all_vals)
    freq = Counter(all_vals)
    de = 1.0 - sum((c / n) ** 2 for c in freq.values())

    if de == 0:
        return float("nan"), len(units)

    alpha = 1.0 - (do / de)
    return alpha, len(units)


def avg_pairwise_jaccard_multi_select(ratings, field):
    vals = []
    for _, annos in ratings.items():
        sets = []
        for _, fields in annos.items():
            v = fields.get(field)
            if v is None:
                continue
            sets.append(set(v) if isinstance(v, tuple) else {v})

        for s1, s2 in itertools.combinations(sets, 2):
            union = len(s1 | s2)
            inter = len(s1 & s2)
            j = (inter / union) if union else 1.0
            vals.append(j)

    return float(np.mean(vals)) if vals else float("nan"), len(vals)


def field_stats(ratings):
    field_lengths = defaultdict(Counter)
    fields_seen = set()

    for _, annos in ratings.items():
        for _, fdict in annos.items():
            for field, v in fdict.items():
                fields_seen.add(field)
                if v is None:
                    field_lengths[field]["none"] += 1
                else:
                    field_lengths[field][len(v) if isinstance(v, tuple) else 1] += 1

    single_choice = []
    multi_select = []

    for field in sorted(fields_seen):
        counts = field_lengths[field]
        has_multi = any(k not in ("none", 1) for k in counts.keys())
        if has_multi:
            multi_select.append(field)
        else:
            single_choice.append(field)

    return single_choice, multi_select


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Label Studio export json path")
    args = ap.parse_args()

    tasks = load_tasks(args.json)
    ratings = extract_ratings(tasks)

    task_count = len(tasks)
    ann_count = sum(len(t.get("annotations", [])) for t in tasks)

    print(f"Tasks: {task_count}")
    print(f"Annotations total: {ann_count}")

    single_fields, multi_fields = field_stats(ratings)

    print("\nSingle choice fields")
    for field in single_fields:
        pa, npa = pairwise_exact_agreement(ratings, field)
        k, nk = avg_pairwise_kappa_single_choice(ratings, field)
        a, nu = krippendorff_alpha_nominal_single_choice(ratings, field)
        print(f"\nField: {field}")
        print(f"Pairwise exact agreement: {pa:.4f} based on {npa} pairs")
        print(f"Average pairwise Cohen kappa: {k:.4f} based on {nk} aligned items")
        print(f"Krippendorff alpha nominal: {a:.4f} based on {nu} tasks")

    print("\nMulti select fields")
    for field in multi_fields:
        pa, npa = pairwise_exact_agreement(ratings, field)
        j, nj = avg_pairwise_jaccard_multi_select(ratings, field)
        print(f"\nField: {field}")
        print(f"Pairwise exact agreement: {pa:.4f} based on {npa} pairs")
        print(f"Average pairwise Jaccard: {j:.4f} based on {nj} pairs")


if __name__ == "__main__":
    main()
