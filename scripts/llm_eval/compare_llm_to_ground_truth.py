#!/usr/bin/env python3
"""Compare LLM annotation CSV outputs against ground truth labels.

Supports ground truth from either:
- Label Studio JSON task exports (e.g. annotation_dashboard/03-18.json)
- Dataset CSV files with annotation columns (e.g. unified_dataset.csv)

Outputs:
- long-form per-item comparison CSV
- per-model/per-field summary CSV
- markdown summary report
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

FIELDS: Sequence[str] = (
    "humor_presence",
    "joke_topic",
    "rhetorical_device",
    "stand_up",
    "humor_type",
    "target_category",
    "dark_intensity",
)

TEXT_FIELDS = {"humor_presence", "stand_up", "humor_type", "dark_intensity"}
LIST_FIELDS = {"joke_topic", "rhetorical_device", "target_category"}

LABEL_STUDIO_FIELD_MAP = {
    "humor_presence": "humor_presence",
    "joke_topic": "joke_topic",
    "rhetorical_device": "rhetorical_device",
    "stand_up": "stand_up",
    "humor_type": "humor_type",
    "target_category": "target_category",
    "dark_intensity": "dark_intensity",
    "note": "note",
}


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_scalar(value: object) -> str:
    text = normalize_text(value)
    return text.lower()


def parse_listlike(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, dict):
        if "choices" in value and isinstance(value["choices"], list):
            raw_items = value["choices"]
        else:
            raw_items = []
    else:
        text = normalize_text(value)
        if not text:
            return []
        if text.startswith("{") and text.endswith("}"):
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                obj = None
            if isinstance(obj, dict) and isinstance(obj.get("choices"), list):
                raw_items = obj["choices"]
            else:
                raw_items = [text]
        elif text.startswith("[") and text.endswith("]"):
            try:
                obj = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                obj = None
            if isinstance(obj, list):
                raw_items = obj
            else:
                raw_items = [part.strip() for part in text[1:-1].split(",") if part.strip()]
        else:
            raw_items = [part.strip() for part in text.split(",") if part.strip()]

    normalized = []
    seen = set()
    for item in raw_items:
        item_text = normalize_text(item)
        if not item_text:
            continue
        item_key = item_text.lower()
        if item_key not in seen:
            seen.add(item_key)
            normalized.append(item_text)
    return sorted(normalized, key=str.lower)


def extract_video_id(value: object) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    candidate = text.split("/")[-1]
    candidate = re.sub(r"\.(mp4|mov|mkv|webm)$", "", candidate, flags=re.IGNORECASE)
    return candidate.strip()


def detect_model_name(path: Path) -> str:
    name = path.stem
    if name.endswith("_annotations"):
        name = name[: -len("_annotations")]
    return name


def choose_annotation(task: dict) -> Optional[dict]:
    annotations = task.get("annotations") or []
    if not annotations:
        return None
    complete = [ann for ann in annotations if not ann.get("was_cancelled")]
    if not complete:
        complete = annotations
    complete.sort(key=lambda ann: ann.get("updated_at") or ann.get("created_at") or "")
    return complete[-1]


def annotation_results_to_record(results: Iterable[dict]) -> Dict[str, object]:
    record: Dict[str, object] = {field: ([] if field in LIST_FIELDS else "") for field in FIELDS}
    for result in results:
        field = LABEL_STUDIO_FIELD_MAP.get(result.get("from_name"))
        if not field:
            continue
        value = result.get("value") or {}
        choices = value.get("choices")
        if field in LIST_FIELDS:
            record[field] = parse_listlike(choices)
        else:
            selected = choices[0] if isinstance(choices, list) and choices else value.get("text", [""])
            if isinstance(selected, list):
                selected = selected[0] if selected else ""
            record[field] = normalize_text(selected)
    return record


def load_ground_truth_json(path: Path) -> Dict[str, Dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records: Dict[str, Dict[str, object]] = {}
    for task in data:
        annotation = choose_annotation(task)
        if not annotation:
            continue
        qid = normalize_text(task.get("id") or task.get("inner_id"))
        if not qid:
            continue
        record = annotation_results_to_record(annotation.get("result") or [])
        task_data = task.get("data") or {}
        record.update(
            {
                "id": qid,
                "task_id": normalize_text(task.get("id")),
                "video_id": extract_video_id(task_data.get("video")),
                "transcript_text": normalize_text(task_data.get("transcript_text")),
            }
        )
        records[qid] = record
    return records


def load_ground_truth_csv(path: Path) -> Dict[str, Dict[str, object]]:
    records: Dict[str, Dict[str, object]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            qid = normalize_text(row.get("id") or row.get("inner_id") or row.get("task_id"))
            if not qid:
                continue
            record: Dict[str, object] = {
                "id": qid,
                "task_id": normalize_text(row.get("task_id")),
                "video_id": extract_video_id(row.get("video_id") or row.get("video") or row.get("url")),
                "transcript_text": normalize_text(row.get("transcript_text")),
            }
            for field in FIELDS:
                if field in LIST_FIELDS:
                    record[field] = parse_listlike(row.get(field))
                else:
                    record[field] = normalize_text(row.get(field))
            records[qid] = record
    return records


def load_ground_truth(path: Path) -> Dict[str, Dict[str, object]]:
    if path.suffix.lower() == ".json":
        return load_ground_truth_json(path)
    if path.suffix.lower() == ".csv":
        return load_ground_truth_csv(path)
    raise ValueError(f"Unsupported ground truth file type: {path}")


def load_llm_rows(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            qid = normalize_text(row.get("QID") or row.get("id"))
            if not qid:
                continue
            parsed: Dict[str, object] = {
                "model": detect_model_name(path),
                "source_file": path.name,
                "id": qid,
                "parse_issues": normalize_text(row.get("ParseIssues")),
            }
            for field in FIELDS:
                if field in LIST_FIELDS:
                    parsed[field] = parse_listlike(row.get(field))
                else:
                    parsed[field] = normalize_text(row.get(field))
            rows.append(parsed)
    return rows


def values_match(field: str, truth: object, pred: object) -> bool:
    if field in LIST_FIELDS:
        return [normalize_scalar(x) for x in parse_listlike(truth)] == [normalize_scalar(x) for x in parse_listlike(pred)]
    return normalize_scalar(truth) == normalize_scalar(pred)


def display_value(field: str, value: object) -> str:
    if field in LIST_FIELDS:
        items = parse_listlike(value)
        return " | ".join(items) if items else "[]"
    return normalize_text(value)


def safe_pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return math.nan
    return numerator / denominator


def compare_records(ground_truth: Dict[str, Dict[str, object]], llm_rows: List[Dict[str, object]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    long_rows: List[Dict[str, object]] = []
    summary_map: Dict[Tuple[str, str], Dict[str, object]] = {}

    for llm_row in llm_rows:
        model = str(llm_row["model"])
        qid = str(llm_row["id"])
        truth_row = ground_truth.get(qid)

        overall_matches = 0
        overall_total = 0
        missing_truth = truth_row is None

        for field in FIELDS:
            key = (model, field)
            if key not in summary_map:
                summary_map[key] = {
                    "model": model,
                    "field": field,
                    "matched": 0,
                    "evaluated": 0,
                    "missing_ground_truth": 0,
                    "examples": Counter(),
                }
            summary = summary_map[key]

            if truth_row is None:
                summary["missing_ground_truth"] += 1
                long_rows.append(
                    {
                        "model": model,
                        "id": qid,
                        "video_id": "",
                        "field": field,
                        "ground_truth": "",
                        "prediction": display_value(field, llm_row.get(field)),
                        "match": "",
                        "parse_issues": llm_row.get("parse_issues", ""),
                    }
                )
                continue

            truth_value = truth_row.get(field)
            pred_value = llm_row.get(field)
            match = values_match(field, truth_value, pred_value)
            summary["evaluated"] += 1
            summary["matched"] += int(match)
            if not match:
                example_key = f"GT={display_value(field, truth_value)} || PRED={display_value(field, pred_value)}"
                summary["examples"][example_key] += 1

            overall_total += 1
            overall_matches += int(match)

            long_rows.append(
                {
                    "model": model,
                    "id": qid,
                    "video_id": truth_row.get("video_id", ""),
                    "field": field,
                    "ground_truth": display_value(field, truth_value),
                    "prediction": display_value(field, pred_value),
                    "match": int(match),
                    "parse_issues": llm_row.get("parse_issues", ""),
                }
            )

        if not missing_truth:
            key = (model, "__overall__")
            if key not in summary_map:
                summary_map[key] = {
                    "model": model,
                    "field": "__overall__",
                    "matched": 0,
                    "evaluated": 0,
                    "missing_ground_truth": 0,
                    "examples": Counter(),
                }
            summary_map[key]["matched"] += overall_matches
            summary_map[key]["evaluated"] += overall_total

    summary_rows: List[Dict[str, object]] = []
    for (model, field), summary in sorted(summary_map.items()):
        top_examples = summary["examples"].most_common(5)
        summary_rows.append(
            {
                "model": model,
                "field": field,
                "matched": summary["matched"],
                "evaluated": summary["evaluated"],
                "accuracy": round(safe_pct(summary["matched"], summary["evaluated"]), 4)
                if summary["evaluated"]
                else "",
                "missing_ground_truth": summary["missing_ground_truth"],
                "top_mismatches": " ; ".join(f"{count}x {example}" for example, count in top_examples),
            }
        )
    return long_rows, summary_rows


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_markdown_report(
    ground_truth_path: Path,
    llm_dir: Path,
    summary_rows: List[Dict[str, object]],
    llm_file_count: int,
    ground_truth_count: int,
) -> str:
    per_model: Dict[str, Dict[str, Dict[str, object]]] = defaultdict(dict)
    for row in summary_rows:
        per_model[str(row["model"])][str(row["field"])] = row

    overall_ranking = []
    for model, fields in per_model.items():
        overall = fields.get("__overall__")
        if overall and overall.get("evaluated"):
            overall_ranking.append((float(overall["accuracy"]), model, overall))
    overall_ranking.sort(reverse=True)

    lines = [
        "# LLM vs Ground Truth Summary",
        "",
        f"- Ground truth file: `{ground_truth_path}`",
        f"- LLM output directory: `{llm_dir}`",
        f"- Ground truth items loaded: **{ground_truth_count}**",
        f"- LLM output files compared: **{llm_file_count}**",
        "",
        "## Overall ranking",
        "",
        "| Rank | Model | Field-level accuracy | Matches | Evaluated |",
        "|---|---|---:|---:|---:|",
    ]
    for idx, (_, model, overall) in enumerate(overall_ranking, start=1):
        lines.append(
            f"| {idx} | {model} | {float(overall['accuracy']):.2%} | {overall['matched']} | {overall['evaluated']} |"
        )
    if not overall_ranking:
        lines.append("| - | No aligned records found | - | - | - |")

    for model in sorted(per_model):
        fields = per_model[model]
        lines.extend(["", f"## {model}", "", "| Field | Accuracy | Matches | Evaluated | Top mismatches |", "|---|---:|---:|---:|---|"])
        ordered_fields = ["__overall__", *FIELDS]
        for field in ordered_fields:
            row = fields.get(field)
            if not row:
                continue
            accuracy = row["accuracy"]
            accuracy_str = f"{float(accuracy):.2%}" if accuracy != "" else ""
            top_mismatches = str(row.get("top_mismatches", ""))
            lines.append(
                f"| {field} | {accuracy_str} | {row['matched']} | {row['evaluated']} | {top_mismatches} |"
            )

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ground-truth",
        default="annotation_dashboard/03-18.json",
        help="Ground truth JSON or CSV file. Defaults to the Label Studio JSON export.",
    )
    parser.add_argument(
        "--llm-dir",
        default="scripts/llm_eval/outputs",
        help="Directory containing *_annotations.csv files.",
    )
    parser.add_argument(
        "--output-prefix",
        default="scripts/llm_eval/outputs/llm_vs_ground_truth",
        help="Prefix for the generated CSV/Markdown outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ground_truth_path = Path(args.ground_truth)
    llm_dir = Path(args.llm_dir)
    output_prefix = Path(args.output_prefix)

    ground_truth = load_ground_truth(ground_truth_path)
    llm_files = sorted(llm_dir.glob("*_annotations.csv"))
    llm_rows: List[Dict[str, object]] = []
    for llm_file in llm_files:
        llm_rows.extend(load_llm_rows(llm_file))

    long_rows, summary_rows = compare_records(ground_truth, llm_rows)

    long_csv = output_prefix.with_name(output_prefix.name + "_long.csv")
    summary_csv = output_prefix.with_name(output_prefix.name + "_summary.csv")
    summary_md = output_prefix.with_name(output_prefix.name + "_summary.md")

    write_csv(
        long_csv,
        long_rows,
        ["model", "id", "video_id", "field", "ground_truth", "prediction", "match", "parse_issues"],
    )
    write_csv(
        summary_csv,
        summary_rows,
        ["model", "field", "matched", "evaluated", "accuracy", "missing_ground_truth", "top_mismatches"],
    )
    summary_md.write_text(
        build_markdown_report(ground_truth_path, llm_dir, summary_rows, len(llm_files), len(ground_truth)),
        encoding="utf-8",
    )

    print(f"Ground truth items loaded: {len(ground_truth)}")
    print(f"LLM files compared: {len(llm_files)}")
    print(f"Wrote: {long_csv}")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {summary_md}")


if __name__ == "__main__":
    main()
