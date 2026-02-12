#!/usr/bin/env python3
"""Print completed annotation counts for the last 7 days from a Label Studio-style JSON export."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def parse_iso_datetime(value: str) -> datetime | None:
    """Parse common ISO-8601 datetime strings and return timezone-aware UTC datetime."""
    if not value:
        return None

    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


def extract_user_label(completed_by: Any) -> str:
    """Get a readable user identifier from Label Studio completed_by field."""
    if isinstance(completed_by, dict):
        for key in ("email", "username", "first_name", "id"):
            value = completed_by.get(key)
            if value:
                return str(value)
    if completed_by is None:
        return "unknown"
    return str(completed_by)


def iter_annotations(payload: Any):
    """Yield annotation objects from a top-level task list or wrapped object."""
    if isinstance(payload, list):
        tasks = payload
    elif isinstance(payload, dict) and isinstance(payload.get("tasks"), list):
        tasks = payload["tasks"]
    else:
        tasks = []

    for task in tasks:
        annotations = task.get("annotations", []) if isinstance(task, dict) else []
        for annotation in annotations:
            if isinstance(annotation, dict):
                yield annotation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count completed annotations in the last 7 days and group by user."
    )
    parser.add_argument("json_file", type=Path, help="Path to the exported JSON file")
    args = parser.parse_args()

    with args.json_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    now = datetime.now(timezone.utc)
    window_start = now - timedelta(days=7)

    by_user = Counter()
    total = 0

    for annotation in iter_annotations(payload):
        if annotation.get("was_cancelled"):
            continue

        timestamp = (
            annotation.get("updated_at")
            or annotation.get("created_at")
            or annotation.get("completed_at")
        )
        dt = parse_iso_datetime(str(timestamp)) if timestamp is not None else None
        if dt is None or dt < window_start:
            continue

        total += 1
        by_user[extract_user_label(annotation.get("completed_by"))] += 1

    print(f"Completed annotations in last 7 days: {total}")
    if by_user:
        print("By user:")
        for user, count in by_user.most_common():
            print(f"  - {user}: {count}")
    else:
        print("By user: none")


if __name__ == "__main__":
    main()
