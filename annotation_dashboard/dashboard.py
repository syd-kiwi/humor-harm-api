import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# =========================
# Label Studio field config
# =========================
FUNNY_FIELD = "humor_presence"          # "Humor" | "Not Humor"
RHETORICAL_FIELD = "rhetorical_device"  # choices
HUMOR_TYPE_FIELD = "humor_type"         # choices
TOPIC_FIELD = "joke_topic"              # may not exist yet

FUNNY_VALUE = "humor"
NOT_FUNNY_VALUE = "not humor"

VOTE_MODE_DEFAULT = "majority"  # "majority" or "latest"

DEFAULT_JSON_PATH = "/home/kiwi-pandas/Documents/humor-harm-api/annotation_dashboard/02-22.json"


def try_load_json_list(p: Path) -> Optional[List[Dict[str, Any]]]:
    """Try to load a JSON file that is either a list of tasks or a wrapped dict."""
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for k in ["tasks", "results", "data"]:
            if k in data and isinstance(data[k], list):
                return data[k]
        if "id" in data and "data" in data:
            return [data]

    return None


def try_load_jsonl(p: Path) -> Optional[List[Dict[str, Any]]]:
    """Try to load JSONL: one JSON object per line."""
    tasks: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                return None
            if isinstance(obj, dict):
                tasks.append(obj)
            else:
                return None
    return tasks if tasks else None


def load_tasks(json_path: str) -> List[Dict[str, Any]]:
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {json_path}")

    # First try normal JSON
    try:
        tasks = try_load_json_list(p)
        if tasks is not None:
            return tasks
    except json.JSONDecodeError:
        pass

    # Then try JSONL
    tasks = try_load_jsonl(p)
    if tasks is not None:
        return tasks

    raise ValueError(
        "Could not parse file as JSON list or JSONL. "
        "If this is a different Label Studio export format, paste the first 30 lines."
    )


def normalize_choice(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        s = v.strip()
        return [s] if s else []
    if isinstance(v, list):
        out = []
        for x in v:
            s = str(x).strip()
            if s:
                out.append(s)
        return out
    s = str(v).strip()
    return [s] if s else []


def extract_from_result_items(
    result_items: List[Dict[str, Any]]
) -> Tuple[Optional[str], List[str], Optional[str], Optional[str]]:
    funny_found: Optional[str] = None
    topics_found: List[str] = []
    rhetorical_found: Optional[str] = None
    humor_type_found: Optional[str] = None

    for r in result_items or []:
        from_name = str(r.get("from_name", "")).strip()
        value = r.get("value", {}) or {}
        choices = normalize_choice(value.get("choices"))
        if not choices:
            continue

        c0 = choices[0]
        c0_lower = c0.lower()

        if from_name == FUNNY_FIELD:
            if c0_lower == FUNNY_VALUE:
                funny_found = "Funny"
            elif c0_lower == NOT_FUNNY_VALUE:
                funny_found = "Not funny"
            else:
                if "not" in c0_lower and "humor" in c0_lower:
                    funny_found = "Not funny"
                elif "humor" in c0_lower:
                    funny_found = "Funny"

        elif from_name == TOPIC_FIELD:
            topics_found.extend([t.strip() for t in choices if str(t).strip()])

        elif from_name == RHETORICAL_FIELD:
            rhetorical_found = c0

        elif from_name == HUMOR_TYPE_FIELD:
            humor_type_found = c0

    seen = set()
    topics_found = [x for x in topics_found if not (x in seen or seen.add(x))]

    return funny_found, topics_found, rhetorical_found, humor_type_found


def pick_from_annotations(
    annotations: List[Dict[str, Any]], vote_mode: str
) -> Tuple[Optional[str], List[str], Optional[str], Optional[str]]:
    if not annotations:
        return None, [], None, None

    extracted = []
    for ann in annotations:
        result_items = ann.get("result", []) or []
        funny, topics, rhet, htype = extract_from_result_items(result_items)
        created_at = str(ann.get("created_at", ""))  # may be empty
        extracted.append((funny, topics, rhet, htype, created_at))

    if vote_mode == "latest":
        extracted.sort(key=lambda x: x[4] or "")
        funny, topics, rhet, htype, _ = extracted[-1]
        return funny, topics, rhet, htype

    # majority vote for funny
    funny_votes = [x[0] for x in extracted if x[0] in ["Funny", "Not funny"]]
    funny_label = None
    if funny_votes:
        c = Counter(funny_votes)
        if c["Funny"] > c["Not funny"]:
            funny_label = "Funny"
        elif c["Not funny"] > c["Funny"]:
            funny_label = "Not funny"
        else:
            extracted.sort(key=lambda x: x[4] or "")
            funny_label = extracted[-1][0]

    # union topics
    topics_union: List[str] = []
    seen = set()
    for _, topics, _, _, _ in extracted:
        for t in topics:
            if t not in seen:
                seen.add(t)
                topics_union.append(t)

    # latest non-empty for rhet and humor type
    extracted.sort(key=lambda x: x[4] or "")
    rhet_latest = next((x[2] for x in reversed(extracted) if x[2]), None)
    htype_latest = next((x[3] for x in reversed(extracted) if x[3]), None)

    return funny_label, topics_union, rhet_latest, htype_latest


def build_df(tasks: List[Dict[str, Any]], vote_mode: str) -> pd.DataFrame:
    rows = []
    for t in tasks:
        task_id = t.get("id")
        data = t.get("data", {}) or {}
        video = data.get("video", "")
        transcript = data.get("transcript_text", "")

        annotations = t.get("annotations", []) or []
        funny, topics, rhet, htype = pick_from_annotations(annotations, vote_mode)

        has_funny = funny in ["Funny", "Not funny"]
        has_topic = len(topics) > 0

        # New completion definition:
        # Complete = annotated + humor_presence filled + topic filled
        complete = bool(annotations) and has_funny and has_topic

        rows.append(
            {
                "task_id": task_id,
                "video": video,
                "funny": funny,
                "rhetorical_device": rhet or "",
                "humor_type": htype or "",
                "joke_topic": ", ".join(topics) if topics else "",
                "has_topic": has_topic,
                "complete": complete,
                "has_annotation": bool(annotations),
                "total_annotations": t.get("total_annotations", len(annotations)),
                "created_at": t.get("created_at", ""),
                "updated_at": t.get("updated_at", ""),
                "transcript_len": len(transcript or ""),
            }
        )

    return pd.DataFrame(rows)


def plot_bar(series: pd.Series, title: str, ylabel: str = "Count"):
    counts = series.value_counts(dropna=False)
    fig = plt.figure()
    ax = plt.gca()
    counts.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    st.pyplot(fig)


def explode_topics(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["topic_list"] = tmp["joke_topic"].fillna("").apply(
        lambda s: [x.strip() for x in s.split(",") if x.strip()]
    )
    tmp = tmp.explode("topic_list")
    tmp = tmp[tmp["topic_list"].notna() & (tmp["topic_list"] != "")]
    return tmp


# =========================
# Streamlit app
# =========================
st.set_page_config(page_title="Humor Harm Dashboard", layout="wide")
st.title("Humor Harm Label Studio Dashboard")

with st.sidebar:
    st.header("Input")
    json_path = st.text_input("Path to Label Studio export", value=DEFAULT_JSON_PATH)
    vote_mode = st.selectbox("Resolve multiple annotations", ["majority", "latest"], index=0)
    st.caption("majority = vote for funny, latest non-empty for other fields")


tasks: List[Dict[str, Any]] = []
try:
    tasks = load_tasks(json_path)
except Exception as e:
    st.error(f"Failed to load file: {e}")
    st.stop()

df = build_df(tasks, vote_mode)

# Metrics
total = len(df)
funny_ct = int((df["funny"] == "Funny").sum())
not_funny_ct = int((df["funny"] == "Not funny").sum())
missing_funny_ct = int(df["funny"].isna().sum())
complete_ct = int(df["complete"].sum())
annotated_ct = int(df["has_annotation"].sum())

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total tasks", f"{total}")
c2.metric("Funny", f"{funny_ct}")
c3.metric("Not funny", f"{not_funny_ct}")
c4.metric("Missing humor label", f"{missing_funny_ct}")
c5.metric("Annotated", f"{annotated_ct}")
c6.metric("Complete", f"{complete_ct}")

# Filters
st.subheader("Filters")
f1, f2, f3, f4 = st.columns(4)
only_annotated = f1.checkbox("Only annotated", value=False)
only_complete = f2.checkbox("Only complete", value=False)
funny_filter = f3.selectbox("Humor label", ["All", "Funny", "Not funny", "Missing"], index=0)
topic_contains = f4.text_input("Topic contains", value="")

df_f = df.copy()
if only_annotated:
    df_f = df_f[df_f["has_annotation"]]
if only_complete:
    df_f = df_f[df_f["complete"]]
if funny_filter != "All":
    if funny_filter == "Missing":
        df_f = df_f[df_f["funny"].isna()]
    else:
        df_f = df_f[df_f["funny"] == funny_filter]
if topic_contains.strip():
    df_f = df_f[df_f["joke_topic"].str.contains(topic_contains.strip(), case=False, na=False)]

# Charts
left, right = st.columns(2)
with left:
    st.subheader("Funny vs Not funny")
    plot_bar(df_f["funny"].fillna("Missing"), "Humor presence")
with right:
    st.subheader("Completion status")
    plot_bar(df_f["complete"].map({True: "Complete", False: "Not complete"}), "Complete vs not complete")

# Distributions
st.subheader("Distributions")
d1, d2 = st.columns(2)
with d1:
    plot_bar(df_f["rhetorical_device"].replace("", "Missing"), "Rhetorical device")
with d2:
    plot_bar(df_f["humor_type"].replace("", "Missing"), "Humor type")

# Topics
st.subheader("Topics")
exploded = explode_topics(df)
if len(exploded) == 0:
    st.info("No joke_topic found yet in your annotations. Once you add it, this section will populate.")
else:
    grp = exploded.groupby("topic_list").agg(
        tasks_with_topic=("task_id", "nunique"),
        complete_tasks=("complete", "sum"),
    ).reset_index().rename(columns={"topic_list": "topic"})
    grp["completion_rate"] = (grp["complete_tasks"] / grp["tasks_with_topic"] * 100.0).round(2)
    st.dataframe(grp.sort_values("tasks_with_topic", ascending=False), use_container_width=True)

# Table
st.subheader("Tasks table")
show_cols = [
    "task_id",
    "funny",
    "rhetorical_device",
    "humor_type",
    "joke_topic",
    "complete",
    "has_annotation",
    "total_annotations",
    "updated_at",
    "video",
]
st.dataframe(
    df_f[show_cols].sort_values(["complete", "has_annotation", "updated_at"], ascending=False),
    use_container_width=True,
)