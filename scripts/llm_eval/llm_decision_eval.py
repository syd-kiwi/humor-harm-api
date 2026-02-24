import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import anthropic
import openai
import torch
from google import genai
from openai import OpenAI
from together import Together
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = """You are an annotation assistant for humor content analysis.
Your task is to read a content item and return labels for the schema below.

Return ONLY valid JSON (no markdown), with exactly these keys:
{
  "humor_presence": "Humor" | "Not Humor" | "Almost or not sure",
  "joke_topic": ["Relationships" | "Family" | "Work" | "Money" | "School" | "Politics and society" | "Crime and law enforcement" | "War and conflict" | "Celebrity and pop culture" | "Other"],
  "rhetorical_device": ["Irony" | "Satire" | "None of these"],
  "stand_up": "Yes" | "No",
  "humor_type": "Regular Humor" | "Dark Humor" | "None of these",
  "target_category": ["Gender / Sex related" | "Mental Health" | "Disability" | "Race / Ethnicity" | "Violence / Death" | "Crime" | "Other Sensitive Target"],
  "dark_intensity": "1 - Mild" | "2 - Moderate" | "3 - Severe" | null,
  "note": "short optional note"
}

Rules:
- joke_topic and rhetorical_device are required lists.
- If humor_presence is "Not Humor", use:
  - joke_topic: []
  - rhetorical_device: ["None of these"]
  - humor_type: "None of these"
  - target_category: []
  - dark_intensity: null
- If humor_type is not "Dark Humor", set target_category to [] and dark_intensity to null.
- If rhetorical_device has "Irony" or "Satire", do not include "None of these".
- note can be empty string if no extra note.
"""

ALLOWED_VALUES = {
    "humor_presence": ["Humor", "Not Humor", "Almost or not sure"],
    "joke_topic": [
        "Relationships",
        "Family",
        "Work",
        "Money",
        "School",
        "Politics and society",
        "Crime and law enforcement",
        "War and conflict",
        "Celebrity and pop culture",
        "Other",
    ],
    "rhetorical_device": ["Irony", "Satire", "None of these"],
    "stand_up": ["Yes", "No"],
    "humor_type": ["Regular Humor", "Dark Humor", "None of these"],
    "target_category": [
        "Gender / Sex related",
        "Mental Health",
        "Disability",
        "Race / Ethnicity",
        "Violence / Death",
        "Crime",
        "Other Sensitive Target",
    ],
    "dark_intensity": ["1 - Mild", "2 - Moderate", "3 - Severe"],
}


MODEL_LIST = [
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-R1-0528-tput",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Ministral-3-14B-Instruct-2512",
    "mistralai/Mistral-Small-24B-Instruct-2501",
    "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gpt-5-mini-2025-08-07",
    "gpt-4.1-mini-2025-04-14",
    "gpt-3.5-turbo-0125",
    "gpt-4o-2024-08-06",
    "o4-mini-2025-04-16",
    "o3-mini",
    "Qwen/Qwen2.5-7B-Instruct-Turbo",
    "Qwen/Qwen3-Next-80B-A3B-Thinking",
    "claude-3-haiku-20240307",
    "claude-3-5-haiku-20241022",
    "claude-haiku-4-5-20251001",
]


def load_model_and_tokenizer(model_path: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def generate_output(model, tokenizer, input_ids, max_new_tokens: int):
    gen_config = model.generation_config
    gen_config.max_new_tokens = max_new_tokens
    gen_config.pad_token_id = tokenizer.pad_token_id

    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids, device=model.device)

    output_ids = model.generate(input_ids.unsqueeze(0), generation_config=gen_config)[0]
    return output_ids[len(input_ids) :]


def complete_conversation(model, tokenizer, system_prompt: str, user_prompt: str, size=1024):
    message = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )

    output_ids = generate_output(model, tokenizer, message["input_ids"], size)
    output_str = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return output_str


def get_answer_openai(client: OpenAI, model_name: str, system_prompt: str, question: str) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def get_answer_claude(client, model_name: str, system_prompt: str, question: str) -> str:
    response = client.messages.create(
        model=model_name,
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": question}],
    )
    return response.content[0].text


def get_answer_gemini(client, model_name: str, system_prompt: str, question: str) -> str:
    response = client.models.generate_content(
        model=model_name,
        config=genai.types.GenerateContentConfig(system_instruction=system_prompt),
        contents=question,
    )
    return response.text


def get_answer_together(client, model_name: str, system_prompt: str, question: str) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=0,
    )
    return response.choices[0].message.content


def iter_jsonl_records(dataset_path: str):
    if os.path.isdir(dataset_path):
        for filename in os.listdir(dataset_path):
            if not filename.lower().endswith(".jsonl"):
                continue
            file_path = os.path.join(dataset_path, filename)
            with open(file_path, "r", encoding="utf8") as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    yield obj, f"{filename}__{idx}"
    else:
        with open(dataset_path, "r", encoding="utf8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                yield obj, idx


def get_content_text(obj: Dict[str, Any]) -> str:
    for key in ["text", "transcript", "content", "caption", "question"]:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def build_annotation_prompt(obj: Dict[str, Any]) -> str:
    text = get_content_text(obj)
    metadata_bits = []
    for key in ["qid", "id", "title", "lang", "source"]:
        if key in obj and obj[key] not in [None, ""]:
            metadata_bits.append(f"{key}: {obj[key]}")

    metadata_block = "\n".join(metadata_bits)
    return (
        "Annotate the following content using the target schema.\n\n"
        f"Metadata:\n{metadata_block if metadata_block else 'None'}\n\n"
        f"Content:\n{text}"
    )


def extract_json_block(raw_output: str) -> Optional[str]:
    if not raw_output:
        return None

    stripped = raw_output.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    match = re.search(r"\{[\s\S]*\}", stripped)
    return match.group(0) if match else None


def sanitize_list(values: Any, allowed: List[str]) -> List[str]:
    if not isinstance(values, list):
        return []
    seen = set()
    output = []
    for value in values:
        if value in allowed and value not in seen:
            seen.add(value)
            output.append(value)
    return output


def normalize_annotation(parsed: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    issues = []

    humor_presence = parsed.get("humor_presence")
    if humor_presence not in ALLOWED_VALUES["humor_presence"]:
        humor_presence = "Almost or not sure"
        issues.append("invalid humor_presence -> defaulted")

    joke_topic = sanitize_list(parsed.get("joke_topic", []), ALLOWED_VALUES["joke_topic"])
    rhetorical_device = sanitize_list(
        parsed.get("rhetorical_device", []), ALLOWED_VALUES["rhetorical_device"]
    )

    if any(x in rhetorical_device for x in ["Irony", "Satire"]):
        rhetorical_device = [x for x in rhetorical_device if x != "None of these"]
    if not rhetorical_device:
        rhetorical_device = ["None of these"]

    stand_up = parsed.get("stand_up")
    if stand_up not in ALLOWED_VALUES["stand_up"]:
        stand_up = "No"
        issues.append("invalid stand_up -> defaulted")

    humor_type = parsed.get("humor_type")
    if humor_type not in ALLOWED_VALUES["humor_type"]:
        humor_type = "None of these"
        issues.append("invalid humor_type -> defaulted")

    target_category = sanitize_list(
        parsed.get("target_category", []), ALLOWED_VALUES["target_category"]
    )

    dark_intensity = parsed.get("dark_intensity")
    if dark_intensity not in ALLOWED_VALUES["dark_intensity"]:
        dark_intensity = None

    if humor_presence == "Not Humor":
        joke_topic = []
        rhetorical_device = ["None of these"]
        humor_type = "None of these"
        target_category = []
        dark_intensity = None

    if humor_type != "Dark Humor":
        target_category = []
        dark_intensity = None

    note = parsed.get("note", "")
    if not isinstance(note, str):
        note = str(note)

    normalized = {
        "humor_presence": humor_presence,
        "joke_topic": joke_topic,
        "rhetorical_device": rhetorical_device,
        "stand_up": stand_up,
        "humor_type": humor_type,
        "target_category": target_category,
        "dark_intensity": dark_intensity,
        "note": note,
    }
    return normalized, issues


def parse_annotation(raw_output: str) -> Tuple[Dict[str, Any], List[str]]:
    json_blob = extract_json_block(raw_output)
    if not json_blob:
        fallback, issues = normalize_annotation({})
        return fallback, ["json_not_found", *issues]

    try:
        parsed = json.loads(json_blob)
    except json.JSONDecodeError:
        fallback, issues = normalize_annotation({})
        return fallback, ["json_parse_error", *issues]

    normalized, issues = normalize_annotation(parsed)
    return normalized, issues


def make_client(model_name: str):
    model_name_lower = model_name.lower()
    if "gpt" in model_name_lower or "o4-" in model_name_lower or model_name_lower.startswith("o3-"):
        return openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if "claude" in model_name_lower:
        return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    if "gemini" in model_name_lower:
        return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    if "/" in model_name_lower:
        return Together(api_key=os.getenv("TOGETHER_API_KEY"))
    raise ValueError(f"Cannot find API for {model_name}")


def online_llm_annotate_jsonl(client, dataset_path: str, model_name: str, output_path: str, verbose=True):
    model_name_lower = model_name.lower()
    rows = []

    for obj, idx in iter_jsonl_records(dataset_path):
        prompt = build_annotation_prompt(obj)

        if "gpt" in model_name_lower or "o4-" in model_name_lower or model_name_lower.startswith("o3-"):
            raw_answer = get_answer_openai(client, model_name, SYSTEM_PROMPT, prompt)
        elif "claude" in model_name_lower:
            raw_answer = get_answer_claude(client, model_name, SYSTEM_PROMPT, prompt)
        elif "gemini" in model_name_lower:
            raw_answer = get_answer_gemini(client, model_name, SYSTEM_PROMPT, prompt)
        else:
            raw_answer = get_answer_together(client, model_name, SYSTEM_PROMPT, prompt)

        annotation, parse_issues = parse_annotation(raw_answer)
        qid = obj.get("qid", obj.get("id", idx))

        row = {
            "QID": qid,
            "RawResponse": raw_answer,
            "ParseIssues": "|".join(parse_issues),
            **annotation,
        }
        rows.append(row)

        if verbose:
            print(f"QID={qid}")
            print(f"Prompt: {repr(prompt[:300])}...")
            print(f"Raw: {repr(raw_answer[:300])}...")
            print(f"Annotation: {annotation}")
            print(f"Issues: {parse_issues}\n")

    write_outputs(rows, model_name_lower.split("/")[-1], output_path)


def local_llm_annotate_jsonl(model_path: str, dataset_path: str, output_path: str, verbose=True):
    model, tokenizer = load_model_and_tokenizer(model_path)
    rows = []

    for obj, idx in iter_jsonl_records(dataset_path):
        prompt = build_annotation_prompt(obj)
        raw_answer = complete_conversation(model, tokenizer, SYSTEM_PROMPT, prompt)

        annotation, parse_issues = parse_annotation(raw_answer)
        qid = obj.get("qid", obj.get("id", idx))

        row = {
            "QID": qid,
            "RawResponse": raw_answer,
            "ParseIssues": "|".join(parse_issues),
            **annotation,
        }
        rows.append(row)

        if verbose:
            print(f"QID={qid} | Issues={parse_issues}")

    write_outputs(rows, os.path.basename(model_path), output_path)


def write_outputs(rows: List[Dict[str, Any]], model_tag: str, output_path: str):
    os.makedirs(output_path, exist_ok=True)

    jsonl_path = os.path.join(output_path, f"{model_tag}_annotations.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    import pandas as pd

    csv_path = os.path.join(output_path, f"{model_tag}_annotations.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", default=None)
    parser.add_argument("--online_only", default=False, action="store_true")
    parser.add_argument("--local_only", default=False, action="store_true")
    parser.add_argument(
        "--dataset_path",
        default="./questions.jsonl",
        help="JSONL file or directory containing JSONL files with text/transcript/content fields.",
    )
    parser.add_argument(
        "--output_path",
        default="./outputs",
        help="Directory to save model annotation outputs.",
    )
    args = parser.parse_args()

    dataset_path = args.dataset_path
    models_path = "./models"
    output_path = args.output_path

    model_list = MODEL_LIST
    if args.model_name is not None:
        model_list = [args.model_name]

    if not args.online_only:
        if not os.path.isdir(models_path):
            print(f"[!] Local models directory not found: {models_path}")
        else:
            for model_name in os.listdir(models_path):
                if model_name == args.model_name or args.model_name is None:
                    model_path = os.path.join(models_path, model_name)
                    print(f"Loading local model {model_name}")
                    local_llm_annotate_jsonl(model_path, dataset_path, output_path)

    if not args.local_only:
        for model_name in model_list:
            print(f"Loading online model {model_name}")
            client = make_client(model_name)
            online_llm_annotate_jsonl(client, dataset_path, model_name, output_path)


if __name__ == "__main__":
    main()
