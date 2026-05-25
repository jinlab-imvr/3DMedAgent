from __future__ import annotations

import asyncio
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from openai import AsyncOpenAI


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
CT_CLIP_DIR = PROJECT_ROOT / "CT_Clip"
for path in (CURRENT_DIR, PROJECT_ROOT, CT_CLIP_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import (  # noqa: E402
    BASE_BACKOFF,
    MAX_CONCURRENCY,
    MAX_RETRIES,
    MODEL_NAME,
    OPENAI_API_KEY,
    SYSTEM_PROMPT,
    TARGET_SUBTYPES,
)
from Tool_Box.io import safe_json_loads  # noqa: E402


aclient = AsyncOpenAI(api_key=OPENAI_API_KEY)


async def call_gpt(prompt: str, model: str = MODEL_NAME) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            response = await aclient.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content
        except Exception:
            if attempt == MAX_RETRIES - 1:
                raise
            await asyncio.sleep(BASE_BACKOFF * (2**attempt) + random.random())
    raise RuntimeError("unreachable")


def parse_target_subtypes(value: str) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def select_target_rows(
    vqa_path: str,
    max_per_subtype: int,
    target_subtypes: List[str],
    question_type: str = "",
) -> pd.DataFrame:
    df = pd.read_csv(vqa_path).reset_index(names="case_idx")
    if question_type:
        df = df[df["question type"].astype(str) == question_type].copy()
    if target_subtypes:
        df = df[df["question subtype"].isin(target_subtypes)].copy()
    if max_per_subtype > 0:
        df = df.groupby("question subtype", group_keys=False).head(max_per_subtype)
    return df.reset_index(drop=True)


def read_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_existing_records(save_path: str) -> List[Dict[str, Any]]:
    data = read_json(save_path)
    return data if isinstance(data, list) else []


def upsert_record(records: List[Dict[str, Any]], new_record: Dict[str, Any]) -> List[Dict[str, Any]]:
    key = (new_record["case_idx"], new_record["question_subtype"])
    records = [
        record
        for record in records
        if (record.get("case_idx"), record.get("question_subtype")) != key
    ]
    records.append(new_record)
    return sorted(records, key=lambda item: (int(item.get("case_idx", 10**9)), item.get("question_subtype", "")))


async def save_record(save_dir: str, record: Dict[str, Any], file_lock: asyncio.Lock) -> None:
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{record['image_id']}.json")
    async with file_lock:
        records = upsert_record(load_existing_records(save_path), record)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=4)


def normalized_final_answer(value: Any) -> str:
    text = str(value or "").strip().upper()
    if re.fullmatch(r"[A-Z]", text):
        return text
    match = re.search(r"\b([A-Z])\b", text)
    return match.group(1) if match else text[:1]


def parse_answer_response(raw_response: str) -> Dict[str, Any]:
    parsed = safe_json_loads(raw_response)
    if isinstance(parsed, dict):
        return parsed

    match = re.search(r'"final_answer"\s*:\s*"?([A-E])"?', raw_response, flags=re.IGNORECASE)
    return {"final_answer": match.group(1).upper()} if match else {}


def make_base_record(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "case_idx": int(row["case_idx"]),
        "image_id": row["Image ID"],
        "dataset": row.get("dataset", ""),
        "question_type": row.get("question type", ""),
        "question_subtype": row.get("question subtype", ""),
        "question": row.get("multiple-choice question", ""),
        "gt_answer": row.get("correct option", ""),
        "facts_memory": {},
        "debug_memory": {},
        "reasoning_memory": {},
        "memory_schema_warnings": [],
        "tool_registry": [],
        "tool_candidates": [],
        "tool_selection": {},
        "runtime_tool_included": False,
        "GPT_raw_result": "",
        "GPT_summarized_result": "",
        "evidence_ids_used": [],
        "slice_candidates_relevant": [],
        "uncertainty": "",
        "assumptions": [],
    }
