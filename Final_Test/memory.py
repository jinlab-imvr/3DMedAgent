#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Leak-safe Final_Test memory utilities.

This module builds a text-only evidence memory for 3DMedAgent-style
experiments. It deliberately separates internal audit fields from the memory
rendered to GPT so CSV organ / lesion / answer labels are not used as evidence.

The prompt memory keeps lightweight all-organ and all-global-lesion context
while selecting heavier CT-CLIP detail evidence by the inferred query.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from slice_coordinates import (
    map_vqa_percent_to_ctclip_slice,
    section_index_from_ctclip_slice,
)


ALLOWED_ORGANS = ("liver", "kidney", "colon", "pancreas", "spleen")
ALLOWED_LESION_TYPES = ("lesion", "tumor", "cyst")
TOTAL_SLICES = 240

SOURCE_PRIORITY = {
    "mcq_option_slice": 0,
    "lesion_top_slice": 1,
    "lesion_top_section_midpoint": 2,
    "runtime_component": 3,
    "report_lesion_slice": 4,
    "organ_max_area": 5,
    "organ_center": 6,
    "organ_uniform": 7,
    "fallback": 8,
}

LEAKY_RUNTIME_KEYS = {
    "correct_option",
    "correct_count",
    "count_diff",
    "answer_value",
    "hu_diff_signed_error",
    "hu_diff_abs_error",
    "is_correct",
    "evaluable_mcq",
    "is_evaluable_mcq",
}


@dataclass
class MemorySourcePaths:
    clip_global_dir: str
    clip_detail_dir: str
    clip_detail_slice_dir: str
    mask_dir: str
    region_memory_dir: Optional[str] = None


def row_get(row: Any, key: str, default: Any = "") -> Any:
    if isinstance(row, dict):
        value = row.get(key, default)
    else:
        try:
            value = row[key]
        except Exception:
            value = default
    try:
        if value != value:
            return default
    except Exception:
        pass
    return default if value is None else value


def read_json(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_organ_choice(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower()
    text = text.replace("_", " ")
    mapping = {
        "hepatic": "liver",
        "renal": "kidney",
        "left kidney": "kidney",
        "right kidney": "kidney",
        "pancreatic": "pancreas",
        "colonic": "colon",
        "bowel": "colon",
        "intestine": "colon",
        "splenic": "spleen",
    }
    text = mapping.get(text, text)
    return text if text in ALLOWED_ORGANS else None


def normalize_lesion_choice(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower()
    mapping = {
        "tumour": "tumor",
        "tumors": "tumor",
        "tumours": "tumor",
        "mass": "tumor",
        "masses": "tumor",
        "cysts": "cyst",
        "lesions": "lesion",
        "abnormality": "lesion",
        "abnormalities": "lesion",
    }
    text = mapping.get(text, text)
    return text if text in ALLOWED_LESION_TYPES else None


def normalize_lesion_terms(text: str) -> str:
    text = re.sub(r"\btumors?\b", "lesion", str(text), flags=re.IGNORECASE)
    text = re.sub(r"\bcysts?\b", "lesion", text, flags=re.IGNORECASE)
    text = re.sub(r"\blesions\b", "lesion", text, flags=re.IGNORECASE)
    return text


ORGAN_PATTERNS = (
    ("liver", r"\b(liver|hepatic)\b"),
    ("kidney", r"\b(kidney|kidneys|renal|left kidney|right kidney)\b"),
    ("colon", r"\b(colon|colonic|bowel|intestine)\b"),
    ("pancreas", r"\b(pancreas|pancreatic)\b"),
    ("spleen", r"\b(spleen|splenic)\b"),
)


def split_question_stem_and_options(question: Any) -> Tuple[str, Dict[str, str]]:
    text = str(question or "")
    matches = list(re.finditer(r"\b([A-Z]):\s*", text))
    if not matches:
        return text, {}
    stem = text[: matches[0].start()].strip()
    options: Dict[str, str] = {}
    for idx, match in enumerate(matches):
        letter = match.group(1)
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        options[letter] = text[start:end].strip()
    return stem, options


def find_organs_in_text(text: Any) -> List[str]:
    found: List[Tuple[int, int, str]] = []
    lowered = str(text or "").lower()
    for organ, pattern in ORGAN_PATTERNS:
        for match in re.finditer(pattern, lowered):
            found.append((match.start(), match.end(), organ))
    found.sort(key=lambda item: (item[0], item[1]))
    organs: List[str] = []
    for _, _, organ in found:
        if organ not in organs:
            organs.append(organ)
    return organs


def lesion_types_from_text(text: Any) -> List[str]:
    q = str(text or "").lower()
    lesion_types: List[str] = []
    if re.search(r"\b(tumou?r|tumou?rs|mass|masses|malignant)\b", q):
        lesion_types.append("tumor")
    if re.search(r"\b(cyst|cysts)\b", q):
        lesion_types.append("cyst")
    if re.search(r"\b(lesion|lesions|abnormality|abnormalities|finding|findings)\b", q):
        lesion_types.append("lesion")
    return lesion_types


def _ordered_unique(values: Iterable[Any], allowed: Iterable[str]) -> List[str]:
    allowed_set = set(allowed)
    output: List[str] = []
    for value in values:
        text = str(value)
        if text in allowed_set and text not in output:
            output.append(text)
    return output


def _has_lesion_detail_need(question: str, intent: str, lesion_types: List[str]) -> bool:
    q = str(question or "").lower()
    if intent in {
        "localize_lesion_slice",
        "count_lesions",
        "compare_hu_or_attenuation",
        "determine_lesion_existence",
    }:
        return True
    explicit_lesion_terms = bool(
        re.search(r"\b(lesion|lesions|tumou?r|tumou?rs|mass|masses|cyst|cysts|abnormality|abnormalities)\b", q)
    )
    return explicit_lesion_terms or bool(re.search(r"\b(largest|biggest|dominant)\s+(lesion|tumou?r|mass|cyst)\b", q))


def make_detail_targets(target_organ: str, lesion_type: str, lesion_types_to_score: List[str], question: str, intent: str) -> List[Dict[str, str]]:
    if not _has_lesion_detail_need(question, intent, lesion_types_to_score):
        return []
    lesion_types = _ordered_unique([lesion_type, *lesion_types_to_score], ALLOWED_LESION_TYPES)
    return [
        {"organ": target_organ, "lesion_type": item}
        for item in lesion_types[:3]
    ]


def infer_query_from_question_rule(question: str) -> Dict[str, Any]:
    """Deterministic fallback that uses only the question text."""
    stem, options = split_question_stem_and_options(question)
    q = str(question or "").lower()
    stem_organs = find_organs_in_text(stem)
    option_organs = _ordered_unique(
        [
            organ
            for option_text in options.values()
            for organ in find_organs_in_text(option_text)
        ],
        ALLOWED_ORGANS,
    )
    full_organs = find_organs_in_text(question)
    target_organ = stem_organs[0] if stem_organs else (full_organs[0] if full_organs else None)

    stem_lesions = lesion_types_from_text(stem)
    full_lesions = lesion_types_from_text(question)
    if "tumor" in full_lesions and "cyst" in full_lesions and re.search(r"\b(versus|vs\.?|type|classification|classify)\b", q):
        lesion_type = "lesion"
        lesion_types_to_score = ["lesion", "tumor", "cyst"]
    elif "cyst" in stem_lesions or ("cyst" in full_lesions and "tumor" not in full_lesions):
        lesion_type = "cyst"
        lesion_types_to_score = ["cyst"]
    elif "tumor" in stem_lesions or ("tumor" in full_lesions and "cyst" not in full_lesions):
        lesion_type = "tumor"
        lesion_types_to_score = ["tumor"]
    else:
        lesion_type = "lesion"
        lesion_types_to_score = ["lesion"]

    if "slice" in q or "%" in q:
        intent = "localize_lesion_slice"
    elif "how many" in q or "count" in q or "number of" in q:
        intent = "count_lesions"
    elif "hu" in q or "attenuation" in q:
        intent = "compare_hu_or_attenuation"
    elif "exist" in q or "present" in q or "presence" in q:
        intent = "determine_lesion_existence"
    else:
        intent = "answer_multiple_choice"
    target_organ = target_organ or "liver"
    target_organs = _ordered_unique([target_organ, *stem_organs, *option_organs], ALLOWED_ORGANS)
    detail_targets = make_detail_targets(target_organ, lesion_type, lesion_types_to_score, question, intent)

    return {
        "target_organ": target_organ,
        "primary_target_organ": target_organ,
        "target_organs": target_organs,
        "stem_organs": stem_organs,
        "option_organs": option_organs,
        "lesion_type": lesion_type,
        "lesion_types_to_score": lesion_types_to_score,
        "detail_targets": detail_targets,
        "question_intent": intent,
        "confidence": 0.95 if stem_organs or full_organs else 0.25,
        "rationale": (
            "Parsed primary organ from the question stem and tracked option organs separately."
            if stem_organs
            else "Parsed organ and lesion type from the question text only."
            if full_organs
            else "No canonical organ was explicit; defaulted to liver for schema safety."
        ),
        "normalizer": "rule_fallback",
    }


def coerce_inferred_query(raw: Any, question: str) -> Dict[str, Any]:
    fallback = infer_query_from_question_rule(question)
    raw = raw if isinstance(raw, dict) else {}
    target_organ = normalize_organ_choice(raw.get("target_organ")) or fallback["target_organ"]
    lesion_type = normalize_lesion_choice(raw.get("lesion_type")) or fallback["lesion_type"]
    primary_target_organ = normalize_organ_choice(raw.get("primary_target_organ")) or target_organ
    target_organs = _ordered_unique(
        [
            primary_target_organ,
            *[normalize_organ_choice(item) or "" for item in raw.get("target_organs", []) if isinstance(raw.get("target_organs"), list)],
            *fallback.get("target_organs", []),
        ],
        ALLOWED_ORGANS,
    )
    stem_organs = _ordered_unique(
        [normalize_organ_choice(item) or "" for item in raw.get("stem_organs", [])]
        if isinstance(raw.get("stem_organs"), list)
        else fallback.get("stem_organs", []),
        ALLOWED_ORGANS,
    )
    option_organs = _ordered_unique(
        [normalize_organ_choice(item) or "" for item in raw.get("option_organs", [])]
        if isinstance(raw.get("option_organs"), list)
        else fallback.get("option_organs", []),
        ALLOWED_ORGANS,
    )
    lesion_types_to_score = _ordered_unique(
        [lesion_type, *[
            normalize_lesion_choice(item) or ""
            for item in raw.get("lesion_types_to_score", [])
        ]]
        if isinstance(raw.get("lesion_types_to_score"), list)
        else [lesion_type, *fallback.get("lesion_types_to_score", [])],
        ALLOWED_LESION_TYPES,
    )
    confidence = raw.get("confidence", fallback["confidence"])
    try:
        confidence = float(confidence)
    except Exception:
        confidence = float(fallback["confidence"])
    confidence = max(0.0, min(1.0, confidence))
    intent = str(raw.get("question_intent") or fallback["question_intent"]).strip()
    rationale = str(raw.get("rationale") or fallback["rationale"]).strip()
    normalizer = str(raw.get("normalizer") or raw.get("source") or "gpt").strip()
    raw_detail_targets = raw.get("detail_targets") if isinstance(raw.get("detail_targets"), list) else []
    detail_targets: List[Dict[str, str]] = []
    for item in raw_detail_targets:
        if not isinstance(item, dict):
            continue
        organ = normalize_organ_choice(item.get("organ"))
        lesion = normalize_lesion_choice(item.get("lesion_type"))
        if organ and lesion and {"organ": organ, "lesion_type": lesion} not in detail_targets:
            detail_targets.append({"organ": organ, "lesion_type": lesion})
    if not detail_targets:
        detail_targets = make_detail_targets(primary_target_organ, lesion_type, lesion_types_to_score, question, intent)
    return {
        "target_organ": target_organ,
        "primary_target_organ": primary_target_organ,
        "target_organs": target_organs or [target_organ],
        "stem_organs": stem_organs,
        "option_organs": option_organs,
        "lesion_type": lesion_type,
        "lesion_types_to_score": lesion_types_to_score or [lesion_type],
        "detail_targets": detail_targets,
        "question_intent": intent,
        "confidence": confidence,
        "rationale": rationale,
        "normalizer": normalizer,
        "allowed_organs": list(ALLOWED_ORGANS),
        "allowed_lesion_types": list(ALLOWED_LESION_TYPES),
    }


def build_query_normalization_prompt(question: str) -> str:
    return f"""Infer leak-safe query targets from the multiple-choice question only.

Allowed target_organ values: {list(ALLOWED_ORGANS)}
Allowed lesion_type values: {list(ALLOWED_LESION_TYPES)}

Rules:
- Do not use any hidden dataset labels.
- Choose one primary target_organ from the question stem when possible.
- Track option organs separately; do not let an option organ override the primary target organ.
- target_organs can include multiple allowed organs when the question mentions or compares several organs.
- If the question says tumor/mass/malignant, choose tumor.
- If the question says cyst, choose cyst.
- If the question only says lesion/abnormality/finding, choose lesion.
- If the question compares cyst versus tumor, set lesion_type to lesion and include tumor/cyst in lesion_types_to_score.

Output STRICT JSON:
{{
  "target_organ": "<one allowed organ>",
  "primary_target_organ": "<one allowed organ>",
  "target_organs": ["<allowed organ>", "..."],
  "stem_organs": ["<allowed organ>", "..."],
  "option_organs": ["<allowed organ>", "..."],
  "lesion_type": "<one allowed lesion type>",
  "lesion_types_to_score": ["<allowed lesion type>", "..."],
  "detail_targets": [{{"organ": "<allowed organ>", "lesion_type": "<allowed lesion type>"}}],
  "question_intent": "<short intent>",
  "confidence": <0.0-1.0>,
  "rationale": "<brief text>"
}}

Question:
{question}
"""


def parse_numeric_options(mcq: Any) -> Dict[str, float]:
    matches = re.findall(r"\b([A-Z]):\s*([-+]?\d+(?:\.\d+)?)", str(mcq or ""))
    return {letter: float(value) for letter, value in matches}


def parse_option_texts(mcq: Any) -> Dict[str, str]:
    text = str(mcq or "")
    matches = list(re.finditer(r"\b([A-Z]):\s*", text))
    options: Dict[str, str] = {}
    for idx, match in enumerate(matches):
        letter = match.group(1)
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        options[letter] = text[start:end].strip()
    return options


def is_slice_question(question: Any) -> bool:
    q = str(question or "").lower()
    return "slice" in q or "%" in q or "percent" in q


def slice_index_from_percent(percent: float) -> int:
    return int(max(0, min(TOTAL_SLICES - 1, round(float(percent) / 100.0 * (TOTAL_SLICES - 1)))))


def section_index_from_percent(percent: float) -> int:
    return int(max(0, min(23, math.floor(float(percent) / 100.0 * 24))))


def z_percent(slice_index_240: int) -> float:
    return round(100.0 * int(slice_index_240) / (TOTAL_SLICES - 1), 4)


def clamp_slice_index(value: Any) -> int:
    try:
        idx = int(round(float(value)))
    except Exception:
        idx = 0
    return max(0, min(TOTAL_SLICES - 1, idx))


def make_candidate(
    slice_index_240: Any,
    source_type: str,
    target: str,
    score: Optional[float],
    description: str,
    evidence_ids: Iterable[str],
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    idx = clamp_slice_index(slice_index_240)
    candidate = {
        "slice_index_240": idx,
        "z_percent": z_percent(idx),
        "source_type": source_type,
        "target": target,
        "score": None if score is None else float(score),
        "support_scores": [
            {
                "source_type": source_type,
                "score": None if score is None else float(score),
            }
        ],
        "description": str(description),
        "evidence_ids": sorted({str(eid) for eid in evidence_ids if eid}),
        "_priority": SOURCE_PRIORITY.get(source_type, 99),
    }
    if extra_fields:
        candidate.update({str(k): v for k, v in extra_fields.items()})
    return candidate


def merge_and_limit_candidates(
    candidates: Iterable[Dict[str, Any]],
    max_items: int = 10,
    merge_distance: int = 8,
) -> List[Dict[str, Any]]:
    candidate_list = [c for c in candidates if isinstance(c, dict)]
    option_candidates = [
        dict(c)
        for c in candidate_list
        if c.get("source_type") == "mcq_option_slice"
    ]
    if option_candidates:
        option_candidates = sorted(
            option_candidates,
            key=lambda item: (
                str(item.get("option_id", "")),
                int(item.get("slice_index_240", 0)),
            ),
        )
        seen_options = set()
        preserved_options = []
        for item in option_candidates:
            option_key = (item.get("option_id"), int(item.get("slice_index_240", 0)))
            if option_key in seen_options:
                continue
            seen_options.add(option_key)
            item.pop("_priority", None)
            item["merged_source_types"] = [item.get("source_type")]
            preserved_options.append(item)
        if len(preserved_options) >= max_items:
            return preserved_options[:max_items]
        non_option_candidates = [
            c for c in candidate_list if c.get("source_type") != "mcq_option_slice"
        ]
        return [
            *preserved_options,
            *merge_and_limit_candidates(
                non_option_candidates,
                max_items=max_items - len(preserved_options),
                merge_distance=merge_distance,
            ),
        ][:max_items]

    ordered = sorted(
        candidate_list,
        key=lambda item: (
            int(item.get("_priority", SOURCE_PRIORITY.get(item.get("source_type"), 99))),
            int(item.get("slice_index_240", 0)),
        ),
    )
    merged: List[Dict[str, Any]] = []
    for candidate in ordered:
        idx = int(candidate["slice_index_240"])
        target_existing = None
        for existing in merged:
            if abs(int(existing["slice_index_240"]) - idx) <= merge_distance:
                target_existing = existing
                break
        if target_existing is None:
            candidate = dict(candidate)
            candidate["merged_source_types"] = [candidate["source_type"]]
            merged.append(candidate)
            continue

        if candidate["source_type"] not in target_existing["merged_source_types"]:
            target_existing["merged_source_types"].append(candidate["source_type"])
        target_existing["support_scores"].extend(candidate.get("support_scores") or [])
        for field, merged_field in (
            ("coordinate_transform_status", "coordinate_transform_statuses"),
            ("coordinate_frame", "coordinate_frames"),
            ("raw_slice_index", "raw_slice_indices"),
            ("source_depth", "source_depths"),
        ):
            values = []
            if target_existing.get(field) is not None:
                values.append(target_existing.get(field))
            values.extend(target_existing.get(merged_field) or [])
            if candidate.get(field) is not None:
                values.append(candidate.get(field))
            values.extend(candidate.get(merged_field) or [])
            if values:
                deduped = []
                for value in values:
                    if value not in deduped:
                        deduped.append(value)
                target_existing[merged_field] = deduped
        if candidate.get("is_approximate_coordinate"):
            target_existing["has_approximate_coordinate_evidence"] = True
        if candidate.get("description") and candidate["description"] not in target_existing["description"]:
            target_existing["description"] = (
                target_existing["description"].rstrip(".")
                + ". Also: "
                + candidate["description"].rstrip(".")
                + "."
            )
        target_existing["evidence_ids"] = sorted(
            set(target_existing.get("evidence_ids") or []) | set(candidate.get("evidence_ids") or [])
        )
        if candidate["source_type"] == target_existing["source_type"] and candidate.get("score") is not None:
            old_score = target_existing.get("score")
            target_existing["score"] = (
                candidate["score"] if old_score is None else max(float(old_score), float(candidate["score"]))
            )
        elif target_existing.get("score") is None and candidate.get("score") is not None:
            target_existing["score"] = candidate["score"]

    final = sorted(
        merged,
        key=lambda item: (
            int(item.get("_priority", SOURCE_PRIORITY.get(item.get("source_type"), 99))),
            int(item.get("slice_index_240", 0)),
        ),
    )[:max_items]
    for item in final:
        item.pop("_priority", None)
        if item.get("coordinate_transform_status") and not item.get("coordinate_transform_statuses"):
            item["coordinate_transform_statuses"] = [item["coordinate_transform_status"]]
        if item.get("coordinate_frame") and not item.get("coordinate_frames"):
            item["coordinate_frames"] = [item["coordinate_frame"]]
    return final


def report_heading_for_organ(organ: str) -> str:
    return {
        "liver": "Liver",
        "kidney": "Kidney",
        "colon": "Colon",
        "pancreas": "Pancreas",
        "spleen": "Spleen",
    }.get(organ, organ.capitalize())


def extract_report_block(report: str, organ: str) -> str:
    heading = report_heading_for_organ(organ)
    headings = "|".join(report_heading_for_organ(item) for item in ALLOWED_ORGANS)
    pattern = rf"(?:^|\n){re.escape(heading)}:\s*(.*?)(?=\n(?:{headings}):|\Z)"
    match = re.search(pattern, str(report or ""), flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def parse_report_metrics(report: str, organ: str) -> Dict[str, Any]:
    block = extract_report_block(report, organ)
    volume_match = re.search(r"(?:total\s+\w+\s+volume|volume):\s*([-+]?\d+(?:\.\d+)?)\s*cm", block, re.I)
    mean_hu_match = re.search(r"Mean HU value:\s*([-+]?\d+(?:\.\d+)?)\s*\+/-\s*([-+]?\d+(?:\.\d+)?)", block, re.I)
    right_kidney = re.search(r"right kidney volume:\s*([-+]?\d+(?:\.\d+)?)\s*cm", block, re.I)
    left_kidney = re.search(r"left kidney volume:\s*([-+]?\d+(?:\.\d+)?)\s*cm", block, re.I)
    status_line = ""
    for line in block.splitlines():
        stripped = line.strip()
        if stripped:
            status_line = stripped
            break
    return {
        "report_block": block,
        "status_text": status_line,
        "volume_cm3": float(volume_match.group(1)) if volume_match else None,
        "mean_hu": float(mean_hu_match.group(1)) if mean_hu_match else None,
        "std_hu": float(mean_hu_match.group(2)) if mean_hu_match else None,
        "right_kidney_volume_cm3": float(right_kidney.group(1)) if right_kidney else None,
        "left_kidney_volume_cm3": float(left_kidney.group(1)) if left_kidney else None,
    }


def extract_report_lesion_slices(report: str, organ: str) -> List[int]:
    block = extract_report_block(report, organ)
    slices = [clamp_slice_index(value) for value in re.findall(r"\bslice\s+(\d+)\b", block, re.I)]
    return sorted(set(slices))


def load_ctclip_organ_mask(dataset: str, image_id: str, organ: str, mask_dir: str):
    try:
        from CT_Clip.clip_utils import load_organ_mask
    except Exception:
        return None
    try:
        return load_organ_mask(dataset, image_id, organ, mask_dir)
    except Exception:
        return None


def mask_to_numpy_dhw(mask: Any) -> Optional[np.ndarray]:
    if mask is None:
        return None
    try:
        if hasattr(mask, "detach"):
            arr = mask.detach().cpu().numpy()
        else:
            arr = np.asarray(mask)
    except Exception:
        return None
    if arr.ndim != 3:
        return None
    if arr.shape[0] != TOTAL_SLICES and arr.shape[-1] == TOTAL_SLICES:
        arr = np.moveaxis(arr, -1, 0)
    if arr.shape[0] != TOTAL_SLICES:
        return None
    return arr > 0


def summarize_mask(mask: np.ndarray) -> Dict[str, Any]:
    areas = mask.sum(axis=(1, 2))
    positive = np.flatnonzero(areas > 0)
    if positive.size == 0:
        return {"available": False}
    z_min = int(positive.min())
    z_max = int(positive.max())
    z_center = int(round((z_min + z_max) / 2.0))
    max_area_slice = int(areas.argmax())
    return {
        "available": True,
        "z_range_240": [z_min, z_max],
        "z_percent_range": [z_percent(z_min), z_percent(z_max)],
        "center_slice_240": z_center,
        "max_area_slice_240": max_area_slice,
        "max_area_voxels": int(areas[max_area_slice]),
        "nonzero_slice_count": int(positive.size),
    }


def uniform_slices_from_range(start: int, end: int, count: int) -> List[int]:
    if count <= 0:
        return []
    if start > end:
        start, end = end, start
    if start == end:
        return [start]
    return sorted({clamp_slice_index(v) for v in np.linspace(start, end, count)})


def find_region_memory(region_memory_dir: Optional[str], dataset: str, image_id: str) -> Optional[Dict[str, Any]]:
    if not region_memory_dir:
        return None
    candidates = [
        Path(region_memory_dir) / dataset / f"{image_id}_slice_memory.json",
        Path(region_memory_dir) / f"{image_id}_slice_memory.json",
    ]
    for path in candidates:
        if path.exists():
            return read_json(str(path))
    return None


def build_organ_memory_and_candidates(
    dataset: str,
    image_id: str,
    target_organ: str,
    report: str,
    paths: MemorySourcePaths,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[str]]:
    metrics = parse_report_metrics(report, target_organ)
    organ_memory: Dict[str, Any] = {
        "target_organ": target_organ,
        "report_metrics": metrics,
        "mask_summary": {},
        "suborgan_mask_summaries": {},
    }
    candidates: List[Dict[str, Any]] = []
    missing: List[str] = []

    mask = mask_to_numpy_dhw(load_ctclip_organ_mask(dataset, image_id, target_organ, paths.mask_dir))
    summary = summarize_mask(mask) if mask is not None else {"available": False}
    organ_memory["mask_summary"] = summary
    if summary.get("available"):
        z_min, z_max = summary["z_range_240"]
        candidates.append(make_candidate(
            summary["max_area_slice_240"],
            "organ_max_area",
            target_organ,
            float(summary["max_area_voxels"]),
            f"Representative {target_organ} slice with the largest organ cross-section.",
            [f"organ_mask:{target_organ}:max_area"],
        ))
        candidates.append(make_candidate(
            summary["center_slice_240"],
            "organ_center",
            target_organ,
            None,
            f"Central slice of the segmented {target_organ} z-range.",
            [f"organ_mask:{target_organ}:z_range"],
        ))
        for idx in uniform_slices_from_range(z_min, z_max, 3):
            candidates.append(make_candidate(
                idx,
                "organ_uniform",
                target_organ,
                None,
                f"Uniform coverage slice inside the segmented {target_organ} range.",
                [f"organ_mask:{target_organ}:uniform"],
            ))
    else:
        missing.append(f"missing_organ_mask:{target_organ}")

    if target_organ == "kidney":
        for suborgan in ("left kidney", "right kidney"):
            submask = mask_to_numpy_dhw(load_ctclip_organ_mask(dataset, image_id, suborgan, paths.mask_dir))
            subsummary = summarize_mask(submask) if submask is not None else {"available": False}
            organ_memory["suborgan_mask_summaries"][suborgan] = subsummary
            if subsummary.get("available"):
                candidates.append(make_candidate(
                    subsummary["max_area_slice_240"],
                    "organ_max_area",
                    suborgan,
                    float(subsummary["max_area_voxels"]),
                    f"Representative {suborgan} slice with the largest cross-section.",
                    [f"organ_mask:{suborgan}:max_area"],
                ))

    if not candidates:
        region_memory = find_region_memory(paths.region_memory_dir, dataset, image_id)
        region_slices = (region_memory or {}).get("region_slices") or {}
        ranges = []
        if target_organ == "kidney":
            for key in ("kidney", "left kidney", "right kidney"):
                if key in region_slices:
                    ranges.append((key, region_slices[key]))
        elif target_organ in region_slices:
            ranges.append((target_organ, region_slices[target_organ]))
        for key, value in ranges:
            if isinstance(value, (list, tuple)) and len(value) == 2:
                start, end = clamp_slice_index(value[0]), clamp_slice_index(value[1])
                for idx in uniform_slices_from_range(start, end, 3):
                    candidates.append(make_candidate(
                        idx,
                        "organ_uniform",
                        key,
                        None,
                        f"Uniform slice from fallback region_slices for {key}.",
                        [f"region_slices:{key}"],
                    ))
        if not candidates:
            for idx in (60, 120, 180):
                candidates.append(make_candidate(
                    idx,
                    "fallback",
                    target_organ,
                    None,
                    "Whole-volume fallback slice because organ-level slice evidence was unavailable.",
                    ["fallback:whole_volume_uniform"],
                ))

    return organ_memory, candidates, missing


def organ_role(organ: str, primary_target_organ: str, target_organs: List[str], option_organs: List[str]) -> str:
    if organ == primary_target_organ:
        return "primary_target"
    if organ in option_organs:
        return "option_organ"
    if organ in target_organs:
        return "mentioned_target"
    return "background"


def build_all_organ_memory(
    dataset: str,
    image_id: str,
    report: str,
    paths: MemorySourcePaths,
    primary_target_organ: str,
    target_organs: List[str],
    option_organs: List[str],
) -> Dict[str, Any]:
    all_memory: Dict[str, Any] = {}
    for organ in ALLOWED_ORGANS:
        mask = mask_to_numpy_dhw(load_ctclip_organ_mask(dataset, image_id, organ, paths.mask_dir))
        all_memory[organ] = {
            "role": organ_role(organ, primary_target_organ, target_organs, option_organs),
            "report_metrics": parse_report_metrics(report, organ),
            "mask_summary": summarize_mask(mask) if mask is not None else {"available": False},
        }
    return all_memory


def get_clip_global_memory(
    image_id: str,
    target_organ: str,
    lesion_type: str,
    paths: MemorySourcePaths,
) -> Tuple[Dict[str, Any], List[str]]:
    path = os.path.join(paths.clip_global_dir, f"{image_id}.json")
    data = read_json(path)
    if not data:
        return {"available": False, "path": path, "failure_reason": "missing_clip_global"}, ["missing_clip_global"]
    organs = data.get("organs") or {}
    target_scores = organs.get(target_organ) or {}
    all_for_type = {
        organ: scores.get(lesion_type)
        for organ, scores in organs.items()
        if isinstance(scores, dict)
    }
    available = target_scores.get(lesion_type) is not None
    return {
        "available": bool(available),
        "path": path,
        "target_probability": target_scores.get(lesion_type),
        "target_organ_probabilities": target_scores,
        "all_organ_probabilities_for_lesion_type": all_for_type,
        "evidence_id": "clip_global:target_probability",
        "failure_reason": "" if available else "missing_target_probability",
    }, ([] if available else ["missing_clip_global_target_probability"])


def get_clip_global_matrix(image_id: str, paths: MemorySourcePaths) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    path = os.path.join(paths.clip_global_dir, f"{image_id}.json")
    data = read_json(path)
    if not data:
        return {}, ["missing_clip_global"]
    organs = data.get("organs") or {}
    matrix: Dict[str, Dict[str, Any]] = {}
    for organ in ALLOWED_ORGANS:
        scores = organs.get(organ) or {}
        matrix[organ] = {
            lesion_type: scores.get(lesion_type)
            for lesion_type in ALLOWED_LESION_TYPES
        }
    return matrix, []


def build_lesion_type_comparison(
    clip_global_matrix: Dict[str, Dict[str, Any]],
    target_organ: str,
    lesion_types_to_score: List[str],
) -> Dict[str, Any]:
    organ_scores = clip_global_matrix.get(target_organ) or {}
    compared = {
        lesion_type: organ_scores.get(lesion_type)
        for lesion_type in _ordered_unique(lesion_types_to_score, ALLOWED_LESION_TYPES)
    }
    numeric = {
        lesion_type: float(score)
        for lesion_type, score in compared.items()
        if isinstance(score, (int, float))
    }
    if len(numeric) < 2:
        return _without_empty({
            "target_organ": target_organ,
            "probabilities": compared,
        })
    ranked = sorted(numeric.items(), key=lambda item: item[1], reverse=True)
    return {
        "target_organ": target_organ,
        "probabilities": compared,
        "preferred_by_clip_global": ranked[0][0],
        "margin": round(float(ranked[0][1] - ranked[1][1]), 6),
    }


def detail_target_key(organ: str, lesion_type: str) -> str:
    return f"{organ}:{lesion_type}"


def section_entry(section_by_index: Dict[int, Dict[str, Any]], section_index: int) -> Dict[str, Any]:
    section = section_by_index.get(int(section_index)) or {}
    return {
        "section_index": int(section_index),
        "probability": float(section.get("probability", 0.0) or 0.0),
        "slice_index_range": section.get("slice_index_range"),
        "z_percent_range": section.get("z_percent_range"),
        "organ_area": section.get("organ_area"),
        "organ_patch_count": section.get("organ_patch_count"),
        "missing_section_or_zero_probability": not bool(section) or float(section.get("probability", 0.0) or 0.0) == 0.0,
    }


def get_clip_detail_memory(
    image_id: str,
    target_organ: str,
    lesion_type: str,
    paths: MemorySourcePaths,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[str]]:
    path = os.path.join(paths.clip_detail_dir, f"{image_id}.json")
    data = read_json(path)
    if not data:
        return {"available": False, "path": path, "failure_reason": "missing_clip_detail"}, [], ["missing_clip_detail"]
    organ_pack = (data.get("organs") or {}).get(target_organ) or {}
    lesion_pack = organ_pack.get(lesion_type) or {}
    sections = lesion_pack.get("sections") or []
    if not sections:
        return {
            "available": False,
            "path": path,
            "failure_reason": "missing_target_sections",
            "global_probability": lesion_pack.get("global_probability"),
            "max_section_index": lesion_pack.get("max_section_index"),
        }, [], ["missing_clip_detail_target_sections"]

    top_sections = sorted(
        sections,
        key=lambda item: float(item.get("probability", 0.0) or 0.0),
        reverse=True,
    )[:5]
    candidates = []
    for section in top_sections[:3]:
        slice_range = section.get("slice_index_range") or []
        if isinstance(slice_range, list) and len(slice_range) == 2:
            midpoint = int(round((float(slice_range[0]) + float(slice_range[1])) / 2.0))
        else:
            midpoint = int(section.get("section_index", 0)) * 10 + 5
        candidates.append(make_candidate(
            midpoint,
            "lesion_top_section_midpoint",
            f"{target_organ}:{lesion_type}",
            float(section.get("probability", 0.0) or 0.0),
            (
                f"Midpoint of CT-CLIP section {section.get('section_index')} with high "
                f"{lesion_type} probability in {target_organ}."
            ),
            [f"clip_detail:{target_organ}:{lesion_type}:section:{section.get('section_index')}"],
        ))
    return {
        "available": True,
        "path": path,
        "global_probability": lesion_pack.get("global_probability"),
        "max_section_index": lesion_pack.get("max_section_index"),
        "top_sections": top_sections,
        "evidence_id": "clip_detail:top_sections",
    }, candidates, []


def get_clip_detail_slice_memory(
    image_id: str,
    target_organ: str,
    lesion_type: str,
    paths: MemorySourcePaths,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[str]]:
    path = os.path.join(paths.clip_detail_slice_dir, f"{image_id}.json")
    data = read_json(path)
    if not data:
        return {"available": False, "path": path, "failure_reason": "missing_clip_detail_slice"}, [], ["missing_clip_detail_slice"]
    organ_pack = (data.get("organs") or {}).get(target_organ) or {}
    lesion_pack = organ_pack.get(lesion_type) or {}
    top_slices = lesion_pack.get("top_slices") or []
    if not top_slices:
        return {
            "available": False,
            "path": path,
            "failure_reason": "missing_target_top_slices",
            "global_probability": lesion_pack.get("global_probability"),
        }, [], ["missing_clip_detail_slice_target_top_slices"]
    candidates = []
    for item in top_slices[:5]:
        idx = item.get("slice_index")
        candidates.append(make_candidate(
            idx,
            "lesion_top_slice",
            f"{target_organ}:{lesion_type}",
            item.get("probability"),
            (
                f"CT-CLIP slice-level top candidate for {lesion_type} in {target_organ}; "
                f"probability={item.get('probability')}."
            ),
            [f"clip_detail_slice:{target_organ}:{lesion_type}:slice:{idx}"],
        ))
    return {
        "available": True,
        "path": path,
        "global_probability": lesion_pack.get("global_probability"),
        "top_slices": top_slices[:10],
        "slice_probabilities_available": isinstance(lesion_pack.get("slice_probabilities"), list),
        "evidence_id": "clip_detail_slice:top_slices",
    }, candidates, []


def get_slice_probability(
    clip_detail_slice_memory: Dict[str, Any],
    source_path: str,
    target_organ: str,
    lesion_type: str,
    slice_index_240: int,
) -> Optional[float]:
    if not clip_detail_slice_memory.get("slice_probabilities_available"):
        return None
    data = read_json(source_path)
    lesion_pack = (((data or {}).get("organs") or {}).get(target_organ) or {}).get(lesion_type) or {}
    values = lesion_pack.get("slice_probabilities")
    if not isinstance(values, list):
        return None
    idx = clamp_slice_index(slice_index_240)
    if idx >= len(values):
        return None
    try:
        return float(values[idx])
    except Exception:
        return None


def sanitize_runtime_tool(tool: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if tool is None:
        return None
    if isinstance(tool, dict):
        return {
            str(k): sanitize_runtime_tool(v)
            for k, v in tool.items()
            if str(k) not in LEAKY_RUNTIME_KEYS
        }
    if isinstance(tool, list):
        return [sanitize_runtime_tool(item) for item in tool]
    return tool


def runtime_tools_list(tool: Optional[Any]) -> List[Dict[str, Any]]:
    if tool is None:
        return []
    if isinstance(tool, list):
        return [item for item in tool if isinstance(item, dict)]
    if isinstance(tool, dict) and isinstance(tool.get("runtime_tools"), list):
        return [item for item in tool.get("runtime_tools", []) if isinstance(item, dict)]
    if isinstance(tool, dict):
        return [tool]
    return []


def primary_runtime_tool(tool: Optional[Any]) -> Optional[Dict[str, Any]]:
    tools = runtime_tools_list(tool)
    for item in tools:
        if item.get("available"):
            return item
    return tools[0] if tools else None


def runtime_tool_candidates(tool: Optional[Any], target_organ: str, lesion_type: str) -> List[Dict[str, Any]]:
    tools = runtime_tools_list(tool)
    candidates: List[Dict[str, Any]] = []
    for runtime_tool in tools:
        candidates.extend(_single_runtime_tool_candidates(runtime_tool, target_organ, lesion_type))
    return candidates


def _single_runtime_tool_candidates(tool: Optional[Dict[str, Any]], target_organ: str, lesion_type: str) -> List[Dict[str, Any]]:
    if not tool or not tool.get("available"):
        return []
    candidates = []
    for comp in tool.get("components") or []:
        center = comp.get("center_location") or []
        if len(center) >= 1:
            idx = int(center[0]) * 10 + 5
            candidates.append(make_candidate(
                idx,
                "runtime_component",
                f"{target_organ}:{lesion_type}",
                comp.get("avg_score"),
                (
                    f"Runtime CT-CLIP component candidate centered at temporal patch "
                    f"{center[0]} for {lesion_type} in {target_organ}."
                ),
                [f"runtime_tool:{tool.get('tool_name')}:component:{comp.get('id')}"],
            ))
    prior_sections = (((tool.get("prior_info") or {}).get("sections")) or [])[:3]
    for section in prior_sections:
        slice_range = section.get("slice_index_range") or []
        if isinstance(slice_range, list) and len(slice_range) == 2:
            idx = int(round((float(slice_range[0]) + float(slice_range[1])) / 2.0))
        else:
            idx = int(section.get("section_index", 0)) * 10 + 5
        candidates.append(make_candidate(
            idx,
            "runtime_component",
            f"{target_organ}:{lesion_type}",
            section.get("probability"),
            (
                f"Runtime HU/counting tool prior section for {lesion_type} in "
                f"{target_organ}."
            ),
            [f"runtime_tool:{tool.get('tool_name')}:prior_section:{section.get('section_index')}"],
        ))
    for location, details in (tool.get("all_scores") or {}).items():
        if not isinstance(details, dict):
            continue
        idx = details.get("slice_idx")
        if idx is None and details.get("temporal_patch") is not None:
            idx = int(details.get("temporal_patch")) * 10 + 5
        if idx is None:
            continue
        candidates.append(make_candidate(
            idx,
            "runtime_component",
            f"{target_organ}:{lesion_type}",
            details.get("score"),
            f"Runtime CT-CLIP location evidence for {location}.",
            [f"runtime_tool:{tool.get('tool_name')}:location:{location}"],
        ))
    return candidates


def build_option_evidence(
    question: str,
    image_id: str,
    dataset: str,
    target_organ: str,
    lesion_type: str,
    clip_detail_memory: Dict[str, Any],
    clip_detail_slice_memory: Dict[str, Any],
    runtime_tool: Optional[Any],
    paths: MemorySourcePaths,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    option_texts = parse_option_texts(question)
    numeric_options = parse_numeric_options(question)
    option_evidence: Dict[str, Any] = {}
    candidates: List[Dict[str, Any]] = []
    section_by_index = {
        int(section["section_index"]): section
        for section in clip_detail_memory.get("top_sections", [])
        if "section_index" in section
    }
    # Include all sections, not just top sections, when available.
    detail_data = read_json(clip_detail_memory.get("path", ""))
    lesion_pack = ((((detail_data or {}).get("organs") or {}).get(target_organ) or {}).get(lesion_type) or {})
    for section in lesion_pack.get("sections") or []:
        if "section_index" in section:
            section_by_index[int(section["section_index"])] = section

    for option_id, text in sorted(option_texts.items()):
        entry: Dict[str, Any] = {
            "option_text": text,
            "numeric_value": numeric_options.get(option_id),
            "evidence_ids": [],
        }
        if option_id in numeric_options and is_slice_question(question):
            percent = numeric_options[option_id]
            coordinate = map_vqa_percent_to_ctclip_slice(percent, image_id, dataset)
            idx = int(coordinate["slice_index_240"])
            section_index = section_index_from_ctclip_slice(idx)
            primary = section_entry(section_by_index, section_index)
            slice_prob = get_slice_probability(
                clip_detail_slice_memory,
                clip_detail_slice_memory.get("path", ""),
                target_organ,
                lesion_type,
                idx,
            )
            entry.update({
                "option_kind": "slice_percent",
                "slice_percent": percent,
                "slice_index_240": idx,
                "ctclip_percent": coordinate.get("ctclip_percent"),
                "raw_slice_index": coordinate.get("raw_slice_index"),
                "source_depth": coordinate.get("source_depth"),
                "coordinate_frame": coordinate.get("coordinate_frame"),
                "coordinate_transform_status": coordinate.get("coordinate_transform_status"),
                "coordinate_transform_note": coordinate.get("coordinate_transform_note"),
                "is_approximate_coordinate": coordinate.get("is_approximate_coordinate"),
                "direct_slice_index_240": coordinate.get("direct_slice_index_240"),
                "section_index": section_index,
                "section_evidence": primary,
                "slice_probability": slice_prob,
            })
            entry["evidence_ids"].extend([
                f"mcq_option:{option_id}:slice:{idx}",
                f"clip_detail:{target_organ}:{lesion_type}:section:{section_index}",
            ])
            candidates.append(make_candidate(
                idx,
                "mcq_option_slice",
                f"option {option_id}",
                slice_prob if slice_prob is not None else primary.get("probability"),
                (
                    f"MCQ option {option_id} maps {percent}% to CT-CLIP slice {idx} "
                    f"({coordinate.get('coordinate_transform_status')}) for possible later visual verification."
                ),
                entry["evidence_ids"],
                {
                    "option_id": option_id,
                    "slice_probability": slice_prob,
                    "section_probability": primary.get("probability"),
                    "raw_slice_index": coordinate.get("raw_slice_index"),
                    "source_depth": coordinate.get("source_depth"),
                    "coordinate_frame": coordinate.get("coordinate_frame"),
                    "coordinate_transform_status": coordinate.get("coordinate_transform_status"),
                    "ctclip_percent": coordinate.get("ctclip_percent"),
                    "is_approximate_coordinate": coordinate.get("is_approximate_coordinate"),
                },
            ))
        elif option_id in numeric_options:
            entry["option_kind"] = "numeric"
        else:
            entry["option_kind"] = "text"

        tool_evidence = []
        for tool in runtime_tools_list(runtime_tool):
            if not tool.get("available"):
                continue
            tool_name = tool.get("tool_name")
            evidence: Dict[str, Any] = {
                "tool_name": tool_name,
                "recommended": option_id in set(tool.get("recommended_options") or []),
            }
            if option_id in (tool.get("option_count_differences") or {}):
                evidence["abs_error"] = (tool.get("option_count_differences") or {}).get(option_id)
            if option_id in (tool.get("option_errors") or {}):
                evidence["abs_error"] = (tool.get("option_errors") or {}).get(option_id)
            if option_id in (tool.get("option_scores") or {}):
                evidence["score"] = (tool.get("option_scores") or {}).get(option_id)
            if option_id in (tool.get("option_location_scores") or {}):
                evidence["score"] = (tool.get("option_location_scores") or {}).get(option_id)
            if evidence.get("recommended") or any(key in evidence for key in ("abs_error", "score")):
                tool_evidence.append(evidence)
                entry["evidence_ids"].append(f"runtime_tool:{tool_name}:option:{option_id}")
        if tool_evidence:
            entry["tool_evidence"] = tool_evidence
            entry["tool_recommended"] = any(item.get("recommended") for item in tool_evidence)
            first_abs_error = next((item.get("abs_error") for item in tool_evidence if item.get("abs_error") is not None), None)
            first_score = next((item.get("score") for item in tool_evidence if item.get("score") is not None), None)
            if first_abs_error is not None:
                entry["tool_abs_error"] = first_abs_error
            if first_score is not None:
                entry["tool_score"] = first_score
        option_evidence[option_id] = entry
    return option_evidence, candidates


def build_facts_memory(
    row: Dict[str, Any],
    report: str,
    inferred_query: Dict[str, Any],
    paths: MemorySourcePaths,
    runtime_tool: Optional[Any] = None,
) -> Dict[str, Any]:
    question = str(row_get(row, "multiple-choice question", ""))
    image_id = str(row_get(row, "Image ID"))
    dataset = str(row_get(row, "dataset"))
    inferred_query = coerce_inferred_query(inferred_query, question)
    target_organ = inferred_query["target_organ"]
    primary_target_organ = inferred_query.get("primary_target_organ") or target_organ
    target_organs = inferred_query.get("target_organs") or [target_organ]
    option_organs = inferred_query.get("option_organs") or []
    lesion_type = inferred_query["lesion_type"]
    detail_targets = inferred_query.get("detail_targets") or []
    sanitized_runtime = sanitize_runtime_tool(runtime_tool)
    sanitized_runtime_tools = runtime_tools_list(sanitized_runtime)
    sanitized_primary_runtime = primary_runtime_tool(sanitized_runtime)

    all_organ_memory = build_all_organ_memory(
        dataset,
        image_id,
        report,
        paths,
        primary_target_organ,
        target_organs,
        option_organs,
    )
    organ_memory, organ_candidates, organ_missing = build_organ_memory_and_candidates(
        dataset,
        image_id,
        target_organ,
        report,
        paths,
    )
    clip_global_memory, global_missing = get_clip_global_memory(image_id, target_organ, lesion_type, paths)
    clip_global_matrix, matrix_missing = get_clip_global_matrix(image_id, paths)
    clip_detail_by_target: Dict[str, Dict[str, Any]] = {}
    clip_detail_slice_by_target: Dict[str, Dict[str, Any]] = {}
    detail_candidates: List[Dict[str, Any]] = []
    slice_candidates: List[Dict[str, Any]] = []
    detail_missing: List[str] = []
    slice_missing: List[str] = []
    for item in detail_targets:
        if not isinstance(item, dict):
            continue
        organ = normalize_organ_choice(item.get("organ"))
        lesion = normalize_lesion_choice(item.get("lesion_type"))
        if not organ or not lesion:
            continue
        key = detail_target_key(organ, lesion)
        detail_memory, target_detail_candidates, target_detail_missing = get_clip_detail_memory(
            image_id,
            organ,
            lesion,
            paths,
        )
        slice_memory, target_slice_candidates, target_slice_missing = get_clip_detail_slice_memory(
            image_id,
            organ,
            lesion,
            paths,
        )
        clip_detail_by_target[key] = detail_memory
        clip_detail_slice_by_target[key] = slice_memory
        detail_candidates.extend(target_detail_candidates)
        slice_candidates.extend(target_slice_candidates)
        detail_missing.extend(f"{key}:{value}" for value in target_detail_missing)
        slice_missing.extend(f"{key}:{value}" for value in target_slice_missing)
    primary_key = detail_target_key(target_organ, lesion_type)
    clip_detail_memory = clip_detail_by_target.get(primary_key) or {"available": False, "failure_reason": "detail_target_not_selected"}
    clip_detail_slice_memory = clip_detail_slice_by_target.get(primary_key) or {"available": False, "failure_reason": "detail_target_not_selected"}
    report_lesion_slices = extract_report_lesion_slices(report, target_organ)
    report_candidates = [
        make_candidate(
            idx,
            "report_lesion_slice",
            f"{target_organ}:{lesion_type}",
            None,
            f"Structured report mentions a lesion-related slice in the {target_organ} section.",
            [f"structured_report:{target_organ}:slice:{idx}"],
        )
        for idx in report_lesion_slices
    ]
    option_evidence, option_candidates = build_option_evidence(
        question,
        image_id,
        dataset,
        target_organ,
        lesion_type,
        clip_detail_memory,
        clip_detail_slice_memory,
        sanitized_runtime,
        paths,
    )
    runtime_candidates = runtime_tool_candidates(sanitized_runtime, target_organ, lesion_type)

    candidate_slice_queue = merge_and_limit_candidates(
        [
            *option_candidates,
            *slice_candidates,
            *detail_candidates,
            *runtime_candidates,
            *report_candidates,
            *organ_candidates,
        ],
        max_items=10,
        merge_distance=8,
    )

    missing = [
        *organ_missing,
        *global_missing,
        *matrix_missing,
        *detail_missing,
        *slice_missing,
    ]
    if inferred_query["confidence"] < 0.5:
        missing.append("low_confidence_query_normalization")
    for tool in sanitized_runtime_tools:
        if not tool.get("available"):
            missing.append(f"runtime_tool_unavailable:{tool.get('tool_name')}:{tool.get('failure_reason', 'unknown')}")

    audit_organ = normalize_organ_choice(row_get(row, "organ"))
    audit_lesion = normalize_lesion_choice(row_get(row, "lesion")) or "lesion"

    facts_memory = {
        "case_context": {
            "image_id": image_id,
            "dataset": dataset,
            "shape": row_get(row, "shape", ""),
            "spacing": row_get(row, "spacing", ""),
            "sex": row_get(row, "sex", ""),
            "age": row_get(row, "age", ""),
            "contrast": row_get(row, "contrast", ""),
            "scanner": row_get(row, "scanner", ""),
        },
        "question": question,
        "inferred_query": inferred_query,
        "report_memory": {
            "full_structured_report": str(report or "")[:8000],
            "target_organ_block": extract_report_block(report, target_organ),
        },
        "organ_memory": organ_memory,
        "all_organ_memory": all_organ_memory,
        "lesion_memory": {
            "clip_global": clip_global_memory,
            "clip_global_matrix": clip_global_matrix,
            "selected_detail_targets": detail_targets,
            "clip_detail": clip_detail_memory,
            "clip_detail_slice": clip_detail_slice_memory,
            "clip_detail_by_target": clip_detail_by_target,
            "clip_detail_slice_by_target": clip_detail_slice_by_target,
            "lesion_type_comparison": build_lesion_type_comparison(
                clip_global_matrix,
                target_organ,
                inferred_query.get("lesion_types_to_score") or [lesion_type],
            ),
            "report_lesion_slices": report_lesion_slices,
            "runtime_tool": sanitized_primary_runtime,
            "runtime_tools": sanitized_runtime_tools,
        },
        "candidate_slice_queue": candidate_slice_queue,
        "option_evidence": option_evidence,
        "conflicts_and_missingness": {
            "missing": sorted(set(missing)),
            "conflicts": [],
            "notes": [
                "Candidate slices are text-only pointers for future visual verification.",
                "No image pixels are provided to the answer model in this v3 pipeline.",
            ],
        },
        "internal_audit": {
            "csv_organ": audit_organ,
            "csv_lesion": audit_lesion,
            "inferred_matches_csv_organ": audit_organ == target_organ,
            "inferred_matches_csv_lesion": audit_lesion == lesion_type,
            "gt_answer": row_get(row, "correct option", ""),
            "question_type": row_get(row, "question type", ""),
            "question_subtype": row_get(row, "question subtype", ""),
        },
    }
    return facts_memory


def _is_present(value: Any) -> bool:
    return value is not None and value != "" and value != [] and value != {}


def _shorten_text(value: Any, max_chars: int = 240) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _without_empty(data: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in data.items() if _is_present(value)}


def _compact_report_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(metrics, dict):
        return {}
    return _without_empty({
        key: value
        for key, value in metrics.items()
        if key != "report_block"
    })


def _compact_clip_global(memory: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(memory, dict):
        return {}
    return _without_empty({
        "available": bool(memory.get("available")),
        "target_probability": memory.get("target_probability"),
        "target_organ_probabilities": memory.get("target_organ_probabilities"),
        "evidence_id": memory.get("evidence_id"),
        "failure_reason": memory.get("failure_reason"),
    })


def _compact_section(section: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(section, dict):
        return {}
    return _without_empty({
        "section_index": section.get("section_index"),
        "probability": section.get("probability"),
        "slice_index_range": section.get("slice_index_range"),
        "z_percent_range": section.get("z_percent_range"),
        "organ_area": section.get("organ_area"),
        "organ_patch_count": section.get("organ_patch_count"),
    })


def _compact_slice(item: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    return _without_empty({
        "slice_index": item.get("slice_index"),
        "probability": item.get("probability"),
        "z_percent": item.get("z_percent"),
    })


def _compact_clip_detail(memory: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(memory, dict):
        return {}
    top_sections = [
        section
        for section in (_compact_section(item) for item in (memory.get("top_sections") or [])[:3])
        if section
    ]
    return _without_empty({
        "available": bool(memory.get("available")),
        "global_probability": memory.get("global_probability"),
        "max_section_index": memory.get("max_section_index"),
        "top_sections": top_sections,
        "evidence_id": memory.get("evidence_id"),
        "failure_reason": memory.get("failure_reason"),
    })


def _compact_clip_detail_slice(memory: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(memory, dict):
        return {}
    top_slices = [
        item
        for item in (_compact_slice(entry) for entry in (memory.get("top_slices") or [])[:5])
        if item
    ]
    return _without_empty({
        "available": bool(memory.get("available")),
        "global_probability": memory.get("global_probability"),
        "top_slices": top_slices,
        "evidence_id": memory.get("evidence_id"),
        "failure_reason": memory.get("failure_reason"),
    })


def _compact_all_organ_memory(memory: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for organ in ALLOWED_ORGANS:
        item = memory.get(organ) if isinstance(memory, dict) else {}
        if not isinstance(item, dict):
            continue
        compact[organ] = _without_empty({
            "role": item.get("role"),
            "report_metrics": _compact_report_metrics(item.get("report_metrics") or {}),
            "mask_summary": item.get("mask_summary"),
        })
    return _without_empty(compact)


def _compact_detail_map(memory: Dict[str, Any], slice_level: bool = False) -> Dict[str, Any]:
    if not isinstance(memory, dict):
        return {}
    compact: Dict[str, Any] = {}
    for key, value in memory.items():
        compact_value = _compact_clip_detail_slice(value) if slice_level else _compact_clip_detail(value)
        if compact_value:
            compact[key] = compact_value
    return compact


def _compact_runtime_tool(tool: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(tool, dict):
        return None
    compact = {
        "available": bool(tool.get("available")),
        "tool_name": tool.get("tool_name"),
        "failure_reason": tool.get("failure_reason"),
        "recommended_options": tool.get("recommended_options"),
        "evidence_strength": tool.get("evidence_strength"),
        "key_measurement": tool.get("key_measurement"),
    }
    return _without_empty(compact)


def _compact_runtime_tools(tool: Optional[Any]) -> List[Dict[str, Any]]:
    compact_tools = []
    for item in runtime_tools_list(tool):
        compact = _compact_runtime_tool(item)
        if compact:
            compact_tools.append(compact)
    return compact_tools


def _compact_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    description = str(candidate.get("description") or "")
    if ". Also: " in description:
        description = description.split(". Also: ", 1)[0].rstrip(".") + "."
    if candidate.get("option_id"):
        description = f"Option {candidate.get('option_id')} maps to this CT-CLIP slice."
    return _without_empty({
        "slice_index_240": candidate.get("slice_index_240"),
        "z_percent": candidate.get("z_percent"),
        "option_id": candidate.get("option_id"),
        "source_type": candidate.get("source_type"),
        "target": candidate.get("target"),
        "score": candidate.get("score"),
        "slice_probability": candidate.get("slice_probability"),
        "section_probability": candidate.get("section_probability"),
        "description": _shorten_text(description, max_chars=160),
        "evidence_ids": (candidate.get("evidence_ids") or [])[:3],
    })


def _compact_tool_evidence(item: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    return _without_empty({
        "tool_name": item.get("tool_name"),
        "recommended": item.get("recommended"),
        "abs_error": item.get("abs_error"),
        "score": item.get("score"),
    })


def _compact_option_evidence(option_evidence: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for option_id, entry in (option_evidence or {}).items():
        if not isinstance(entry, dict):
            continue
        section_evidence = entry.get("section_evidence") if isinstance(entry.get("section_evidence"), dict) else {}
        compact[str(option_id)] = _without_empty({
            "option_text": entry.get("option_text"),
            "numeric_value": entry.get("numeric_value"),
            "option_kind": entry.get("option_kind"),
            "slice_percent": entry.get("slice_percent"),
            "slice_index_240": entry.get("slice_index_240"),
            "section_index": entry.get("section_index"),
            "section_probability": section_evidence.get("probability"),
            "slice_probability": entry.get("slice_probability"),
            "tool_recommended": entry.get("tool_recommended"),
            "tool_abs_error": entry.get("tool_abs_error"),
            "tool_score": entry.get("tool_score"),
            "tool_evidence": [
                item
                for item in (_compact_tool_evidence(tool_item) for tool_item in (entry.get("tool_evidence") or []))
                if item
            ],
            "evidence_ids": (entry.get("evidence_ids") or [])[:6],
        })
    return compact


def build_compact_facts_memory(verbose_memory: Dict[str, Any]) -> Dict[str, Any]:
    """Build the prompt-facing memory while keeping verbose memory for debug."""
    case_context = verbose_memory.get("case_context") or {}
    inferred_query = verbose_memory.get("inferred_query") or {}
    report_memory = verbose_memory.get("report_memory") or {}
    organ_memory = verbose_memory.get("organ_memory") or {}
    all_organ_memory = verbose_memory.get("all_organ_memory") or {}
    lesion_memory = verbose_memory.get("lesion_memory") or {}
    conflicts = verbose_memory.get("conflicts_and_missingness") or {}

    target_block = str(report_memory.get("target_organ_block") or "").strip()
    compact_report = {"target_organ_block": target_block} if target_block else {
        "full_structured_report_fallback": _shorten_text(
            report_memory.get("full_structured_report", ""),
            max_chars=1500,
        )
    }

    compact_organ_memory = _without_empty({
        "target_organ": organ_memory.get("target_organ"),
        "report_metrics": _compact_report_metrics(organ_memory.get("report_metrics") or {}),
        "mask_summary": organ_memory.get("mask_summary"),
    })

    compact_lesion_memory = _without_empty({
        "clip_global": _compact_clip_global(lesion_memory.get("clip_global") or {}),
        "clip_global_matrix": lesion_memory.get("clip_global_matrix") or {},
        "selected_detail_targets": lesion_memory.get("selected_detail_targets"),
        "clip_detail_by_target": _compact_detail_map(lesion_memory.get("clip_detail_by_target") or {}),
        "clip_detail_slice_by_target": _compact_detail_map(
            lesion_memory.get("clip_detail_slice_by_target") or {},
            slice_level=True,
        ),
        "lesion_type_comparison": lesion_memory.get("lesion_type_comparison"),
        "report_lesion_slices": lesion_memory.get("report_lesion_slices"),
        "runtime_tool": _compact_runtime_tool(lesion_memory.get("runtime_tool")),
        "runtime_tools": _compact_runtime_tools(lesion_memory.get("runtime_tools")),
    })

    compact_memory = {
        "case_context": _without_empty({
            "image_id": case_context.get("image_id"),
            "dataset": case_context.get("dataset"),
            "shape": case_context.get("shape"),
            "spacing": case_context.get("spacing"),
        }),
        "question": verbose_memory.get("question"),
        "inferred_query": _without_empty({
            "target_organ": inferred_query.get("target_organ"),
            "primary_target_organ": inferred_query.get("primary_target_organ"),
            "target_organs": inferred_query.get("target_organs"),
            "stem_organs": inferred_query.get("stem_organs"),
            "option_organs": inferred_query.get("option_organs"),
            "lesion_type": inferred_query.get("lesion_type"),
            "lesion_types_to_score": inferred_query.get("lesion_types_to_score"),
            "detail_targets": inferred_query.get("detail_targets"),
            "question_intent": inferred_query.get("question_intent"),
            "confidence": inferred_query.get("confidence"),
            "rationale": inferred_query.get("rationale"),
        }),
        "report_memory": _without_empty(compact_report),
        "organ_memory": compact_organ_memory,
        "all_organ_memory": _compact_all_organ_memory(all_organ_memory),
        "lesion_memory": compact_lesion_memory,
        "candidate_slice_queue": [
            item
            for item in (_compact_candidate(candidate) for candidate in verbose_memory.get("candidate_slice_queue") or [])
            if item
        ],
        "option_evidence": _compact_option_evidence(verbose_memory.get("option_evidence") or {}),
        "conflicts_and_missingness": _without_empty({
            "missing": conflicts.get("missing"),
            "conflicts": conflicts.get("conflicts"),
        }),
    }
    return compact_memory


def validate_facts_memory(memory: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []
    inferred = memory.get("inferred_query") or {}
    if inferred.get("target_organ") not in ALLOWED_ORGANS:
        warnings.append("invalid_target_organ")
    if inferred.get("lesion_type") not in ALLOWED_LESION_TYPES:
        warnings.append("invalid_lesion_type")
    candidates = memory.get("candidate_slice_queue") or []
    if len(candidates) > 10:
        warnings.append("too_many_candidate_slices")
    for idx, candidate in enumerate(candidates):
        slice_idx = candidate.get("slice_index_240")
        if not isinstance(slice_idx, int) or not 0 <= slice_idx < TOTAL_SLICES:
            warnings.append(f"candidate_{idx}_invalid_slice")
        if not candidate.get("description"):
            warnings.append(f"candidate_{idx}_missing_description")
        if not isinstance(candidate.get("evidence_ids"), list) or not candidate.get("evidence_ids"):
            warnings.append(f"candidate_{idx}_missing_evidence_ids")
    return warnings


def _compact_json(data: Any, max_chars: int = 12000) -> str:
    text = json.dumps(data, ensure_ascii=False, indent=2)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...<truncated>"


def render_memory_for_prompt(memory: Dict[str, Any], reasoning_memory: Optional[Any] = None) -> str:
    """Render only non-audit fields for GPT answer prompting."""
    public_memory = {k: v for k, v in memory.items() if k != "internal_audit"}
    lines = [
        "Leak-safe text-only evidence memory:",
        "- The target organ and lesion type below were inferred from the question, not copied from CSV labels.",
        "- Candidate slices are not images; they are textual pointers for possible future verification.",
        "",
        _compact_json(public_memory, max_chars=18000),
    ]
    if reasoning_memory:
        lines.extend([
            "",
            "Read-only compressed reasoning memory:",
            _compact_json(reasoning_memory, max_chars=6000),
        ])
    return "\n".join(lines)


def build_reasoning_memory_prompt(memory: Dict[str, Any]) -> str:
    public_memory = {k: v for k, v in memory.items() if k != "internal_audit"}
    return f"""Compress the following rule-based evidence memory for a text-only medical VQA answerer.

Rules:
- Do not add facts that are not present in the memory.
- Do not use hidden CSV labels or ground truth.
- Preserve uncertainty, missing evidence, CT-CLIP probabilities, option evidence, and candidate slice descriptions.
- Output STRICT JSON only.

Output schema:
{{
  "summary": "<short evidence summary>",
  "key_evidence": ["...", "..."],
  "option_guidance": {{"A": "..."}},
  "candidate_slice_summary": ["...", "..."],
  "missing_or_uncertain": ["...", "..."]
}}

Memory:
{_compact_json(public_memory, max_chars=20000)}
"""


def build_answer_prompt(question: str, memory: Dict[str, Any], reasoning_memory: Optional[Any] = None) -> str:
    rendered = render_memory_for_prompt(memory, reasoning_memory=reasoning_memory)
    return f"""You are answering a multiple-choice 3D medical VQA question using text-only evidence.

Important constraints:
- You are NOT given images in this step.
- Do not use CSV organ, lesion, answer, or correct-option labels.
- Use the inferred query, option evidence, CT-CLIP evidence, structured report evidence, and candidate slice descriptions.
- Candidate slices are future visual-verification pointers only; treat them as text evidence from their listed source.
- Return exactly one option letter in final_answer.

Output STRICT JSON:
{{
  "analysis": "<brief reasoning grounded in the memory>",
  "final_answer": "<ONE option letter only>",
  "evidence_ids_used": ["<evidence id>", "..."],
  "slice_candidates_relevant": [<slice_index_240>, ...],
  "uncertainty": "<low|medium|high plus short reason>",
  "assumptions": ["...", "..."]
}}

Question:
{question}

{rendered}
"""
