#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Runtime CT-CLIP planning, validation, and cache helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional

from Tool_Box.io import safe_json_loads
from slice_coordinates import map_vqa_percent_to_ctclip_slice


TOOL_REGISTRY = [
    {
        "tool_name": "ctclip_counting",
        "use_when": "Count distinct lesions in the inferred target organ.",
        "inputs": ["numeric_options optional for MCQ matching"],
        "limits": "Does not count lesions in a named subregion.",
    },
    {
        "tool_name": "ctclip_location",
        "use_when": "Score explicit candidate anatomical regions or slice percentages for lesion evidence.",
        "inputs": ["locations", "location_type: organ|slice", "option_to_location optional"],
        "limits": "Ranks lesion evidence within provided regions/slices; not suitable for contact, distance, adjacency, or nearest-organ questions.",
    },
    {
        "tool_name": "ctclip_hu_difference",
        "use_when": "Estimate absolute HU difference between lesion and inferred target organ for numeric MCQ options.",
        "inputs": ["numeric_options"],
        "limits": "Requires numeric HU-difference options.",
    },
    {
        "tool_name": "ctclip_hu_attenuation",
        "use_when": "Classify lesion attenuation as hypoattenuating, isoattenuating, or hyperattenuating.",
        "inputs": ["text options describing attenuation"],
        "limits": "Only answers attenuation class.",
    },
]

TOOL_NAMES = {item["tool_name"] for item in TOOL_REGISTRY}

LEAKY_TOOL_KEYS = {
    "answer",
    "correct_option",
    "correct_count",
    "count_diff",
    "answer_value",
    "hu_diff_signed_error",
    "hu_diff_abs_error",
    "is_correct",
    "is_evaluable_mcq",
    "evaluable_mcq",
}


def build_tool_registry() -> List[Dict[str, Any]]:
    return json.loads(json.dumps(TOOL_REGISTRY))


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


def parse_numeric_options(mcq: Any) -> Dict[str, float]:
    return {
        letter: float(value)
        for letter, value in re.findall(r"\b([A-Z]):\s*([-+]?\d+(?:\.\d+)?)", str(mcq or ""))
    }


def normalize_location_label(value: Any, target_organ: str) -> Optional[str]:
    text = str(value or "").strip().lower().replace("_", " ")
    if not text:
        return None
    segment_match = re.search(r"(?:segment|seg)\s*([1-8])\b", text)
    if segment_match:
        return segment_match.group(1)
    if target_organ == "liver" and re.fullmatch(r"[1-8]", text):
        return text
    if "left" in text:
        return "left"
    if "right" in text:
        return "right"
    organ_alias = {
        "intestine": "colon",
        "bowel": "colon",
        "stomach": "stomach",
        "colon": "colon",
        "spleen": "spleen",
        "kidney": "kidney",
        "pancreas": "pancreas",
        "liver": "liver",
    }
    return organ_alias.get(text, text.replace(" ", "_"))


def normalize_location_values(value: Any, target_organ: str) -> List[str]:
    text = str(value or "").strip().lower()
    if not text:
        return []
    segment_values = re.findall(r"(?:segment|seg)\s*([1-8])\b", text)
    if segment_values:
        return segment_values
    if target_organ == "liver":
        numeric_values = re.findall(r"\b([1-8])\b", text)
        if len(numeric_values) > 1 or re.fullmatch(r"[1-8]", text):
            return numeric_values
    values: List[str] = []
    for part in re.split(r"\s*(?:,|/|;|\band\b|\bor\b)\s*", text):
        label = normalize_location_label(part, target_organ)
        if label and label not in values:
            values.append(label)
    return values


def _compact_json(data: Any, max_chars: int = 16000) -> str:
    text = json.dumps(data, ensure_ascii=False, indent=2)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...<truncated>"


def build_tool_planning_prompt(
    question: str,
    inferred_query: Dict[str, Any],
    facts_memory: Dict[str, Any],
    tool_registry: List[Dict[str, Any]],
) -> str:
    option_texts = parse_option_texts(question)
    return f"""Plan optional runtime tool calls for this medical VQA case.

Rules:
- First inspect the compact facts_memory. If it already contains enough text/numeric evidence to answer, return an empty selected_tool_calls list.
- Select tools only when their output can add direct, material evidence for this MCQ.
- Do not call a tool for a task outside the tool's stated limits.
- ctclip_location only ranks lesion evidence among explicitly provided regions/slices. Do not use it for global existence, physical contact, distance, adjacency, nearest-organ, lesion clustering, diameter/volume, whole-volume count, or continuity reasoning.
- ctclip_counting only counts lesions in the whole inferred target organ. Do not use it for location-specific counts.
- Use HU tools only when the MCQ options directly ask for attenuation class or numeric HU difference.
- The final answer model is text-only, but runtime tools can access cached CT-CLIP embeddings, masks, reports, and HU arrays by image_id.
- Do not use hidden CSV labels, ground-truth answers, or correct-option labels.
- Do not invent tool names. Use only the provided registry.
- For location tools, provide explicit locations and location_type ("organ" or "slice").
- For slice questions, use MCQ numeric percentages as locations with location_type "slice".
- Output STRICT JSON only.

Output schema:
{{
  "selected_tool_calls": [
    {{
      "tool_call_id": "<short unique id>",
      "tool_name": "<registry tool_name>",
      "purpose": "<why this tool adds evidence beyond memory>",
      "inputs": {{
        "locations": ["left", "right", "1", "50.0"],
        "location_type": "organ|slice",
        "option_to_location": {{"A": "left"}},
        "parameters": {{}}
      }},
      "expected_evidence": "<what evidence should be produced>"
    }}
  ],
  "rationale": "<brief reason, including why no tool is needed if selected_tool_calls is empty>",
  "confidence": <0.0-1.0>
}}

Question:
{question}

MCQ options:
{json.dumps(option_texts, ensure_ascii=False, indent=2)}

Inferred query:
{json.dumps(inferred_query, ensure_ascii=False, indent=2)}

Compact facts_memory:
{_compact_json(facts_memory)}

Tool registry:
{json.dumps(tool_registry, ensure_ascii=False, indent=2)}
"""


async def plan_runtime_tool_calls_with_gpt(
    question: str,
    inferred_query: Dict[str, Any],
    facts_memory: Dict[str, Any],
    tool_registry: List[Dict[str, Any]],
    selector: str,
    model: str,
    call_gpt: Callable[..., Awaitable[str]],
) -> Dict[str, Any]:
    if selector != "gpt":
        return {
            "selected_tool_calls": [],
            "rationale": "runtime tool planner disabled",
            "confidence": 1.0,
            "selector": selector,
        }
    try:
        response = await call_gpt(
            build_tool_planning_prompt(question, inferred_query, facts_memory, tool_registry),
            model=model,
        )
        parsed = safe_json_loads(response) or {}
        raw_calls = parsed.get("selected_tool_calls", [])
        if not isinstance(raw_calls, list):
            raw_calls = []
        return {
            "selected_tool_calls": raw_calls,
            "rationale": str(parsed.get("rationale", "")),
            "confidence": float(parsed.get("confidence", 0.0) or 0.0),
            "selector": "gpt",
            "raw_response": response,
        }
    except Exception as exc:
        return {
            "selected_tool_calls": [],
            "rationale": "runtime tool planner failed; no fallback tool was called",
            "confidence": 0.0,
            "selector": "gpt_failed_no_fallback",
            "selector_error": str(exc),
        }


def _default_tool_parameters(tool_name: str) -> Dict[str, Any]:
    if tool_name == "ctclip_counting":
        return {"threshold": 0.6, "min_size": 3, "connectivity": 2}
    if tool_name == "ctclip_location":
        return {"scoring_method": "hybrid"}
    if tool_name in {"ctclip_hu_difference", "ctclip_hu_attenuation"}:
        return {"top_percent": 0.005, "max_patches": 8, "hu_stat": "mean"}
    return {}


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _parse_float_like(value: Any) -> Optional[float]:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value or ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except Exception:
        return None


def normalize_tool_calls(
    raw_calls: Any,
    row: Dict[str, Any],
    inferred_query: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], List[str]]:
    question = str(row.get("multiple-choice question", ""))
    option_texts = parse_option_texts(question)
    numeric_options = parse_numeric_options(question)
    target_organ = str(inferred_query.get("target_organ"))
    lesion_type = str(inferred_query.get("lesion_type"))
    base_inputs = {
        "image_id": str(row.get("Image ID")),
        "dataset": str(row.get("dataset")),
        "target_organ": target_organ,
        "lesion_type": lesion_type,
        "question_intent": inferred_query.get("question_intent"),
        "options": option_texts,
    }
    warnings: List[str] = []
    normalized: List[Dict[str, Any]] = []
    used_ids: set[str] = set()

    for idx, call in enumerate(raw_calls if isinstance(raw_calls, list) else []):
        if not isinstance(call, dict):
            warnings.append(f"tool_call_{idx}:not_a_dict")
            continue
        tool_name = str(call.get("tool_name") or "")
        if tool_name not in TOOL_NAMES:
            warnings.append(f"tool_call_{idx}:unknown_tool:{tool_name}")
            continue

        raw_inputs = call.get("inputs") if isinstance(call.get("inputs"), dict) else {}
        inputs = {**base_inputs, "parameters": _default_tool_parameters(tool_name)}
        if isinstance(raw_inputs.get("parameters"), dict):
            inputs["parameters"].update({
                str(k): v for k, v in raw_inputs["parameters"].items()
                if str(k) in {"threshold", "min_size", "connectivity", "scoring_method", "top_percent", "max_patches", "hu_stat"}
            })

        if tool_name in {"ctclip_counting", "ctclip_hu_difference"}:
            if not numeric_options:
                warnings.append(f"tool_call_{idx}:{tool_name}:missing_numeric_options")
                continue
            inputs["numeric_options"] = numeric_options
        if tool_name == "ctclip_location":
            location_type = str(raw_inputs.get("location_type") or "organ").lower()
            if location_type not in {"organ", "slice"}:
                warnings.append(f"tool_call_{idx}:{tool_name}:invalid_location_type:{location_type}")
                continue
            inputs["location_type"] = location_type

            option_to_location = raw_inputs.get("option_to_location") if isinstance(raw_inputs.get("option_to_location"), dict) else {}
            if location_type == "slice":
                locations = []
                coordinate_transforms: Dict[str, Any] = {}
                for item in _as_list(raw_inputs.get("locations")):
                    value = _parse_float_like(item)
                    if value is not None:
                        mapped = map_vqa_percent_to_ctclip_slice(value, row.get("Image ID"), row.get("dataset"))
                        coordinate_transforms[str(value)] = mapped
                        locations.append(str(mapped.get("ctclip_percent", value)))
                if not locations and numeric_options:
                    for value in numeric_options.values():
                        mapped = map_vqa_percent_to_ctclip_slice(value, row.get("Image ID"), row.get("dataset"))
                        coordinate_transforms[str(value)] = mapped
                        locations.append(str(mapped.get("ctclip_percent", value)))
                normalized_option_map = {}
                for opt, value in (option_to_location or {}).items():
                    parsed_value = _parse_float_like(value)
                    if opt in option_texts and parsed_value is not None:
                        mapped = map_vqa_percent_to_ctclip_slice(parsed_value, row.get("Image ID"), row.get("dataset"))
                        coordinate_transforms[f"option:{opt}"] = mapped
                        normalized_option_map[opt] = str(mapped.get("ctclip_percent", parsed_value))
                if not normalized_option_map and numeric_options:
                    normalized_option_map = {}
                    for opt, value in numeric_options.items():
                        mapped = map_vqa_percent_to_ctclip_slice(value, row.get("Image ID"), row.get("dataset"))
                        coordinate_transforms[f"option:{opt}"] = mapped
                        normalized_option_map[opt] = str(mapped.get("ctclip_percent", value))
                if coordinate_transforms:
                    inputs["coordinate_transforms"] = coordinate_transforms
            else:
                locations = []
                for item in _as_list(raw_inputs.get("locations")):
                    for label in normalize_location_values(item, target_organ):
                        if label not in locations:
                            locations.append(label)
                normalized_option_map = {}
                for opt, value in (option_to_location or {}).items():
                    labels = normalize_location_values(value, target_organ)
                    if opt in option_texts and labels:
                        normalized_option_map[opt] = labels[0]
                        for label in labels:
                            if label not in locations:
                                locations.append(label)
            if not locations:
                warnings.append(f"tool_call_{idx}:{tool_name}:missing_locations")
                continue
            inputs["locations"] = locations
            if normalized_option_map:
                inputs["option_to_location"] = normalized_option_map

        raw_id = str(call.get("tool_call_id") or tool_name)
        tool_call_id = re.sub(r"[^a-zA-Z0-9_]+", "_", raw_id).strip("_").lower() or tool_name
        while tool_call_id in used_ids:
            tool_call_id = f"{tool_call_id}_{idx}"
        used_ids.add(tool_call_id)
        normalized.append({
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "purpose": str(call.get("purpose") or ""),
            "inputs": inputs,
            "expected_evidence": str(call.get("expected_evidence") or ""),
        })
    return normalized, warnings


def strip_leaky_tool_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): strip_leaky_tool_fields(item)
            for key, item in value.items()
            if str(key) not in LEAKY_TOOL_KEYS and not str(key).endswith("_error_to_gt")
        }
    if isinstance(value, list):
        return [strip_leaky_tool_fields(item) for item in value]
    return value


def _numeric_values(data: Any, reverse: bool = True) -> List[tuple[str, float]]:
    if not isinstance(data, dict):
        return []
    values = []
    for key, value in data.items():
        try:
            values.append((str(key), float(value)))
        except Exception:
            continue
    return sorted(values, key=lambda item: item[1], reverse=reverse)


def _margin_strength(margin: Optional[float], strong: float, medium: float) -> str:
    if margin is None:
        return "not_actionable"
    if margin >= strong:
        return "strong"
    if margin >= medium:
        return "medium"
    if margin > 0:
        return "weak"
    return "not_actionable"


def enrich_runtime_tool_output(output: Dict[str, Any]) -> Dict[str, Any]:
    """Add leak-safe, prompt-facing confidence fields to a runtime tool output."""
    if not isinstance(output, dict):
        return output
    tool_name = output.get("tool_name")
    enriched = dict(output)
    strength = "not_actionable"
    margin: Optional[float] = None
    key_measurement = ""

    if tool_name == "ctclip_counting":
        diffs = _numeric_values(output.get("option_count_differences"), reverse=False)
        if diffs:
            margin = (diffs[1][1] - diffs[0][1]) if len(diffs) > 1 else None
            strength = _margin_strength(margin, strong=2.0, medium=1.0)
            key_measurement = (
                f"predicted count {output.get('predicted_count')}; "
                f"best {diffs[0][0]} diff {diffs[0][1]:g}"
                + (f", margin {margin:g}" if margin is not None else "")
            )
    elif tool_name == "ctclip_location":
        scores = _numeric_values(output.get("option_location_scores") or output.get("distributions"), reverse=True)
        if scores:
            margin = (scores[0][1] - scores[1][1]) if len(scores) > 1 else None
            strength = _margin_strength(margin, strong=0.10, medium=0.03)
            key_measurement = (
                f"best {scores[0][0]} score {scores[0][1]:.4f}"
                + (f", margin {margin:.4f}" if margin is not None else "")
            )
    elif tool_name == "ctclip_hu_difference":
        errors = _numeric_values(output.get("option_errors"), reverse=False)
        if errors:
            margin = (errors[1][1] - errors[0][1]) if len(errors) > 1 else None
            strength = _margin_strength(margin, strong=15.0, medium=7.5)
            try:
                predicted_hu_difference = float(output.get("predicted_hu_difference"))
            except Exception:
                predicted_hu_difference = 0.0
            key_measurement = (
                f"predicted HU difference {predicted_hu_difference:.1f}; "
                f"best {errors[0][0]} error {errors[0][1]:.1f}"
                + (f", margin {margin:.1f}" if margin is not None else "")
            )
    elif tool_name == "ctclip_hu_attenuation":
        recommended = output.get("recommended_options") or []
        predicted = output.get("predicted_attenuation")
        strength = "medium" if recommended and predicted else "weak" if predicted else "not_actionable"
        key_measurement = f"predicted attenuation {predicted}" if predicted else ""

    enriched["score_margin"] = margin
    enriched["evidence_strength"] = strength
    enriched["actionable"] = strength in {"strong", "medium"}
    enriched["key_measurement"] = key_measurement
    return enriched


def runtime_cache_path(cache_dir: str, image_id: str) -> str:
    return os.path.join(cache_dir, f"{image_id}.json")


def _read_json(path: str) -> Any:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_runtime_cache(cache_dir: str, image_id: str, dataset: str) -> Dict[str, Any]:
    path = runtime_cache_path(cache_dir, image_id)
    data = _read_json(path)
    if isinstance(data, dict):
        data.setdefault("schema_version", "runtime_tools_v1")
        data.setdefault("image_id", image_id)
        data.setdefault("dataset", dataset)
        data.setdefault("invocations", {})
        return data
    return {
        "schema_version": "runtime_tools_v1",
        "image_id": image_id,
        "dataset": dataset,
        "invocations": {},
    }


def write_runtime_cache(cache_dir: str, image_id: str, data: Dict[str, Any]) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    path = runtime_cache_path(cache_dir, image_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def tool_signature(tool_call: Dict[str, Any]) -> str:
    tool_name = tool_call.get("tool_name")
    inputs = tool_call.get("inputs")
    if tool_name == "ctclip_location" and isinstance(inputs, dict):
        inputs = {
            key: value
            for key, value in inputs.items()
            if key not in {"options", "option_to_location", "coordinate_transforms"}
        }
    payload = {
        "tool_name": tool_name,
        "inputs": inputs,
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _refresh_location_output_for_current_options(output: Dict[str, Any], tool_call: Dict[str, Any]) -> Dict[str, Any]:
    if output.get("tool_name") != "ctclip_location":
        return output
    inputs = tool_call.get("inputs") if isinstance(tool_call.get("inputs"), dict) else {}
    option_to_location = inputs.get("option_to_location") if isinstance(inputs.get("option_to_location"), dict) else {}
    distributions = output.get("distributions") if isinstance(output.get("distributions"), dict) else {}
    refreshed = dict(output)
    refreshed["tool_call_id"] = tool_call.get("tool_call_id", refreshed.get("tool_call_id"))
    if option_to_location:
        option_scores = {opt: distributions.get(str(location), 0.0) for opt, location in option_to_location.items()}
        refreshed["option_location_scores"] = option_scores
        if option_scores:
            option_max = max(option_scores.values())
            refreshed["recommended_options"] = [
                opt for opt, score in option_scores.items()
                if abs(score - option_max) < 1e-6
            ]
    return refreshed


def execute_tool_call_with_cache(
    tool_call: Dict[str, Any],
    row: Dict[str, Any],
    context: Any,
    cache_dir: str,
) -> Dict[str, Any]:
    image_id = str(row.get("Image ID"))
    dataset = str(row.get("dataset"))
    signature = tool_signature(tool_call)
    cache = read_runtime_cache(cache_dir, image_id, dataset)
    existing = (cache.get("invocations") or {}).get(signature)
    if isinstance(existing, dict):
        output = _refresh_location_output_for_current_options(dict(existing.get("output") or {}), tool_call)
        output = enrich_runtime_tool_output(output)
        output["cache_status"] = "hit"
        output["cache_signature"] = signature
        output["cached_tool_call_id"] = existing.get("tool_call_id", output.get("tool_call_id"))
        output["tool_call_id"] = tool_call.get("tool_call_id", output.get("tool_call_id"))
        return output

    try:
        from tools import run_runtime_tool_call  # noqa: WPS433

        raw_output = run_runtime_tool_call(tool_call, row, context)
        output = strip_leaky_tool_fields(raw_output)
        output = enrich_runtime_tool_output(output)
        status = "success" if output.get("available") else "failed"
        failure_reason = str(output.get("failure_reason", ""))
    except Exception as exc:
        output = {
            "tool_name": tool_call.get("tool_name"),
            "tool_call_id": tool_call.get("tool_call_id"),
            "available": False,
            "failure_reason": str(exc),
        }
        status = "failed"
        failure_reason = str(exc)
    output["cache_status"] = "miss"
    output["cache_signature"] = signature
    cache.setdefault("invocations", {})[signature] = {
        "tool_call_id": tool_call.get("tool_call_id"),
        "tool_name": tool_call.get("tool_name"),
        "status": status,
        "cache_status": "miss",
        "inputs": strip_leaky_tool_fields(tool_call.get("inputs") or {}),
        "output": output,
        "failure_reason": failure_reason,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_runtime_cache(cache_dir, image_id, cache)
    return output
