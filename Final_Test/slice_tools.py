#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Single-slice visual planning, rendering, observation, and cache helpers."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import nibabel as nib
import numpy as np

from config import RAW_CT_ROOT, SUBSET_ROOT, as_str
from Tool_Box.io import safe_json_loads
from t1s_prompt_policies import (
    build_observer_policy,
    build_planner_policy,
    build_task_guidance,
    normalize_question_type,
)


TOTAL_SLICES = 240
SPATIAL_SIZE = 480
DEFAULT_DATA_ROOTS = (
    as_str(RAW_CT_ROOT),
    as_str(SUBSET_ROOT),
)
LEAKY_VISUAL_KEYS = {
    "answer",
    "correct_option",
    "correct_count",
    "answer_value",
    "is_correct",
    "gt_answer",
}


IMAGE_TOOL_REGISTRY = [
    {
        "tool_name": "render_raw_slice",
        "use_when": "Inspect one least-transformed CT-CLIP 240-space axial slice using a single abdomen/liver window without overlays or multi-window montage.",
        "inputs": ["slice_index_240 from candidate_slice_queue"],
        "limits": "One abdomen-window slice only; no overlays, no exact quantitative measurements, no full-volume inference.",
    },
    {
        "tool_name": "render_multi_window_slice",
        "use_when": "Inspect one CT-CLIP 240-space axial slice with soft-tissue, liver/abdomen, and vascular-style windows.",
        "inputs": ["slice_index_240 from candidate_slice_queue"],
        "limits": "One axial slice only; do not infer exact quantitative measurements or full-volume counts from this image.",
    },
    {
        "tool_name": "render_organ_overlay_slice",
        "use_when": "Inspect one CT-CLIP 240-space axial slice with target-organ contour context.",
        "inputs": ["slice_index_240 from candidate_slice_queue", "target_organ"],
        "limits": "Organ contour can be missing or noisy; not a ground-truth lesion annotation.",
    },
]
IMAGE_TOOL_NAMES = {item["tool_name"] for item in IMAGE_TOOL_REGISTRY}


def build_image_tool_registry() -> List[Dict[str, Any]]:
    return json.loads(json.dumps(IMAGE_TOOL_REGISTRY))


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


def _compact_json(data: Any, max_chars: int = 18000) -> str:
    text = json.dumps(data, ensure_ascii=False, indent=2)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...<truncated>"


def build_t1s_planning_prompt(
    question: str,
    inferred_query: Dict[str, Any],
    facts_memory: Dict[str, Any],
    image_tool_registry: List[Dict[str, Any]],
    question_type: str = "",
    preliminary_answer: Optional[Dict[str, Any]] = None,
    visual_history: Optional[Dict[str, Any]] = None,
    used_slice_indices: Optional[List[int]] = None,
    iteration_index: int = 1,
    max_iters: int = 1,
) -> str:
    task_guidance = build_task_guidance(question_type)
    candidates = facts_memory.get("candidate_slice_queue") or []
    used = {int(idx) for idx in (used_slice_indices or []) if idx is not None}
    unused_candidates = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("slice_index_240"))
        except Exception:
            continue
        if idx not in used:
            unused_candidates.append(item)
    return f"""Plan one step of a think-with-one-slice loop for this medical VQA case.

Rules:
- This is iteration {iteration_index} of at most {max_iters}. Each iteration may inspect at most ONE axial CT slice.
- Default to selected_visual_call null. Do not inspect an image merely to confirm a likely answer or make evidence feel more vivid.
- First inspect the compact facts_memory, the preliminary text/tool-only answer, and visual_history. Continue only if there is a concrete assumption whose visual status would materially update the reasoning memory.
- Select image inspection to verify or contradict ONE explicit assumption. The assumption may be a final-option criterion or an intermediate visible/medical premise, such as local morphology, target visibility, relative position, organ boundary context, local contact, or continuity evidence at this slice.
- The selected slice does not need to answer the entire MCQ by itself. For range, continuity, distribution, or option-specific slice questions, use the loop to gather one slice-local piece of evidence at a time.
- Do not inspect a slice for weak confirmation, general confidence boosting, consistency checking, or broad exclusion of hidden findings elsewhere.
- Do not ask the image to provide exact quantitative measurements, full-volume certainty, pathology labels, hidden CSV labels, or facts that are not visually observable on the rendered slice.
- If the uncertainty comes mainly from missing non-visual facts and no candidate slice can test a concrete visible assumption, return selected_visual_call null and explain that in rationale.
- You may choose only one slice_index_240 from unused_candidate_slice_queue. Do not invent slice indices and do not reuse used slices.
- If a question compares multiple option-specific candidates, inspect one candidate per iteration when that candidate can verify/contradict a specific option-related assumption.
- Do not plan more iterations just to accumulate confidence, check representativeness, or exclude hidden features on other slices. Each new iteration must test a distinct assumption or a distinct candidate slice needed by the existing assumption chain.
- Use only one image tool from the registry.
- Prefer render_raw_slice when a direct, minimally transformed abdomen-window view is sufficient.
- Prefer render_multi_window_slice for lesion visibility or attenuation-like questions.
- Prefer render_organ_overlay_slice when organ boundary/location context is important.
- The rendered image is visual evidence only; it must not use hidden CSV labels, ground-truth answers, or lesion annotations.
- If runtime/text memory already answers the question well and image inspection can only add weak/local context, do not call an image tool.
- For follow-up iterations, continue only when prior visual history leaves a concrete unresolved assumption or partial evidence chain that another unused candidate slice can materially update; otherwise stop.
- A good selected call must name the assumption being tested and what image evidence would verify or contradict it.
- Output STRICT JSON only.

{build_planner_policy(question_type)}

Output schema:
{{
  "loop_decision": "continue|stop",
  "selected_visual_call": {{
    "tool_call_id": "<short unique id>",
    "tool_name": "<registry tool_name>",
    "slice_index_240": <integer from unused_candidate_slice_queue>,
    "assumption_role": "option_direct|intermediate_only",
    "purpose": "<why this slice/image can add evidence>",
    "expected_evidence": "<what the VLM should inspect>",
    "assumption_to_verify": "<one concrete assumption this slice can verify>"
  }} | null,
  "rationale": "<brief reason, including why no image is needed if null>",
  "stop_reason": "<filled when loop_decision is stop>",
  "confidence": <0.0-1.0>
}}

Question:
{question}

MCQ options:
{json.dumps(parse_option_texts(question), ensure_ascii=False, indent=2)}

Inferred query:
{json.dumps(inferred_query, ensure_ascii=False, indent=2)}

Task-level evidence prior:
{json.dumps(task_guidance, ensure_ascii=False, indent=2)}

Candidate slice queue:
{json.dumps(candidates, ensure_ascii=False, indent=2)}

Used slice indices:
{json.dumps(sorted(used), ensure_ascii=False)}

Unused candidate slice queue:
{json.dumps(unused_candidates, ensure_ascii=False, indent=2)}

Compact facts_memory:
{_compact_json(facts_memory)}

Preliminary text/tool-only answer:
{json.dumps(preliminary_answer or {}, ensure_ascii=False, indent=2)}

Visual history from earlier T1S iterations:
{_compact_json(visual_history or {}, max_chars=9000)}

Image tool registry:
{json.dumps(image_tool_registry, ensure_ascii=False, indent=2)}
"""


async def plan_t1s_visual_call_with_gpt(
    question: str,
    inferred_query: Dict[str, Any],
    facts_memory: Dict[str, Any],
    image_tool_registry: List[Dict[str, Any]],
    preliminary_answer: Optional[Dict[str, Any]],
    model: str,
    call_gpt: Callable[..., Awaitable[str]],
    enabled: bool,
    visual_history: Optional[Dict[str, Any]] = None,
    used_slice_indices: Optional[List[int]] = None,
    iteration_index: int = 1,
    max_iters: int = 1,
    question_type: str = "",
) -> Dict[str, Any]:
    if not enabled:
        return {
            "selected_visual_call": None,
            "rationale": "T1S visual loop disabled",
            "confidence": 1.0,
            "selector": "none",
        }
    try:
        response = await call_gpt(
            build_t1s_planning_prompt(
                question,
                inferred_query,
                facts_memory,
                image_tool_registry,
                question_type=question_type,
                preliminary_answer=preliminary_answer,
                visual_history=visual_history,
                used_slice_indices=used_slice_indices,
                iteration_index=iteration_index,
                max_iters=max_iters,
            ),
            model=model,
        )
        parsed = safe_json_loads(response) or {}
        call = parsed.get("selected_visual_call")
        if not isinstance(call, dict):
            call = None
        return {
            "selected_visual_call": call,
            "loop_decision": parsed.get("loop_decision", "continue" if call else "stop"),
            "rationale": parsed.get("rationale", ""),
            "stop_reason": parsed.get("stop_reason", ""),
            "confidence": parsed.get("confidence", 0.0),
            "selector": "gpt",
            "raw_response": response,
        }
    except Exception as exc:
        return {
            "selected_visual_call": None,
            "loop_decision": "stop",
            "rationale": "T1S planner failed; no image inspected",
            "stop_reason": "planner_failed",
            "confidence": 0.0,
            "selector": "gpt_failed_no_fallback",
            "selector_error": str(exc),
        }


def normalize_t1s_visual_call(
    raw_call: Optional[Dict[str, Any]],
    facts_memory: Dict[str, Any],
    row: Dict[str, Any],
    inferred_query: Dict[str, Any],
    used_slice_indices: Optional[List[int]] = None,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    if not isinstance(raw_call, dict):
        return None, warnings
    tool_name = str(raw_call.get("tool_name") or "")
    if tool_name not in IMAGE_TOOL_NAMES:
        warnings.append(f"unknown_image_tool:{tool_name}")
        return None, warnings
    try:
        selected_idx = int(round(float(raw_call.get("slice_index_240"))))
    except Exception:
        warnings.append("missing_or_invalid_slice_index_240")
        return None, warnings
    candidates = facts_memory.get("candidate_slice_queue") or []
    allowed = {int(item.get("slice_index_240")) for item in candidates if isinstance(item, dict) and item.get("slice_index_240") is not None}
    if selected_idx not in allowed:
        warnings.append(f"slice_index_not_in_candidate_queue:{selected_idx}")
        return None, warnings
    used = {int(idx) for idx in (used_slice_indices or []) if idx is not None}
    if selected_idx in used:
        warnings.append(f"slice_already_used:{selected_idx}")
        return None, warnings
    raw_id = str(raw_call.get("tool_call_id") or tool_name)
    tool_call_id = re.sub(r"[^a-zA-Z0-9_]+", "_", raw_id).strip("_").lower() or tool_name
    return {
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "slice_index_240": max(0, min(TOTAL_SLICES - 1, selected_idx)),
        "assumption_role": str(raw_call.get("assumption_role") or ""),
        "purpose": str(raw_call.get("purpose") or ""),
        "expected_evidence": str(raw_call.get("expected_evidence") or ""),
        "assumption_to_verify": str(raw_call.get("assumption_to_verify") or ""),
        "inputs": {
            "image_id": str(row.get("Image ID")),
            "dataset": str(row.get("dataset")),
            "question_type": normalize_question_type(row.get("question type")),
            "target_organ": inferred_query.get("target_organ"),
            "lesion_type": inferred_query.get("lesion_type"),
            "question_intent": inferred_query.get("question_intent"),
            "slice_index_240": selected_idx,
        },
    }, warnings


def _candidate_nifti_paths(image_id: str, dataset: str, data_roots: Iterable[str]) -> Iterable[str]:
    for root in data_roots:
        yield os.path.join(str(root), str(dataset), "img", f"{image_id}.nii.gz")


def find_nifti_path(image_id: str, dataset: str, data_roots: Optional[Iterable[str]] = None) -> Optional[str]:
    for path in _candidate_nifti_paths(image_id, dataset, data_roots or DEFAULT_DATA_ROOTS):
        if os.path.exists(path):
            return path
    return None


def center_crop_pad_hwd(data: np.ndarray, target_shape: Tuple[int, int, int], pad_value: float) -> np.ndarray:
    h, w, d = data.shape
    th, tw, td = target_shape
    h_start = max((h - th) // 2, 0)
    h_end = min(h_start + th, h)
    w_start = max((w - tw) // 2, 0)
    w_end = min(w_start + tw, w)
    d_start = max((d - td) // 2, 0)
    d_end = min(d_start + td, d)
    cropped = data[h_start:h_end, w_start:w_end, d_start:d_end]
    pad_h_before = (th - cropped.shape[0]) // 2
    pad_h_after = th - cropped.shape[0] - pad_h_before
    pad_w_before = (tw - cropped.shape[1]) // 2
    pad_w_after = tw - cropped.shape[1] - pad_w_before
    pad_d_before = (td - cropped.shape[2]) // 2
    pad_d_after = td - cropped.shape[2] - pad_d_before
    return np.pad(
        cropped,
        ((pad_h_before, pad_h_after), (pad_w_before, pad_w_after), (pad_d_before, pad_d_after)),
        mode="constant",
        constant_values=pad_value,
    )


def load_aligned_ct_volume_dhw(image_id: str, dataset: str, data_roots: Optional[Iterable[str]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    path = find_nifti_path(image_id, dataset, data_roots=data_roots)
    if not path:
        raise FileNotFoundError(f"Missing CT file for {dataset}/{image_id}")
    nii = nib.load(path)
    data = nii.get_fdata().astype(np.float32)
    aligned = center_crop_pad_hwd(data, (SPATIAL_SIZE, SPATIAL_SIZE, TOTAL_SLICES), pad_value=-1024.0)
    return np.transpose(aligned, (2, 0, 1)), {
        "nifti_path": path,
        "raw_shape": tuple(int(v) for v in data.shape),
        "aligned_shape_dhw": tuple(int(v) for v in np.transpose(aligned, (2, 0, 1)).shape),
        "voxel_spacing": tuple(float(v) for v in nii.header.get_zooms()),
    }


def window_uint8(slice_hu: np.ndarray, wl: float, ww: float) -> np.ndarray:
    low = wl - ww / 2.0
    high = wl + ww / 2.0
    img = np.clip(slice_hu, low, high)
    img = (img - low) / (high - low + 1e-6)
    return (img * 255.0).astype(np.uint8)


def _label_panel(rgb: np.ndarray, label: str) -> np.ndarray:
    out = rgb.copy()
    cv2.rectangle(out, (0, 0), (260, 26), (0, 0, 0), thickness=-1)
    cv2.putText(out, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def render_raw_slice(tool_call: Dict[str, Any], render_dir: str) -> Dict[str, Any]:
    inputs = tool_call["inputs"]
    image_id = str(inputs["image_id"])
    dataset = str(inputs["dataset"])
    idx = int(inputs["slice_index_240"])
    volume_dhw, meta = load_aligned_ct_volume_dhw(image_id, dataset)
    image = window_uint8(volume_dhw[idx], 70.0, 150.0)
    os.makedirs(render_dir, exist_ok=True)
    path = os.path.join(render_dir, f"{image_id}_{tool_call['tool_call_id']}_slice{idx:03d}_raw.png")
    ok = cv2.imwrite(path, image)
    if not ok or not os.path.exists(path) or os.path.getsize(path) == 0:
        raise RuntimeError(f"Failed to write rendered image: {path}")
    return {
        "rendered_image_path": path,
        "render_type": "raw_abdomen_window",
        "slice_index_240": idx,
        "warnings": [],
        "render_metadata": {
            **meta,
            "window": {"name": "abdomen/liver", "level": 70.0, "width": 150.0},
            "overlay_used": False,
            "multi_window": False,
        },
    }


def render_multi_window_slice(tool_call: Dict[str, Any], render_dir: str) -> Dict[str, Any]:
    inputs = tool_call["inputs"]
    image_id = str(inputs["image_id"])
    dataset = str(inputs["dataset"])
    idx = int(inputs["slice_index_240"])
    volume_dhw, meta = load_aligned_ct_volume_dhw(image_id, dataset)
    slice_hu = volume_dhw[idx]
    windows = [
        ("soft tissue WL40 WW400", 40.0, 400.0),
        ("abdomen/liver WL70 WW150", 70.0, 150.0),
        ("vascular WL150 WW500", 150.0, 500.0),
    ]
    panels = []
    for label, wl, ww in windows:
        gray = window_uint8(slice_hu, wl, ww)
        panels.append(_label_panel(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), label))
    image = np.concatenate(panels, axis=1)
    os.makedirs(render_dir, exist_ok=True)
    path = os.path.join(render_dir, f"{image_id}_{tool_call['tool_call_id']}_slice{idx:03d}_multi.png")
    ok = cv2.imwrite(path, image)
    if not ok or not os.path.exists(path) or os.path.getsize(path) == 0:
        raise RuntimeError(f"Failed to write rendered image: {path}")
    return {
        "rendered_image_path": path,
        "render_type": "multi_window",
        "slice_index_240": idx,
        "warnings": [],
        "render_metadata": meta,
    }


def _organ_mask_names(target_organ: str) -> List[str]:
    organ = str(target_organ or "").lower().strip()
    if organ == "kidney":
        return ["kidney", "left kidney", "right kidney"]
    return [organ] if organ else []


def load_aligned_organ_mask_dhw(mask_dir: str, dataset: str, image_id: str, target_organ: str) -> Tuple[Optional[np.ndarray], List[str]]:
    warnings: List[str] = []
    masks = []
    for name in _organ_mask_names(target_organ):
        path = os.path.join(mask_dir, str(dataset), str(image_id), f"{name}.nii.gz")
        if not os.path.exists(path):
            warnings.append(f"missing_mask:{name}")
            continue
        try:
            data = nib.load(path).get_fdata().astype(np.float32)
            aligned = center_crop_pad_hwd(data, (SPATIAL_SIZE, SPATIAL_SIZE, TOTAL_SLICES), pad_value=0.0)
            masks.append(np.transpose(aligned > 0, (2, 0, 1)))
        except Exception as exc:
            warnings.append(f"mask_load_error:{name}:{exc}")
    if not masks:
        return None, warnings
    combined = np.logical_or.reduce(masks)
    return combined.astype(np.uint8), warnings


def render_organ_overlay_slice(tool_call: Dict[str, Any], render_dir: str, mask_dir: str) -> Dict[str, Any]:
    inputs = tool_call["inputs"]
    image_id = str(inputs["image_id"])
    dataset = str(inputs["dataset"])
    target_organ = str(inputs.get("target_organ") or "")
    idx = int(inputs["slice_index_240"])
    volume_dhw, meta = load_aligned_ct_volume_dhw(image_id, dataset)
    gray = window_uint8(volume_dhw[idx], 40.0, 400.0)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    mask_dhw, warnings = load_aligned_organ_mask_dhw(mask_dir, dataset, image_id, target_organ)
    overlay_used = False
    if mask_dhw is not None and int(mask_dhw[idx].sum()) > 0:
        contours, _ = cv2.findContours(mask_dhw[idx].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, contours, -1, (0, 255, 0), thickness=2)
        overlay_used = True
    else:
        warnings.append("empty_or_missing_mask_on_selected_slice")
    rgb = _label_panel(rgb, f"soft tissue + {target_organ} contour | slice {idx}")
    os.makedirs(render_dir, exist_ok=True)
    path = os.path.join(render_dir, f"{image_id}_{tool_call['tool_call_id']}_slice{idx:03d}_overlay.png")
    ok = cv2.imwrite(path, rgb)
    if not ok or not os.path.exists(path) or os.path.getsize(path) == 0:
        raise RuntimeError(f"Failed to write rendered image: {path}")
    return {
        "rendered_image_path": path,
        "render_type": "organ_overlay",
        "slice_index_240": idx,
        "overlay_used": overlay_used,
        "warnings": warnings,
        "render_metadata": meta,
    }


def strip_leaky_visual_fields(value: Any) -> Any:
    return _strip_leaky_visual_fields(value)


def _strip_leaky_visual_fields(value: Any, parent_key: str = "") -> Any:
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            key_str = str(key)
            preserve_model_answer = parent_key == "current_judgment" and key_str == "answer"
            if not preserve_model_answer and (
                key_str in LEAKY_VISUAL_KEYS or key_str.endswith("_error_to_gt")
            ):
                continue
            cleaned[key_str] = _strip_leaky_visual_fields(item, parent_key=key_str)
        return cleaned
    if isinstance(value, list):
        return [_strip_leaky_visual_fields(item, parent_key=parent_key) for item in value]
    return value


def t1s_cache_path(cache_dir: str, image_id: str) -> str:
    return os.path.join(cache_dir, f"{image_id}.json")


def read_t1s_cache(cache_dir: str, image_id: str, dataset: str) -> Dict[str, Any]:
    path = t1s_cache_path(cache_dir, image_id)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data.setdefault("schema_version", "t1s_loop_v1")
            data.setdefault("image_id", image_id)
            data.setdefault("dataset", dataset)
            data.setdefault("invocations", {})
            return data
    return {
        "schema_version": "t1s_loop_v1",
        "image_id": image_id,
        "dataset": dataset,
        "invocations": {},
    }


def write_t1s_cache(cache_dir: str, image_id: str, data: Dict[str, Any]) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    with open(t1s_cache_path(cache_dir, image_id), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def t1s_signature(tool_call: Dict[str, Any], observation_prompt_version: str = "v2_6_measurement_recognition_tighter_prompt") -> str:
    payload = {
        "tool_name": tool_call.get("tool_name"),
        "inputs": tool_call.get("inputs"),
        "observation_prompt_version": observation_prompt_version,
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def encode_image_to_base64(path: str) -> str:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError(f"Failed to encode image: {path}")
    return base64.b64encode(buf).decode("utf-8")


def build_visual_observation_prompt(
    question: str,
    facts_memory: Dict[str, Any],
    tool_call: Dict[str, Any],
    render_result: Dict[str, Any],
    question_type: str = "",
) -> str:
    compact_memory = dict(facts_memory)
    visual_history = compact_memory.get("visual_memory") if isinstance(compact_memory.get("visual_memory"), dict) else {}
    task_guidance = build_task_guidance(question_type or (tool_call.get("inputs") or {}).get("question_type"))
    return f"""Inspect the provided single CT slice and update the reasoning memory for this MCQ.

Rules:
- You are shown exactly one rendered axial CT slice from CT-CLIP 240-space.
- Use the image only as additional evidence; do not invent findings not visible on this slice.
- This is one iteration of a multi-turn single-slice loop. Judge the selected assumption on this slice, then update the assumptions/evidence memory.
- Do not infer exact quantitative values, sizes, volumes, counts, or measurements from pixel brightness or a single rendered slice.
- Organ contours, if present, are automatically generated and may be noisy; treat them as context, not ground truth.
- If the image is not helpful or the target is not visible, say so explicitly.
- First evaluate selected_visual_call.assumption_to_verify. Be willing to mark it verified or contradicted when the slice visibly supports that local assumption, even if the whole MCQ answer remains uncertain.
- Use assumption_test.result="verified" when the visible slice evidence supports the assumption, "contradicted" when it refutes the assumption, and "unresolved" when the target/relationship is not visible or cannot be judged.
- Use "critical_intermediate" when this slice verifies an important premise needed for final reasoning, but not the whole MCQ answer by itself.
- Use "weak_context" for background context; use "none" when the image adds no useful evidence.
- Use assumption_test.scope="single_slice_local" when the assumption is answerable on this slice, "multi_slice_partial" when it is one useful piece of a multi-slice evidence chain, and "insufficient" when this slice cannot judge it.
- Set current_judgment.answer to an MCQ option only when this slice plus prior visual history provides enough direct evidence for that option. It is acceptable for current_judgment.answer to be "unclear" while assumption_test is verified or contradicted.
- Use "unclear" for current_judgment.answer when the local evidence updates assumptions but does not by itself decide the final MCQ.
- Treat current_judgment.answer as a conservative one-slice judgment, not as exhaustive full-volume certainty.
- If this slice does not resolve the case but leaves a concrete assumption that another unused candidate slice could materially update, set next_step_suggestion to "inspect_another_slice" and list that assumption under unresolved_assumptions. Otherwise use "stop" or "no_image_helpful".
- Be conservative about scope: single-slice visual evidence must not claim global, quantitative, dynamic, composite-medical-standard, or cross-slice conclusions.
- Do not use hidden CSV labels or ground-truth answers.
- Output STRICT JSON only.

{build_observer_policy(question_type or (tool_call.get("inputs") or {}).get("question_type"))}

Output schema:
{{
  "assumption_test": {{
    "assumption": "<the selected assumption being tested>",
    "result": "verified|contradicted|unresolved",
    "confidence": <0.0-1.0>,
    "answer_relevance": "direct_option|critical_intermediate|weak_context|none",
    "scope": "single_slice_local|multi_slice_partial|insufficient",
    "rationale": "<brief visible evidence and limitation>"
  }},
  "visual_evidence": ["visible image-grounded finding", "..."],
  "visual_assumptions": ["assumption or limitation", "..."],
  "verified_assumptions": ["assumption supported by this slice", "..."],
  "contradicted_assumptions": ["assumption contradicted by this slice", "..."],
  "unresolved_assumptions": ["assumption that remains unresolved and may need another slice", "..."],
  "option_support": {{"A": "<supports|contradicts|unclear + short reason>"}},
  "current_judgment": {{
    "answer": "<ONE option letter from the MCQ options, or 'unclear' if image is insufficient>",
    "confidence": <0.0-1.0>,
    "rationale": "<brief image + memory grounded reasoning>"
  }},
  "uncertainty": "<low|medium|high plus short reason>",
  "next_step_suggestion": "stop|inspect_another_slice|no_image_helpful"
}}

Question:
{question}

Task-level evidence prior:
{json.dumps(task_guidance, ensure_ascii=False, indent=2)}

Selected visual call:
{json.dumps(tool_call, ensure_ascii=False, indent=2)}

Render result:
{json.dumps({k: v for k, v in render_result.items() if k != "rendered_image_path"}, ensure_ascii=False, indent=2)}

Visual history before this slice:
{_compact_json(visual_history, max_chars=9000)}

Compact facts_memory before image:
{_compact_json(compact_memory, max_chars=16000)}
"""


async def call_vlm_with_image(
    prompt: str,
    image_path: str,
    model: str,
    client: Any,
    system_prompt: str,
    max_retries: int,
    base_backoff: float,
    sleep_fn: Callable[[float], Awaitable[None]],
    random_fn: Callable[[], float],
) -> str:
    content = [
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encode_image_to_base64(image_path)}"},
        },
    ]
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
            )
            return response.choices[0].message.content
        except Exception:
            if attempt == max_retries - 1:
                raise
            await sleep_fn(base_backoff * (2**attempt) + random_fn())
    raise RuntimeError("unreachable")


async def execute_t1s_visual_iteration_with_cache(
    tool_call: Dict[str, Any],
    row: Dict[str, Any],
    facts_memory: Dict[str, Any],
    question: str,
    cache_dir: str,
    render_dir: str,
    mask_dir: str,
    vision_model: str,
    client: Any,
    system_prompt: str,
    max_retries: int,
    base_backoff: float,
    sleep_fn: Callable[[float], Awaitable[None]],
    random_fn: Callable[[], float],
) -> Dict[str, Any]:
    image_id = str(row.get("Image ID"))
    dataset = str(row.get("dataset"))
    signature = t1s_signature(tool_call)
    cache = read_t1s_cache(cache_dir, image_id, dataset)
    existing = (cache.get("invocations") or {}).get(signature)
    if isinstance(existing, dict):
        result = dict(existing.get("output") or {})
        result["cache_status"] = "hit"
        result["cache_signature"] = signature
        return result

    case_render_dir = os.path.join(render_dir, str(dataset), str(image_id))
    try:
        if tool_call["tool_name"] == "render_raw_slice":
            render_result = render_raw_slice(tool_call, case_render_dir)
        elif tool_call["tool_name"] == "render_multi_window_slice":
            render_result = render_multi_window_slice(tool_call, case_render_dir)
        elif tool_call["tool_name"] == "render_organ_overlay_slice":
            render_result = render_organ_overlay_slice(tool_call, case_render_dir, mask_dir=mask_dir)
        else:
            raise ValueError(f"unsupported_image_tool:{tool_call['tool_name']}")
        prompt = build_visual_observation_prompt(
            question,
            facts_memory,
            tool_call,
            render_result,
            question_type=str(row.get("question type") or ""),
        )
        raw_response = await call_vlm_with_image(
            prompt=prompt,
            image_path=render_result["rendered_image_path"],
            model=vision_model,
            client=client,
            system_prompt=system_prompt,
            max_retries=max_retries,
            base_backoff=base_backoff,
            sleep_fn=sleep_fn,
            random_fn=random_fn,
        )
        observation = safe_json_loads(raw_response) or {"raw_response": raw_response}
        output = {
            "available": True,
            "failure_reason": "",
            "tool_call": tool_call,
            "render_result": render_result,
            "observation": strip_leaky_visual_fields(observation),
            "raw_response": raw_response,
        }
        status = "success"
        failure_reason = ""
    except Exception as exc:
        output = {
            "available": False,
            "failure_reason": str(exc),
            "tool_call": tool_call,
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
        "inputs": strip_leaky_visual_fields(tool_call.get("inputs") or {}),
        "output": strip_leaky_visual_fields(output),
        "failure_reason": failure_reason,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_t1s_cache(cache_dir, image_id, cache)
    return output


def build_visual_memory_for_prompt(
    visual_result: Optional[Dict[str, Any]],
    iteration_index: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    if not isinstance(visual_result, dict) or not visual_result.get("available"):
        return None
    observation = visual_result.get("observation") if isinstance(visual_result.get("observation"), dict) else {}
    render_result = visual_result.get("render_result") if isinstance(visual_result.get("render_result"), dict) else {}
    tool_call = visual_result.get("tool_call") if isinstance(visual_result.get("tool_call"), dict) else {}
    current_judgment = observation.get("current_judgment") if isinstance(observation.get("current_judgment"), dict) else {}
    if "answer" not in current_judgment:
        current_judgment = {**current_judgment, "answer": "unclear"}
    assumption_test = observation.get("assumption_test") if isinstance(observation.get("assumption_test"), dict) else {}
    if "result" not in assumption_test:
        assumption_test = {**assumption_test, "result": "unresolved"}
    memory = {
        "available": True,
        "iteration_index": iteration_index,
        "tool_name": tool_call.get("tool_name"),
        "slice_index_240": tool_call.get("slice_index_240") or (tool_call.get("inputs") or {}).get("slice_index_240"),
        "render_type": render_result.get("render_type"),
        "render_warnings": render_result.get("warnings"),
        "assumption_test": assumption_test,
        "visual_evidence": observation.get("visual_evidence"),
        "visual_assumptions": observation.get("visual_assumptions"),
        "verified_assumptions": observation.get("verified_assumptions"),
        "contradicted_assumptions": observation.get("contradicted_assumptions"),
        "unresolved_assumptions": observation.get("unresolved_assumptions"),
        "option_support": observation.get("option_support"),
        "current_judgment": current_judgment,
        "uncertainty": observation.get("uncertainty"),
        "next_step_suggestion": observation.get("next_step_suggestion"),
        "evidence_id": f"t1s:{tool_call.get('tool_call_id', 'visual')}",
        "cache_status": visual_result.get("cache_status"),
    }
    return {key: value for key, value in memory.items() if value is not None}


def _float_or_none(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _assumption_evidence_summary(iterations: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    actionable = []
    for item in iterations:
        test = item.get("assumption_test") if isinstance(item.get("assumption_test"), dict) else {}
        result = str(test.get("result") or "unresolved").strip().lower()
        relevance = str(test.get("answer_relevance") or "none").strip().lower()
        counts[result] = counts.get(result, 0) + 1
        confidence = _float_or_none(test.get("confidence"))
        if (
            result in {"verified", "contradicted"}
            and relevance in {"direct_option", "critical_intermediate"}
            and confidence is not None
            and confidence >= 0.6
        ):
            actionable.append(
                {
                    "iteration_index": item.get("iteration_index"),
                    "slice_index_240": item.get("slice_index_240"),
                    "result": result,
                    "confidence": confidence,
                    "answer_relevance": relevance,
                    "scope": test.get("scope"),
                    "assumption": test.get("assumption"),
                    "rationale": test.get("rationale"),
                    "evidence_id": item.get("evidence_id"),
                }
            )
    return {
        "assumption_test_counts": counts,
        "actionable_assumption_evidence": actionable,
        "has_actionable_assumption_evidence": bool(actionable),
    }


def build_visual_loop_memory(iteration_memories: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    iterations = [item for item in iteration_memories if isinstance(item, dict) and item.get("available")]
    if not iterations:
        return None
    latest = iterations[-1]
    used_slice_indices = []
    for item in iterations:
        try:
            used_slice_indices.append(int(item.get("slice_index_240")))
        except Exception:
            continue
    evidence_summary = _assumption_evidence_summary(iterations)
    return {
        "available": True,
        "mode": "multi_turn_single_slice",
        "iteration_count": len(iterations),
        "used_slice_indices": used_slice_indices,
        "iterations": iterations,
        "current_state": {
            "current_judgment": latest.get("current_judgment", {"answer": "unclear"}),
            "uncertainty": latest.get("uncertainty"),
            "next_step_suggestion": latest.get("next_step_suggestion"),
            "open_unresolved_assumptions": latest.get("unresolved_assumptions"),
            "assumption_test_summary": evidence_summary,
            "actionable_assumption_evidence": evidence_summary.get("actionable_assumption_evidence", []),
        },
        "current_judgment": latest.get("current_judgment", {"answer": "unclear"}),
        "assumption_test_summary": evidence_summary,
        "uncertainty": latest.get("uncertainty"),
        "next_step_suggestion": latest.get("next_step_suggestion"),
        "evidence_id": "t1s:loop",
    }
