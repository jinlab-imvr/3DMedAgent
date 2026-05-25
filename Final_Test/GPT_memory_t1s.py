#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Final_Test runner with text memory, runtime CT-CLIP tools, and optional T1S."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from config import (  # noqa: E402
    CLIP_DETAIL_DIR,
    CLIP_DETAIL_SLICE_DIR,
    CLIP_EMBEDDING_DIR,
    CLIP_GLOBAL_DIR,
    CTCLIP_MODEL_PATH,
    MASK_DIR,
    RAW_CT_ROOT,
    REPORT_DIR,
    RUNTIME_CACHE_DIR,
    T1S_CACHE_DIR,
    T1S_RENDER_DIR,
    T1S_SAVE_DIR,
    VQA_PATH,
    as_str,
)
from pipeline_utils import (  # noqa: E402
    BASE_BACKOFF,
    MAX_CONCURRENCY,
    MAX_RETRIES,
    MODEL_NAME,
    SYSTEM_PROMPT,
    TARGET_SUBTYPES,
    aclient,
    call_gpt,
    load_existing_records,
    make_base_record,
    normalized_final_answer,
    parse_answer_response,
    parse_target_subtypes,
    save_record,
    select_target_rows,
)
from Tool_Box.io import read_report_safely, safe_json_loads  # noqa: E402
from memory import (  # noqa: E402
    MemorySourcePaths,
    build_answer_prompt,
    build_compact_facts_memory,
    build_facts_memory,
    build_query_normalization_prompt,
    build_reasoning_memory_prompt,
    coerce_inferred_query,
    infer_query_from_question_rule,
    validate_facts_memory,
)
from runtime_tools import (  # noqa: E402
    build_tool_registry,
    execute_tool_call_with_cache,
    normalize_tool_calls,
    plan_runtime_tool_calls_with_gpt,
)
from t1s_prompt_policies import (  # noqa: E402
    build_final_policy,
    build_task_guidance,
    build_text_answer_policy,
)
from slice_tools import (  # noqa: E402
    build_image_tool_registry,
    build_visual_loop_memory,
    build_visual_memory_for_prompt,
    execute_t1s_visual_iteration_with_cache,
    normalize_t1s_visual_call,
    plan_t1s_visual_call_with_gpt,
)

def collect_existing_keys(save_dir: str) -> set[tuple[int, str]]:
    keys: set[tuple[int, str]] = set()
    if not save_dir or not os.path.isdir(save_dir):
        return keys
    for path in Path(save_dir).glob("*.json"):
        for record in load_existing_records(str(path)):
            try:
                keys.add((int(record.get("case_idx")), str(record.get("question_subtype", ""))))
            except Exception:
                continue
    return keys


def apply_text_answer_prompt_overrides(prompt: str, question_type: str) -> str:
    normalized = str(question_type or "").strip().lower()
    if normalized != "recognition":
        return prompt
    replacements = {
        "- Use the inferred query, option evidence, CT-CLIP evidence, structured report evidence, and candidate slice descriptions.": (
            "- For recognition, use the inferred query, MCQ option text, and CT-CLIP volume-level global evidence "
            "(clip_global / clip_global_matrix) for the present/absent decision."
        ),
        "- Candidate slices are future visual-verification pointers only; treat them as text evidence from their listed source.": (
            "- For recognition, candidate slices, option_evidence slice scores, CT-CLIP detail sections/slices, "
            "and top slices are localization/debug context only; do not use them to decide present/absent."
        ),
        "- Candidate slices are not images; they are textual pointers for possible future verification.": (
            "- For recognition, candidate slices and CT-CLIP detail/slice fields are localization/debug context only; ignore them for present/absent."
        ),
    }
    for old, new in replacements.items():
        prompt = prompt.replace(old, new)
    return prompt


async def normalize_query(question: str, normalizer: str, dry_run: bool, model: str) -> Dict[str, Any]:
    if dry_run or normalizer == "rule":
        return infer_query_from_question_rule(question)
    raw_response = await call_gpt(build_query_normalization_prompt(question), model=model)
    parsed = safe_json_loads(raw_response) or {}
    parsed["normalizer_raw_response"] = raw_response
    return coerce_inferred_query(parsed, question)


def build_t1s_answer_prompt(
    question: str,
    memory: Dict[str, Any],
    reasoning_memory: Optional[Any] = None,
    question_type: str = "",
) -> str:
    prompt = build_answer_prompt(question, memory, reasoning_memory=reasoning_memory)
    prompt = apply_text_answer_prompt_overrides(prompt, question_type)
    if question_type:
        prompt += "\n\n" + build_final_policy(question_type, has_visual_memory=bool(memory.get("visual_memory")))
    text_policy = build_text_answer_policy(question_type)
    if text_policy:
        prompt += "\n\n" + text_policy
    if memory.get("pre_t1s_reasoning"):
        prompt = prompt.replace(
            "Answer the multiple-choice question using only the provided leak-safe memory.",
            "Answer the multiple-choice question using only the provided leak-safe memory. pre_t1s_reasoning is a prior text/tool-only judgment; use it as a baseline and revise it only if later memory adds stronger evidence.",
        )
    if memory.get("visual_memory"):
        prompt = prompt.replace(
            "- You are NOT given images in this step.",
            "- You are NOT given images in this step; however, visual_memory may contain a cached one-slice VLM observation from an earlier T1S step.",
        )
        prompt = prompt.replace(
            "- Candidate slices are future visual-verification pointers only; treat them as text evidence from their listed source.",
            "- Candidate slices are future visual-verification pointers only unless visual_memory explicitly reports an inspected slice observation.",
        )
    return prompt


def compact_answer_for_memory(parsed: Dict[str, Any], raw_response: str) -> Dict[str, Any]:
    evidence_ids_used = parsed.get("evidence_ids_used", [])
    slices_relevant = parsed.get("slice_candidates_relevant", [])
    assumptions = parsed.get("assumptions", [])
    analysis = str(parsed.get("analysis", raw_response or ""))[:1800]
    return {
        "analysis": analysis,
        "final_answer": normalized_final_answer(parsed.get("final_answer", "")),
        "evidence_ids_used": evidence_ids_used if isinstance(evidence_ids_used, list) else [str(evidence_ids_used)],
        "slice_candidates_relevant": slices_relevant if isinstance(slices_relevant, list) else [slices_relevant],
        "uncertainty": str(parsed.get("uncertainty", "")),
        "assumptions": assumptions if isinstance(assumptions, list) else [str(assumptions)],
    }


def apply_answer_to_record(record: Dict[str, Any], answer: Dict[str, Any], raw_response: str = "") -> None:
    record["GPT_raw_result"] = answer.get("analysis", raw_response)
    record["GPT_summarized_result"] = normalized_final_answer(answer.get("final_answer", ""))
    record["evidence_ids_used"] = answer.get("evidence_ids_used", [])
    record["slice_candidates_relevant"] = answer.get("slice_candidates_relevant", [])
    record["uncertainty"] = str(answer.get("uncertainty", ""))
    record["assumptions"] = answer.get("assumptions", [])


def visual_judgment_fields(visual_memory: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(visual_memory, dict):
        return {
            "answer": None,
            "confidence": None,
            "uncertainty": None,
        }
    current = visual_memory.get("current_judgment")
    current = current if isinstance(current, dict) else {}
    return {
        "answer": current.get("answer"),
        "confidence": current.get("confidence"),
        "uncertainty": visual_memory.get("uncertainty"),
    }


def is_visual_judgment_unclear(visual_memory: Optional[Dict[str, Any]]) -> bool:
    fields = visual_judgment_fields(visual_memory)
    answer = str(fields.get("answer") or "").strip().lower()
    return answer in {"", "none", "null", "unclear", "unknown", "n/a"}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _visual_iterations(visual_memory: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(visual_memory, dict):
        return []
    iterations = visual_memory.get("iterations")
    if isinstance(iterations, list):
        return [item for item in iterations if isinstance(item, dict)]
    return [visual_memory] if visual_memory.get("available") else []


def visual_assumption_actionability(visual_memory: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    relevance_counts: Dict[str, int] = {}
    actionable = []
    for item in _visual_iterations(visual_memory):
        test = item.get("assumption_test") if isinstance(item.get("assumption_test"), dict) else {}
        result = str(test.get("result") or "unresolved").strip().lower()
        relevance = str(test.get("answer_relevance") or "none").strip().lower()
        counts[result] = counts.get(result, 0) + 1
        relevance_counts[relevance] = relevance_counts.get(relevance, 0) + 1
        confidence = _as_float(test.get("confidence"), default=0.0)
        if (
            result in {"verified", "contradicted"}
            and relevance in {"direct_option", "critical_intermediate"}
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
    current = visual_judgment_fields(visual_memory)
    current_answer = str(current.get("answer") or "").strip().lower()
    current_confidence = _as_float(current.get("confidence"), default=0.0)
    has_concrete_visual_answer = current_answer not in {"", "none", "null", "unclear", "unknown", "n/a"}
    return {
        "assumption_test_counts": counts,
        "assumption_relevance_counts": relevance_counts,
        "actionable_assumption_evidence": actionable,
        "has_actionable_assumption_evidence": bool(actionable),
        "has_concrete_visual_answer": has_concrete_visual_answer,
        "concrete_visual_answer_confidence": current_confidence if has_concrete_visual_answer else None,
        "has_actionable_visual_evidence": bool(actionable) or (has_concrete_visual_answer and current_confidence >= 0.5),
    }


def visual_memory_requests_another_slice(visual_memory: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(visual_memory, dict):
        return False
    suggestion = str(visual_memory.get("next_step_suggestion") or "").strip().lower()
    if suggestion in {"stop", "no_image_helpful"}:
        return False
    if suggestion == "inspect_another_slice":
        return True
    current_state = visual_memory.get("current_state") if isinstance(visual_memory.get("current_state"), dict) else {}
    unresolved = current_state.get("open_unresolved_assumptions")
    if unresolved is None and isinstance(visual_memory.get("iterations"), list) and visual_memory["iterations"]:
        latest = visual_memory["iterations"][-1]
        if isinstance(latest, dict):
            unresolved = latest.get("unresolved_assumptions")
    return is_visual_judgment_unclear(visual_memory) and bool(unresolved)


async def process_one(
    row: Dict[str, Any],
    args: argparse.Namespace,
    paths: MemorySourcePaths,
    runtime_context_state: Dict[str, Any],
    runtime_lock: asyncio.Lock,
    t1s_lock: asyncio.Lock,
    semaphore: asyncio.Semaphore,
    file_lock: asyncio.Lock,
) -> Dict[str, Any]:
    async with semaphore:
        record = make_base_record(row)
        question = str(row.get("multiple-choice question", ""))
        question_type = str(row.get("question type", ""))
        t1s_task_guidance = build_task_guidance(question_type)
        report_path = os.path.join(args.report_dir, f"{row['Image ID']}_report.csv")
        report = read_report_safely(report_path, log_fn=print)

        if not os.path.exists(report_path):
            record["GPT_raw_result"] = "SKIP: missing_report_file"
            if args.save_skips:
                await save_record(args.save_dir, record, file_lock)
            return record
        if not isinstance(report, str) or not report.strip():
            record["GPT_raw_result"] = "SKIP: empty_or_unreadable_report"
            if args.save_skips:
                await save_record(args.save_dir, record, file_lock)
            return record

        try:
            inferred_query = await normalize_query(
                question,
                normalizer=args.normalizer,
                dry_run=args.dry_run,
                model=args.model,
            )
        except Exception as exc:
            inferred_query = infer_query_from_question_rule(question)
            inferred_query["normalizer"] = "rule_after_error"
            inferred_query["normalizer_error"] = str(exc)

        baseline_verbose_memory = build_facts_memory(
            row=row,
            report=report,
            inferred_query=inferred_query,
            paths=paths,
            runtime_tool=None,
        )
        baseline_facts_memory = build_compact_facts_memory(baseline_verbose_memory)

        tool_registry: List[Dict[str, Any]] = []
        selected_tool_calls: List[Dict[str, Any]] = []
        tool_selection: Dict[str, Any] = {
            "selected_tool_call_ids": [],
            "selected_tool_calls": [],
            "rationale": "runtime tools disabled",
            "confidence": 1.0,
            "selector": "none",
        }
        runtime_tools: List[Dict[str, Any]] = []
        verbose_memory = baseline_verbose_memory
        facts_memory = baseline_facts_memory
        preliminary_answer: Optional[Dict[str, Any]] = None
        preliminary_answer_raw = ""

        if args.include_runtime_tools:
            tool_registry = build_tool_registry()
            if args.dry_run:
                tool_selection = {
                    "selected_tool_call_ids": [],
                    "selected_tool_calls": [],
                    "rationale": "dry run: baseline memory and registry generated; no planner or tool execution performed",
                    "confidence": 1.0,
                    "selector": "dry_run",
                    "planner_memory_source": "compact_facts_memory_without_runtime",
                }
            else:
                raw_tool_plan = await plan_runtime_tool_calls_with_gpt(
                    question=question,
                    inferred_query=inferred_query,
                    facts_memory=baseline_facts_memory,
                    tool_registry=tool_registry,
                    selector=args.tool_selector,
                    model=args.model,
                    call_gpt=call_gpt,
                )
                raw_calls = raw_tool_plan.get("selected_tool_calls") or []
                selected_tool_calls, validation_warnings = normalize_tool_calls(raw_calls, row, inferred_query)
                tool_selection = {
                    "selected_tool_call_ids": [call["tool_call_id"] for call in selected_tool_calls],
                    "selected_tool_calls": selected_tool_calls,
                    "raw_selected_tool_calls": raw_calls,
                    "validation_warnings": validation_warnings,
                    "rationale": raw_tool_plan.get("rationale", ""),
                    "confidence": raw_tool_plan.get("confidence", 0.0),
                    "selector": raw_tool_plan.get("selector", args.tool_selector),
                    "planner_memory_source": "compact_facts_memory_without_runtime",
                }
                if raw_tool_plan.get("raw_response"):
                    tool_selection["raw_response"] = raw_tool_plan["raw_response"]
                if raw_tool_plan.get("selector_error"):
                    tool_selection["selector_error"] = raw_tool_plan["selector_error"]

                if selected_tool_calls:
                    async with runtime_lock:
                        if runtime_context_state.get("context") is None:
                            from tools import CTClipRuntimeContext  # noqa: WPS433

                            runtime_context_state["context"] = CTClipRuntimeContext(
                                model_path=as_str(CTCLIP_MODEL_PATH),
                                embedding_dir=as_str(CLIP_EMBEDDING_DIR),
                                mask_dir=args.mask_dir,
                                data_root=as_str(RAW_CT_ROOT),
                                report_dir=args.report_dir,
                                clip_detail_dir=args.clip_detail_dir,
                                device=runtime_context_state.get("device", "cuda:0"),
                            )
                        runtime_context = runtime_context_state["context"]
                        for tool_call in selected_tool_calls:
                            runtime_tools.append(execute_tool_call_with_cache(
                                tool_call,
                                row,
                                runtime_context,
                                args.runtime_cache_dir,
                            ))
                    runtime_tool = {
                        "tool_name": "runtime_tool_bundle",
                        "available": any(tool.get("available") for tool in runtime_tools),
                        "runtime_tools": runtime_tools,
                        "tool_selection": tool_selection,
                    }
                    verbose_memory = build_facts_memory(
                        row=row,
                        report=report,
                        inferred_query=inferred_query,
                        paths=paths,
                        runtime_tool=runtime_tool,
                    )
                    facts_memory = build_compact_facts_memory(verbose_memory)

            verbose_memory["runtime_tool_registry"] = tool_registry
            verbose_memory["runtime_tool_candidates"] = selected_tool_calls
            verbose_memory["runtime_tool_selection"] = tool_selection

        if args.include_t1s and not args.dry_run and not args.memory_only:
            try:
                preliminary_answer_raw = await call_gpt(
                    build_t1s_answer_prompt(
                        question,
                        facts_memory,
                        reasoning_memory=None,
                        question_type=question_type,
                    ),
                    model=args.model,
                )
                preliminary_parsed = parse_answer_response(preliminary_answer_raw)
                preliminary_answer = compact_answer_for_memory(preliminary_parsed, preliminary_answer_raw)
                facts_memory = dict(facts_memory)
                facts_memory["pre_t1s_reasoning"] = preliminary_answer
                verbose_memory["pre_t1s_answer"] = {
                    **preliminary_answer,
                    "raw_response": preliminary_answer_raw,
                }
            except Exception as exc:
                preliminary_answer = None
                preliminary_answer_raw = ""
                record["pre_t1s_answer"] = {"error": str(exc)}
                verbose_memory["pre_t1s_answer"] = {"error": str(exc)}

        image_tool_registry = build_image_tool_registry() if args.include_t1s else []
        t1s_selection: Dict[str, Any] = {
            "selected_visual_call": None,
            "rationale": "T1S disabled",
            "confidence": 1.0,
            "selector": "none",
        }
        t1s_selections: List[Dict[str, Any]] = []
        visual_iteration_memory: Optional[Dict[str, Any]] = None
        visual_iteration_memories: List[Dict[str, Any]] = []
        visual_results: List[Dict[str, Any]] = []
        used_slice_indices: List[int] = []
        t1s_loop_stop_reason = "T1S disabled"

        if args.include_t1s:
            if args.dry_run:
                t1s_selection = {
                    "selected_visual_call": None,
                    "rationale": "dry run: T1S registry generated; no image planner/render/VLM call performed",
                    "confidence": 1.0,
                    "selector": "dry_run",
                }
                t1s_loop_stop_reason = "dry_run"
            else:
                max_iters = max(1, int(getattr(args, "t1s_max_iters", 1)))
                t1s_loop_stop_reason = "max_iters_reached"
                for iteration_index in range(1, max_iters + 1):
                    raw_t1s_plan = await plan_t1s_visual_call_with_gpt(
                        question=question,
                        inferred_query=inferred_query,
                        facts_memory=facts_memory,
                        image_tool_registry=image_tool_registry,
                        preliminary_answer=preliminary_answer,
                        model=args.model,
                        call_gpt=call_gpt,
                        enabled=True,
                        visual_history=visual_iteration_memory,
                        used_slice_indices=used_slice_indices,
                        iteration_index=iteration_index,
                        max_iters=max_iters,
                        question_type=question_type,
                    )
                    visual_call, visual_warnings = normalize_t1s_visual_call(
                        raw_t1s_plan.get("selected_visual_call"),
                        facts_memory,
                        row,
                        inferred_query,
                        used_slice_indices=used_slice_indices,
                    )
                    t1s_selection = {
                        "iteration_index": iteration_index,
                        "selected_visual_call": visual_call,
                        "raw_selected_visual_call": raw_t1s_plan.get("selected_visual_call"),
                        "validation_warnings": visual_warnings,
                        "loop_decision": raw_t1s_plan.get("loop_decision", "continue" if visual_call else "stop"),
                        "rationale": raw_t1s_plan.get("rationale", ""),
                        "stop_reason": raw_t1s_plan.get("stop_reason", ""),
                        "confidence": raw_t1s_plan.get("confidence", 0.0),
                        "selector": raw_t1s_plan.get("selector", "gpt"),
                        "planner_memory_source": "compact_facts_memory_after_runtime_tools_and_visual_history",
                    }
                    if raw_t1s_plan.get("raw_response"):
                        t1s_selection["raw_response"] = raw_t1s_plan["raw_response"]
                    if raw_t1s_plan.get("selector_error"):
                        t1s_selection["selector_error"] = raw_t1s_plan["selector_error"]
                    t1s_selections.append(t1s_selection)

                    if not visual_call:
                        t1s_loop_stop_reason = (
                            t1s_selection.get("stop_reason")
                            or "planner_returned_no_valid_visual_call"
                        )
                        break

                    visual_result = await execute_t1s_visual_iteration_with_cache(
                        tool_call=visual_call,
                        row=row,
                        facts_memory=facts_memory,
                        question=question,
                        cache_dir=args.t1s_cache_dir,
                        render_dir=args.t1s_render_dir,
                        mask_dir=args.mask_dir,
                        vision_model=args.vision_model,
                        client=aclient,
                        system_prompt=SYSTEM_PROMPT,
                        max_retries=MAX_RETRIES,
                        base_backoff=BASE_BACKOFF,
                        sleep_fn=asyncio.sleep,
                        random_fn=random.random,
                    )
                    visual_results.append(visual_result)
                    compact_iteration = build_visual_memory_for_prompt(visual_result, iteration_index=iteration_index)
                    if not compact_iteration:
                        t1s_loop_stop_reason = "visual_iteration_failed_or_unavailable"
                        break

                    visual_iteration_memories.append(compact_iteration)
                    try:
                        used_slice_indices.append(int(compact_iteration.get("slice_index_240")))
                    except Exception:
                        pass
                    visual_iteration_memory = build_visual_loop_memory(visual_iteration_memories)
                    if visual_iteration_memory:
                        facts_memory = dict(facts_memory)
                        facts_memory["visual_memory"] = visual_iteration_memory

                    if not is_visual_judgment_unclear(visual_iteration_memory):
                        t1s_loop_stop_reason = "visual_judgment_concrete"
                        break
                    if iteration_index >= max_iters:
                        t1s_loop_stop_reason = "max_iters_reached"
                        break
                    if not visual_memory_requests_another_slice(visual_iteration_memory):
                        t1s_loop_stop_reason = "no_verifiable_next_slice_assumption"
                        break

                verbose_memory["visual_iterations"] = visual_results
                verbose_memory["visual_iteration"] = visual_results[-1] if visual_results else None
                verbose_memory["visual_tool_registry"] = image_tool_registry
                verbose_memory["visual_tool_selections"] = t1s_selections
                verbose_memory["visual_tool_selection"] = t1s_selection
                verbose_memory["visual_loop_stop_reason"] = t1s_loop_stop_reason

        record["t1s_task_guidance"] = t1s_task_guidance
        record["tool_registry"] = tool_registry
        record["tool_candidates"] = selected_tool_calls
        record["tool_selection"] = tool_selection
        record["image_tool_registry"] = image_tool_registry
        record["t1s_selection"] = t1s_selection
        record["t1s_selections"] = t1s_selections
        record["visual_iteration_memory"] = visual_iteration_memory
        record["visual_iteration_memories"] = visual_iteration_memories
        record["t1s_iteration_count"] = len(visual_iteration_memories)
        record["t1s_used_slice_indices"] = used_slice_indices
        record["t1s_loop_stop_reason"] = t1s_loop_stop_reason
        if preliminary_answer is not None:
            record["pre_t1s_answer"] = {
                **preliminary_answer,
                "raw_response": preliminary_answer_raw,
            }
        record["facts_memory"] = facts_memory
        record["debug_memory"] = verbose_memory
        record["memory_schema_warnings"] = validate_facts_memory(facts_memory)
        record["runtime_tool_included"] = bool(runtime_tools)
        record["t1s_included"] = bool(visual_iteration_memory)
        visual_fields = visual_judgment_fields(visual_iteration_memory)
        visual_actionability = visual_assumption_actionability(visual_iteration_memory)
        record["t1s_visual_judgment_answer"] = visual_fields.get("answer")
        record["t1s_visual_judgment_confidence"] = visual_fields.get("confidence")
        record["t1s_visual_uncertainty"] = visual_fields.get("uncertainty")
        record["t1s_has_actionable_visual_evidence"] = visual_actionability.get("has_actionable_visual_evidence")
        record["t1s_assumption_test_counts"] = visual_actionability.get("assumption_test_counts")
        record["t1s_assumption_relevance_counts"] = visual_actionability.get("assumption_relevance_counts")
        record["t1s_actionable_assumption_evidence"] = visual_actionability.get("actionable_assumption_evidence")

        if args.dry_run:
            record["GPT_raw_result"] = "DRY_RUN"
            if args.save_dry_run:
                await save_record(args.save_dir, record, file_lock)
            return record

        if args.memory_only:
            record["GPT_raw_result"] = "MEMORY_ONLY"
            await save_record(args.save_dir, record, file_lock)
            return record

        if args.include_t1s and preliminary_answer is not None and visual_iteration_memory is None:
            apply_answer_to_record(record, preliminary_answer, preliminary_answer_raw)
            record["used_pre_t1s_answer_as_final"] = True
            await save_record(args.save_dir, record, file_lock)
            return record

        if (
            args.include_t1s
            and preliminary_answer is not None
            and visual_iteration_memory is not None
            and not visual_actionability.get("has_actionable_visual_evidence")
        ):
            apply_answer_to_record(record, preliminary_answer, preliminary_answer_raw)
            record["used_pre_t1s_answer_as_final"] = True
            record["t1s_final_skip_reason"] = "no_actionable_visual_evidence"
            await save_record(args.save_dir, record, file_lock)
            return record

        reasoning_memory: Optional[Dict[str, Any]] = None
        if args.memory_mode == "hybrid":
            try:
                response = await call_gpt(build_reasoning_memory_prompt(facts_memory), model=args.model)
                reasoning_memory = safe_json_loads(response) or {"raw_summary": response}
                record["reasoning_memory"] = reasoning_memory
            except Exception as exc:
                record["reasoning_memory"] = {"error": str(exc)}
                reasoning_memory = None

        try:
            answer_response = await call_gpt(
                build_t1s_answer_prompt(
                    question,
                    facts_memory,
                    reasoning_memory=reasoning_memory,
                    question_type=question_type,
                ),
                model=args.model,
            )
            parsed = parse_answer_response(answer_response)
            apply_answer_to_record(record, compact_answer_for_memory(parsed, answer_response), answer_response)
            record["used_pre_t1s_answer_as_final"] = False
        except Exception as exc:
            record["GPT_raw_result"] = f"ERROR: {exc}"

        await save_record(args.save_dir, record, file_lock)
        return record


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run Final_Test memory pipeline with optional runtime tools and T1S.")
    parser.add_argument("--vqa-path", default=as_str(VQA_PATH))
    parser.add_argument("--report-dir", default=as_str(REPORT_DIR))
    parser.add_argument("--clip-global-dir", default=as_str(CLIP_GLOBAL_DIR))
    parser.add_argument("--clip-detail-dir", default=as_str(CLIP_DETAIL_DIR))
    parser.add_argument("--clip-detail-slice-dir", default=as_str(CLIP_DETAIL_SLICE_DIR))
    parser.add_argument("--mask-dir", default=as_str(MASK_DIR))
    parser.add_argument("--region-memory-dir", default=None)
    parser.add_argument("--save-dir", default=as_str(T1S_SAVE_DIR))
    parser.add_argument("--question-type", default="")
    parser.add_argument("--target-subtypes", default="")
    parser.add_argument("--max-per-subtype", type=int, default=20)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--vision-model", default=MODEL_NAME)
    parser.add_argument("--normalizer", choices=("gpt", "rule"), default="gpt")
    parser.add_argument("--memory-mode", choices=("facts", "hybrid"), default="hybrid")
    parser.add_argument("--include-runtime-tools", action="store_true")
    parser.add_argument("--tool-selector", choices=("none", "gpt"), default="gpt")
    parser.add_argument("--runtime-cache-dir", default=as_str(RUNTIME_CACHE_DIR))
    parser.add_argument("--include-t1s", action="store_true")
    parser.add_argument("--t1s-max-iters", type=int, default=1)
    parser.add_argument("--t1s-cache-dir", default=as_str(T1S_CACHE_DIR))
    parser.add_argument("--t1s-render-dir", default=as_str(T1S_RENDER_DIR))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-dry-run", action="store_true")
    parser.add_argument("--memory-only", action="store_true")
    parser.add_argument("--save-skips", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--concurrency", type=int, default=MAX_CONCURRENCY)
    args = parser.parse_args()

    target_subtypes = parse_target_subtypes(args.target_subtypes)
    if not args.question_type and not target_subtypes:
        target_subtypes = list(TARGET_SUBTYPES)
    df = select_target_rows(
        args.vqa_path,
        args.max_per_subtype,
        target_subtypes,
        question_type=args.question_type,
    )
    skipped_existing = 0
    if args.skip_existing:
        existing_keys = collect_existing_keys(args.save_dir)
        if existing_keys and not df.empty:
            keep_mask = ~df.apply(
                lambda row: (int(row["case_idx"]), str(row.get("question subtype", ""))) in existing_keys,
                axis=1,
            )
            skipped_existing = int((~keep_mask).sum())
            df = df[keep_mask].reset_index(drop=True)
    rows = df.to_dict("records")
    counts = df["question subtype"].value_counts().to_dict() if not df.empty else {}

    print("Final_Test memory pipeline")
    print(f"Selected rows: {len(rows)}")
    if args.question_type:
        print(f"Question type filter: {args.question_type}")
    if target_subtypes:
        for subtype in target_subtypes:
            print(f"  {subtype}: {counts.get(subtype, 0)}")
    else:
        for subtype, count in sorted(counts.items()):
            print(f"  {subtype}: {count}")
    print(f"Save dir: {args.save_dir}")
    print(f"Dry run: {args.dry_run}")
    print(f"Normalizer: {args.normalizer}")
    print(f"Memory mode: {args.memory_mode}")
    print(f"Runtime tools: {args.include_runtime_tools}")
    print(f"T1S visual loop: {args.include_t1s}")
    print(f"Skip existing: {args.skip_existing} ({skipped_existing} skipped)")
    if args.include_runtime_tools:
        print(f"Tool selector: {args.tool_selector}")
        print(f"Runtime cache dir: {args.runtime_cache_dir}")
    if args.include_t1s:
        print(f"T1S max iterations: {max(1, args.t1s_max_iters)}")
        print(f"T1S cache dir: {args.t1s_cache_dir}")
        print(f"T1S render dir: {args.t1s_render_dir}")
        print(f"Vision model: {args.vision_model}")

    paths = MemorySourcePaths(
        clip_global_dir=args.clip_global_dir,
        clip_detail_dir=args.clip_detail_dir,
        clip_detail_slice_dir=args.clip_detail_slice_dir,
        mask_dir=args.mask_dir,
        region_memory_dir=args.region_memory_dir,
    )

    runtime_device = "cuda:0" if args.device == "cuda" else args.device
    runtime_context_state: Dict[str, Any] = {"context": None, "device": runtime_device}

    semaphore = asyncio.Semaphore(args.concurrency)
    runtime_lock = asyncio.Lock()
    t1s_lock = asyncio.Lock()
    file_lock = asyncio.Lock()
    tasks = [
        process_one(row, args, paths, runtime_context_state, runtime_lock, t1s_lock, semaphore, file_lock)
        for row in rows
    ]

    records: List[Dict[str, Any]] = []
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Final_Test"):
        records.append(await task)

    print(f"Records processed: {len(records)}")
    warnings = [record.get("memory_schema_warnings") for record in records if record.get("memory_schema_warnings")]
    print(f"Records with schema warnings: {len(warnings)}")
    candidate_counts = [
        len(((record.get("facts_memory") or {}).get("candidate_slice_queue") or []))
        for record in records
        if record.get("facts_memory")
    ]
    if candidate_counts:
        print(f"Candidate slice count: min={min(candidate_counts)}, max={max(candidate_counts)}")
    print(f"T1S records with visual iterations: {sum(1 for record in records if record.get('t1s_included'))}")
    print(f"T1S total visual iterations: {sum(int(record.get('t1s_iteration_count') or 0) for record in records)}")

    if args.dry_run and records:
        preview = records[0].get("facts_memory") or {}
        preview = {
            "case_context": preview.get("case_context"),
            "inferred_query": preview.get("inferred_query"),
            "candidate_slice_queue": preview.get("candidate_slice_queue"),
            "visual_memory": preview.get("visual_memory"),
            "option_evidence": preview.get("option_evidence"),
            "warnings": records[0].get("memory_schema_warnings"),
        }
        print("\nDry-run preview:")
        print(json.dumps(preview, indent=2)[:12000])
        if not args.save_dry_run:
            print("Dry run completed. No prediction files were written.")


if __name__ == "__main__":
    asyncio.run(main())
