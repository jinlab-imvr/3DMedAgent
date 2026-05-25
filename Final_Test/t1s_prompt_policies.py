#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Task-level prompt policies for the T1S visual loop.

These policies use only leak-safe public question type metadata. They must not
inspect question subtype, answers, correct options, or any ground-truth fields.
"""

from __future__ import annotations

import json
from typing import Any, Dict


TASK_LEVEL_GUIDANCE = {
    "measurement": {
        "evidence_prior": "Numerical organ/lesion measurements, HU, volume, diameter, structured measurement, and runtime/tool values are primary evidence.",
        "t1s_policy": "Default to no T1S. Use T1S only when the text/tool-only reasoning explicitly says the answer cannot be chosen without a local visual sanity check.",
        "visual_relevance": "Single-slice visual observations are supporting evidence only; they must not replace numerical, tool-derived, or whole-volume evidence.",
    },
    "recognition": {
        "evidence_prior": "CT-CLIP volume-level global existence probabilities are the primary image-derived evidence; report and organ/lesion memory provide context.",
        "t1s_policy": "Default to no T1S. Use T1S only when the text/tool-only reasoning explicitly says global/report/memory evidence is insufficient and local visibility would resolve that conflict.",
        "visual_relevance": "Slice/detail visibility may explain localization or uncertainty, but it must not decide present/absent or override strong global probability evidence.",
    },
    "visual reasoning": {
        "evidence_prior": "Spatial, slice-local, option-specific, and multi-slice accumulated visual evidence can be primary evidence.",
        "t1s_policy": "Use T1S to accumulate one-slice evidence for visible spatial, location, adjacency, distribution, outlier, or candidate-slice assumptions when memory alone is insufficient.",
        "visual_relevance": "High-confidence direct_option or critical_intermediate evidence may materially update final reasoning when it corresponds to visible or spatial MCQ criteria.",
    },
    "medical reasoning": {
        "evidence_prior": "Integrated report, organ/lesion memory, runtime evidence, and clinical/medical criteria are primary evidence.",
        "t1s_policy": "Use T1S only when there is a clear primary-evidence gap that a local slice can test. A single-slice appearance should not by itself overturn text/runtime/global evidence.",
        "visual_relevance": "Single-slice findings are usually weak_context or cautious critical_intermediate; use direct_option only for a directly visible local MCQ criterion.",
    },
}


def normalize_question_type(question_type: Any) -> str:
    text = str(question_type or "").strip().lower()
    return text if text in TASK_LEVEL_GUIDANCE else "unknown"


def build_task_guidance(question_type: Any) -> Dict[str, Any]:
    normalized = normalize_question_type(question_type)
    guidance = TASK_LEVEL_GUIDANCE.get(
        normalized,
        {
            "evidence_prior": "Use the strongest leak-safe memory evidence for the question.",
            "t1s_policy": "Use T1S only when one candidate slice can add concrete visual evidence that memory does not already provide.",
            "visual_relevance": "Treat single-slice visual evidence as local and limited unless it directly matches the MCQ criterion.",
        },
    )
    return {"question_type": normalized, **guidance}


def build_text_answer_policy(question_type: Any) -> str:
    normalized = normalize_question_type(question_type)
    if normalized == "recognition":
        return """Text-only policy for recognition:
- Internally rewrite the question as: "Is the target present?"
- If the CT-CLIP volume-level global evidence supports target presence, choose the MCQ option whose exact option text is "Yes". If it supports absence, choose the option whose exact option text is "No".
- For bare Yes/No options, Yes always means target present and No always means target absent, regardless of surface wording such as "free of".
- Do not use option text like "free of", "contains", "has", or "affected by" to remap Yes/No; use only the exact Yes/No option text.
- Use only clip_global / clip_global_matrix for the present/absent recognition decision. Do not use clip_detail sections/slices, top slices, candidate_slice_queue, option_evidence slice scores, or report normal wording to decide present/absent.
- If report or organ memory conflicts with global recognition evidence, mention the conflict in uncertainty but keep final_answer aligned with the global Yes/No mapping."""
    if normalized == "measurement":
        return ""
    if normalized == "visual reasoning":
        return """Text-only policy for visual reasoning:
- First compare option_evidence and runtime evidence when they directly correspond to MCQ options.
- candidate_slice_queue marks future visual-verification positions; it is not by itself evidence that an option is correct.
- Location/slice probabilities can rank provided candidates, but they do not prove physical contact, global distribution, diameter, or counts unless the memory/tool explicitly provides that evidence."""
    if normalized == "medical reasoning":
        return """Text-only policy for medical reasoning:
- Use report, organ/lesion memory, CT-CLIP global evidence, and medical-standard reasoning as the primary evidence.
- Treat ctclip_location, candidate slices, section/slice scores, and local visibility as supporting localization evidence only.
- Do not let a local/slice/location score overturn stronger report, global, runtime numeric/count, or integrated medical evidence.
- Use runtime evidence to change the answer only when it directly matches the MCQ criterion and does not conflict with stronger primary evidence."""
    return ""


def build_planner_policy(question_type: Any) -> str:
    normalized = normalize_question_type(question_type)
    base = """Task policy for planning:
- Use question type only as evidence-prior metadata, never as subtype routing.
- Each selected slice must verify or contradict one explicit assumption.
- In the purpose or rationale, state whether the assumption addresses a primary evidence gap or only adds supporting local evidence."""
    if normalized == "measurement":
        return base + """
- Measurement questions should default to no visual call because numeric/tool evidence owns the answer.
- Return no call when structured values, runtime/tool values, or option numeric distances already separate the options.
- Call T1S only if pre_t1s_reasoning or facts_memory explicitly leaves an option-level ambiguity that cannot be resolved by numeric/tool evidence and one candidate slice can sanity-check a visible local premise.
- Do not call T1S when the only benefit would be reassurance, visual context, localization, or weak confirmation of an already selected numeric answer.
- Do not inspect slices to estimate exact size, HU, volume, count, or whole-lesion extent."""
    if normalized == "recognition":
        return base + """
- Recognition questions should default to no visual call because volume-level global probability and report/memory evidence own the present/absent decision.
- Return no call when global evidence already supports present or absent, even if candidate slices or detail scores are available.
- Call T1S only if pre_t1s_reasoning or facts_memory explicitly identifies a global/report/memory conflict that local target visibility could resolve.
- Do not call T1S when the only benefit would be localization, visual confirmation, or checking whether a globally supported target is visible on one slice.
- Do not use CT-CLIP detail/slice localization evidence or one rendered slice to override strong global existence/probability evidence."""
    if normalized == "medical reasoning":
        return base + """
- Medical reasoning usually relies on integrated report, memory, runtime evidence, and medical criteria.
- Start from no-call. Call T1S only when the baseline reasoning has a concrete primary-evidence gap that one candidate slice can directly test.
- T1S may verify local morphology, local position, visible contact, or organ-boundary context, but this is usually supporting evidence rather than answer-owning evidence.
- Do not call or continue merely to confirm local appearance, attenuation impression, vessel proximity, or boundary contact when the answer requires integrated/global evidence.
- Continue to another slice only if the previous iteration left a specific unresolved local premise and an unused candidate slice can materially reduce that same gap."""
    if normalized == "visual reasoning":
        return base + """
- Visual reasoning can rely on accumulated single-slice spatial/local evidence.
- Continue when another unused candidate slice can test an option-specific, spatial, location, adjacency, distribution, outlier, or candidate-slice assumption.
- If the preliminary answer is already well supported and the next slice would only search for a weak alternative appearance, stop unless the candidate directly tests a concrete contradiction of that baseline.
- Do not continue for generic confidence; continue only when the next slice can update the visible evidence chain."""
    return base


def build_observer_policy(question_type: Any) -> str:
    normalized = normalize_question_type(question_type)
    base = """Task policy for visual observation:
- Judge the selected assumption locally and confidently when the slice visibly supports or refutes it.
- direct_option means the local visual evidence touches an option's visible criterion; it does not automatically decide the final answer.
- In assumption_test.rationale, note whether the evidence is primary_for_question or supporting_only, and mention conflicts with report/runtime/global memory."""
    if normalized == "measurement":
        return base + """
- Do not infer exact measurements, HU, volume, count, or whole-volume extent from this image.
- Default answer_relevance to weak_context. Use critical_intermediate only for a visible local sanity check of an otherwise ambiguous primary measurement.
- If primary numeric/tool evidence was already adequate, mark this observation weak_context even when the local anatomy is visible.
- Use direct_option only when the MCQ criterion is a directly visible local fact rather than a measurement.
- Mark visual evidence as supporting_only when numerical/tool evidence owns the answer."""
    if normalized == "recognition":
        return base + """
- Target visibility can be useful, but detail/slice evidence is localization support rather than primary recognition evidence.
- Default answer_relevance to weak_context. Use critical_intermediate only when visibility directly addresses a stated conflict in primary recognition evidence.
- If global present/absent evidence was already adequate, mark this observation weak_context even when the target is visible or absent on this slice.
- Use direct_option only when the MCQ criterion is literally local visibility on the selected slice; otherwise do not let target visibility become answer-owning evidence.
- Mark visual evidence as supporting_only when global probability evidence is strong."""
    if normalized == "medical reasoning":
        return base + """
- Local morphology, local attenuation appearance, local position, vessel/organ contact, and organ-boundary context are usually supporting_only.
- Default answer_relevance to weak_context when the finding is only a local appearance clue.
- Use critical_intermediate only when the slice tests a concrete local premise that the baseline explicitly needed and that cannot be read from stronger report/runtime/global evidence.
- Use direct_option only when the MCQ criterion itself is a directly visible local fact. For diagnostic, classification, severity, or management categories, local visual appearance is not direct_option even when it seems persuasive.
- If local appearance conflicts with report/runtime/global evidence, explicitly describe the conflict and avoid overstating relevance."""
    if normalized == "visual reasoning":
        return base + """
- Spatial, location, adjacency, option-candidate, and local outlier evidence can be primary_for_question when it matches the visible MCQ criterion.
- It is acceptable for current_judgment.answer to remain unclear while verified/contradicted assumptions accumulate.
- Use direct_option or critical_intermediate when the inspected slice meaningfully supports the visible evidence chain."""
    return base


def build_final_policy(question_type: Any, has_visual_memory: bool) -> str:
    task_guidance = build_task_guidance(question_type)
    header = f"""Task-level evidence prior:
{json.dumps(task_guidance, ensure_ascii=False, indent=2)}

Use this task-level prior only to weigh evidence types. It is leak-safe task metadata, not an answer label. Do not use question subtype routing."""
    if not has_visual_memory:
        return header

    normalized = task_guidance["question_type"]
    common = """

Additional T1S visual-memory policy:
- Treat pre_t1s_reasoning as the text/tool-only baseline judgment.
- visual_memory is a sequence of one-slice VLM observations, not a full-volume image review.
- assumption_test entries are local visual checks of explicit assumptions, not hidden labels.
- weak_context, none, unresolved, low-confidence, or scope-insufficient visual evidence must not change the baseline answer.
- Single-slice observations must not be treated as exact quantitative, whole-volume, dynamic, or hidden-label evidence."""
    if normalized == "measurement":
        return header + common + """
- For measurement, numerical/tool/structured values own the final answer.
- T1S can only explain or sanity-check a visible local premise when numerical evidence is ambiguous or internally conflicting.
- Do not revise the baseline using visual appearance when numeric, HU, volume, diameter, structured, or runtime evidence already separates the options.
- If T1S was called despite adequate primary evidence, keep the baseline and mention visual evidence only as local context."""
    if normalized == "recognition":
        return header + common + """
- For recognition, CT-CLIP volume-level global probability owns the image-derived present/absent decision; report and organ/lesion memory provide context and conflict notes.
- CT-CLIP detail sections/slices and candidate slices are localization or inspection hints; do not use them as present/absent evidence.
- T1S may revise only when primary global/report/memory evidence is weak or conflicting and the inspected slice directly resolves that conflict.
- Do not let one visible or non-visible slice override strong existence/probability evidence.
- If T1S was called but global evidence is clear, keep the baseline global Yes/No decision."""
    if normalized == "medical reasoning":
        return header + common + """
- For medical reasoning, report, organ/lesion memory, runtime evidence, and medical-standard integration own the final answer.
- critical_intermediate visual evidence updates only the tested local premise; it must not jump by itself to a diagnostic, classification, severity, management, or other composite conclusion.
- direct_option can affect the answer only when the MCQ criterion is directly visible locally and does not conflict with stronger task-primary evidence.
- Local visual appearance of type, severity, invasion, contact, boundary, or management-related criteria is not enough by itself to change the answer.
- If visual evidence conflicts with a baseline supported by report/global/runtime/integrated evidence, keep the baseline unless the visual evidence directly refutes a core visible assumption and no stronger primary evidence conflicts with that refutation.
- When in doubt between a strong pre_t1s_reasoning answer and a single-slice visual impression, keep the pre_t1s_reasoning answer."""
    if normalized == "visual reasoning":
        return header + common + """
- For visual reasoning, accumulated one-slice spatial/local evidence may own the final answer when it matches the visible MCQ criterion.
- High-confidence direct_option and critical_intermediate evidence can revise the baseline when it supports or refutes an option-specific spatial/local assumption.
- If the preliminary answer is strongly supported by text/runtime/global memory and visual evidence is only weak local appearance, keep the baseline.
- If pre_t1s_reasoning has low uncertainty or multiple concrete evidence references, revise only when visual_memory gives a clear, high-confidence contradiction of the same option criterion; otherwise keep the baseline.
- Do not damage strong preliminary answers when visual_memory adds no concrete option-specific or spatial evidence."""
    return header + common + """
- Use visual_memory only when it directly adds evidence stronger than the baseline for the specific MCQ criterion.
- Keep the baseline when visual evidence is only local support or conflicts with stronger memory evidence."""
