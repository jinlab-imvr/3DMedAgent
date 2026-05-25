# Tool_Box/slice_selection.py

import os
import json
import re
import numpy as np
import nibabel as nib
from typing import List, Dict, Tuple, Optional


# =========================
# 基础 uniform 工具
# =========================

def uniform_sample_from_range(start: int, end: int, num: int) -> List[int]:
    if start > end:
        start, end = end, start
    if num <= 1:
        return [(start + end) // 2]
    return np.linspace(start, end, num, dtype=int).tolist()


def uniform_sample_whole_volume(depth: int, num: int) -> List[int]:
    if depth <= 0:
        return []
    if depth <= num:
        return list(range(depth))
    return np.linspace(0, depth - 1, num, dtype=int).tolist()


# =========================
# 1️⃣ Uniform slice selector
# =========================

def select_uniform_slices(
    volume_depth: int,
    num_slices: int,
) -> Tuple[List[int], Dict[int, str]]:
    indices = uniform_sample_whole_volume(volume_depth, num_slices)
    sources = {idx: "uniform" for idx in indices}
    return indices, sources


# =========================
# 2️⃣ Organ-aware selector
# =========================

def select_slices_by_organ(
    memory_path: str,
    organ_name: str,
    volume_depth: int,
    min_slices: int = 5,
) -> Tuple[List[int], Dict[int, str]]:
    """
    memory_path: slice_memory.json
    """
    if not os.path.exists(memory_path):
        indices = uniform_sample_whole_volume(volume_depth, min_slices)
        return indices, {i: "fallback_uniform" for i in indices}

    try:
        mem = json.load(open(memory_path))
        region_slices = mem.get("region_slices", {})
    except Exception:
        indices = uniform_sample_whole_volume(volume_depth, min_slices)
        return indices, {i: "fallback_uniform" for i in indices}

    organ = organ_name.lower()
    selected_organs = []

    if "kidney" in organ:
        if "kidney" in region_slices:
            selected_organs = ["kidney"]
        else:
            for k in ["left kidney", "right kidney"]:
                if k in region_slices:
                    selected_organs.append(k)
    else:
        if organ in region_slices:
            selected_organs = [organ]

    indices = []
    sources = {}

    if len(selected_organs) == 1:
        s, e = region_slices[selected_organs[0]]
        tmp = uniform_sample_from_range(s, e, min_slices)
        for i in tmp:
            indices.append(i)
            sources[i] = "organ"

    elif len(selected_organs) >= 2:
        for org in selected_organs[:2]:
            s, e = region_slices[org]
            tmp = uniform_sample_from_range(s, e, 3)
            for i in tmp:
                indices.append(i)
                sources[i] = "organ"

    if len(indices) < min_slices:
        extra = uniform_sample_whole_volume(volume_depth, min_slices)
        for i in extra:
            if i not in sources:
                indices.append(i)
                sources[i] = "fallback_uniform"

    indices = sorted(set(indices))
    return indices, sources


# =========================
# 3️⃣ Lesion slice parsing
# =========================

def extract_lesion_slices_from_report(
    report_text: str,
    organ_name: str,
) -> List[int]:
    organ_patterns = {
        "liver": r"(liver|hepatic)",
        "pancreas": r"(pancreas|pancreatic)",
        "kidney": r"(kidney|renal)",
        "spleen": r"(spleen|splenic)",
        "colon": r"(colon|colonic)",
    }

    organ = organ_name.lower()
    if organ not in organ_patterns:
        return []

    header_pat = rf"{organ_patterns[organ]}\s+malignant\s+(tumor|tumors|lesion|lesions):"
    header = re.search(header_pat, report_text, flags=re.IGNORECASE)
    if not header:
        return []

    start = header.end()
    next_sec = re.search(r"\n[A-Z][a-zA-Z\s]*:\s*\n", report_text[start:])
    end = start + next_sec.start() if next_sec else len(report_text)

    block = report_text[start:end]

    slices = []
    entries = re.split(r"(?:tumor|lesion)\s+\d+:", block, flags=re.IGNORECASE)
    for ent in entries:
        m = re.search(r"slice\s+(\d+)", ent)
        if m:
            slices.append(int(m.group(1)))

    return slices


# =========================
# 4️⃣ Organ + lesion selector
# =========================

def select_slices_organ_plus_lesion(
    memory_path: str,
    report_text: str,
    organ_name: str,
    volume_depth: int,
    min_slices: int = 5,
) -> Tuple[List[int], Dict[int, str]]:

    lesion_indices = extract_lesion_slices_from_report(
        report_text, organ_name
    )

    indices = []
    sources = {}

    # 1. lesion slices always kept
    for i in lesion_indices:
        indices.append(i)
        sources[i] = "lesion"

    # 2. organ补齐
    if len(indices) < min_slices:
        organ_indices, organ_sources = select_slices_by_organ(
            memory_path, organ_name, volume_depth, min_slices
        )
        for i in organ_indices:
            if i not in sources:
                indices.append(i)
                sources[i] = organ_sources.get(i, "organ")
            else:
                sources[i] = "unified"

    # 3. 再兜底
    if len(indices) < min_slices:
        extra = uniform_sample_whole_volume(volume_depth, min_slices)
        for i in extra:
            if i not in sources:
                indices.append(i)
                sources[i] = "fallback_uniform"

    indices = sorted(set(indices))
    return indices, sources

def select_largest_lesion_slice(
    report_text: str,
    organ_name: str,
    volume_depth: int,
) -> Tuple[List[int], Dict[int, str]]:
    """
    Select ONLY ONE slice corresponding to the largest lesion.

    Strategy:
    1) Parse lesion slice indices from report.
    2) If multiple lesion slices exist, pick the middle one
       (robust proxy for the main / largest lesion).
    3) If no lesion slice is found, fallback to middle slice of the volume.

    Returns:
        indices: [slice_idx]
        sources: {slice_idx: "largest_lesion" | "fallback_uniform"}
    """

    lesion_indices = extract_lesion_slices_from_report(
        report_text, organ_name
    )

    # 保证 slice index 合法
    lesion_indices = [
        int(np.clip(i, 0, volume_depth - 1))
        for i in lesion_indices
    ]
    lesion_indices = sorted(set(lesion_indices))

    # Case 1: report 中有 lesion slice
    if len(lesion_indices) > 0:
        # 选中位数作为“最大 lesion”的 proxy
        mid_idx = lesion_indices[len(lesion_indices) // 2]
        return [mid_idx], {mid_idx: "largest_lesion"}

    # Case 2: report 没提 lesion → fallback
    if volume_depth <= 0:
        return [], {}

    fallback_idx = volume_depth // 2
    return [fallback_idx], {fallback_idx: "fallback_uniform"}


def select_largest_organ_slice(
    organ_mask_path: str,
    volume_depth: int,
) -> Tuple[List[int], Dict[int, str]]:
    """
    Select ONLY ONE slice corresponding to the largest cross-sectional
    area of the given organ mask.

    Assumptions:
    - organ mask shape is (H, W, D)
    - slice axis is the last dimension (D)
    - mask is binary or multi-label (non-zero treated as foreground)

    Args:
        organ_mask_path: path to organ segmentation NIfTI (.nii / .nii.gz)
        volume_depth: number of slices (D), used for sanity clipping

    Returns:
        indices: [slice_idx]
        sources: {slice_idx: "largest_organ_cross_section"}
    """

    if not os.path.exists(organ_mask_path):
        return [], {}

    try:
        mask_nii = nib.load(organ_mask_path)
        mask = mask_nii.get_fdata()
    except Exception:
        return [], {}

    if mask.ndim != 3:
        return [], {}

    H, W, D = mask.shape

    # volume_depth 以主程序为准，双重保护
    D_eff = min(D, volume_depth)
    if D_eff <= 0:
        return [], {}

    # 将 mask 转成 binary
    mask = mask > 0

    best_slice = None
    best_area = 0

    # 在 D 轴上扫描横截面积
    for idx in range(D_eff):
        area = mask[:, :, idx].sum()
        if area > best_area:
            best_area = area
            best_slice = idx

    if best_slice is None or best_area == 0:
        return [], {}

    return [best_slice], {best_slice: "largest_organ_cross_section"}

def select_largest_organ_slice_with_mask2d(
    organ_mask_path: str,
    volume_depth: int,
) -> Tuple[List[int], Dict[int, str], Optional[np.ndarray]]:
    """
    Enhanced version of select_largest_organ_slice:
    - Same input style: organ_mask_path + volume_depth
    - Same first two outputs: indices, sources
    - Extra third output: 2D binary mask (uint8 0/1) on the selected slice

    Assumptions (same as original):
    - organ mask shape is (H, W, D)
    - slice axis is the last dimension (D)
    - mask is binary or multi-label (non-zero treated as foreground)

    Args:
        organ_mask_path: path to organ segmentation NIfTI (.nii / .nii.gz)
        volume_depth: number of slices (D), used for sanity clipping

    Returns:
        indices: [slice_idx]
        sources: {slice_idx: "largest_organ_cross_section"}
        mask2d_u8: (H, W) uint8 in {0,1}; None if failed
    """
    if not os.path.exists(organ_mask_path):
        return [], {}, None

    try:
        mask_nii = nib.load(organ_mask_path)
        mask = mask_nii.get_fdata()
    except Exception:
        return [], {}, None

    if mask.ndim != 3:
        return [], {}, None

    H, W, D = mask.shape

    # volume_depth 以主程序为准，双重保护
    D_eff = min(D, volume_depth)
    if D_eff <= 0:
        return [], {}, None

    # 将 mask 转成 binary
    mask_bin = mask > 0

    best_slice = None
    best_area = 0

    # 在 D 轴上扫描横截面积
    for idx in range(D_eff):
        area = int(mask_bin[:, :, idx].sum())
        if area > best_area:
            best_area = area
            best_slice = idx

    if best_slice is None or best_area == 0:
        return [], {}, None

    # 额外输出：该 slice 的 2D mask（0/1）
    mask2d_u8 = mask_bin[:, :, best_slice].astype(np.uint8)

    return [int(best_slice)], {int(best_slice): "largest_organ_cross_section"}, mask2d_u8