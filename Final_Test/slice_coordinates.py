#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Slice-coordinate helpers for mapping VQA percentages to CT-CLIP space."""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Optional

from config import RAW_CT_ROOT, SUBSET_ROOT, as_str


TOTAL_SLICES = 240
DEFAULT_DATA_ROOTS = (
    as_str(SUBSET_ROOT),
    as_str(RAW_CT_ROOT),
)


def clamp_slice_index(value: Any) -> int:
    try:
        idx = int(round(float(value)))
    except Exception:
        idx = 0
    return max(0, min(TOTAL_SLICES - 1, idx))


def z_percent(slice_index_240: int) -> float:
    return round(100.0 * int(slice_index_240) / (TOTAL_SLICES - 1), 4)


def direct_percent_to_ctclip_index(percent: float) -> int:
    return clamp_slice_index(float(percent) / 100.0 * (TOTAL_SLICES - 1))


def _candidate_nifti_paths(image_id: str, dataset: str, data_roots: Iterable[str]) -> Iterable[str]:
    for root in data_roots:
        yield os.path.join(str(root), str(dataset), "img", f"{image_id}.nii.gz")


def _read_nifti_depth(path: str) -> int:
    import nibabel as nib  # Imported lazily so non-slice cases avoid this dependency path.

    nii = nib.load(path)
    shape = tuple(int(v) for v in nii.header.get_data_shape())
    if len(shape) < 3:
        raise ValueError(f"unexpected_nifti_shape:{shape}")
    return int(shape[2])


def map_vqa_percent_to_ctclip_slice(
    percent: float,
    image_id: Any,
    dataset: Any,
    data_roots: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Map a VQA raw-volume z percentage into CT-CLIP's 240-slice coordinate.

    The CT-CLIP preprocessing center-crops volumes deeper than 240 slices and
    center-pads volumes shallower than 240 slices. If the raw VQA slice would
    be outside the crop window, this helper falls back to the legacy direct
    percent-to-240 mapping and marks the result approximate.
    """
    percent_value = float(percent)
    direct_idx = direct_percent_to_ctclip_index(percent_value)
    base: Dict[str, Any] = {
        "slice_percent": percent_value,
        "direct_slice_index_240": direct_idx,
        "slice_index_240": direct_idx,
        "ctclip_percent": z_percent(direct_idx),
        "coordinate_frame": "vqa_percent_direct_240",
        "coordinate_transform_status": "fallback_direct_240",
        "coordinate_transform_note": "",
        "is_approximate_coordinate": True,
    }

    roots = tuple(data_roots or DEFAULT_DATA_ROOTS)
    existing_path = None
    missing_paths = []
    for path in _candidate_nifti_paths(str(image_id), str(dataset), roots):
        if os.path.exists(path):
            existing_path = path
            break
        missing_paths.append(path)
    if not existing_path:
        base["coordinate_transform_status"] = "fallback_direct_240_missing_nifti"
        base["coordinate_transform_note"] = "No current NIfTI header was found for percent-to-CT-CLIP mapping."
        base["nifti_paths_checked"] = missing_paths[:3]
        return base

    try:
        raw_depth = _read_nifti_depth(existing_path)
    except Exception as exc:
        base["coordinate_transform_status"] = "fallback_direct_240_header_error"
        base["coordinate_transform_note"] = str(exc)
        base["nifti_path"] = existing_path
        return base

    if raw_depth <= 0:
        base["coordinate_transform_status"] = "fallback_direct_240_invalid_depth"
        base["coordinate_transform_note"] = f"Invalid raw depth: {raw_depth}"
        base["nifti_path"] = existing_path
        base["source_depth"] = raw_depth
        return base

    raw_idx = int(round(percent_value / 100.0 * (raw_depth - 1)))
    d_start = max((raw_depth - TOTAL_SLICES) // 2, 0)
    d_end = min(d_start + TOTAL_SLICES, raw_depth)
    kept_depth = d_end - d_start
    pad_before = (TOTAL_SLICES - kept_depth) // 2
    base.update({
        "nifti_path": existing_path,
        "source_depth": raw_depth,
        "raw_slice_index": raw_idx,
        "crop_start": d_start,
        "crop_end_exclusive": d_end,
        "pad_before": pad_before,
    })

    if raw_idx < d_start or raw_idx >= d_end:
        base["coordinate_transform_status"] = "fallback_direct_240_raw_slice_cropped"
        base["coordinate_transform_note"] = "Raw VQA slice is outside CT-CLIP center-crop window."
        return base

    mapped_idx = clamp_slice_index(raw_idx - d_start + pad_before)
    base.update({
        "slice_index_240": mapped_idx,
        "ctclip_percent": z_percent(mapped_idx),
        "coordinate_frame": "ctclip_240",
        "coordinate_transform_status": "mapped_raw_percent_to_ctclip_240",
        "coordinate_transform_note": "Mapped from current NIfTI depth through CT-CLIP center crop/pad.",
        "is_approximate_coordinate": False,
    })
    return base


def section_index_from_ctclip_slice(slice_index_240: Any) -> int:
    return max(0, min(23, int(clamp_slice_index(slice_index_240) // 10)))
