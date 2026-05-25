# process_organs/colon.py

import os
import numpy as np
from scipy import ndimage

from utils.io import load_canonical, resample_image
from utils.image_process import measure_volume, measure_organ_hu


# ---------------------- 1. Organ 级别分析 ---------------------- #

def generate_colon_organ_report(colon_mask, spacing, ct=None):
    """
    Colon organ 报告：仅体积（保持原逻辑简单）
    """
    vol = measure_volume(colon_mask, spacing=spacing, check_border=False)
    text = "Colon:\n"
    if vol is not None:
        text += f"Volume: {np.round(vol / 1000, 1)} cm^3.\n"
    return text, vol


# ---------------------- 2. Lesion 级别分析 ---------------------- #

def analyze_colon_lesion_mask(
    lesion_mask,
    ct,
    colon_mask,
    min_voxels=10,
    original_spacing=None,
    original_ct_shape=None,
    target_spacing=(1.0, 1.0, 1.0),
    overlap_threshold=0.5,
):
    if lesion_mask is None:
        return ""

    bin_mask = (lesion_mask > 0.5).astype(np.uint8)
    if bin_mask.sum() == 0:
        return ""

    labeled, num = ndimage.label(bin_mask)
    if num == 0:
        return ""

    organ_hu, organ_hu_std = (None, None)
    if ct is not None and colon_mask is not None:
        organ_hu, organ_hu_std = measure_organ_hu(colon_mask, 0 * colon_mask, ct)

    text = ""
    header_written = False

    for comp_label in range(1, num + 1):
        comp = (labeled == comp_label)
        voxels = int(comp.sum())
        if voxels < min_voxels:
            continue

        # ---------- 新增：colon 内 overlap 过滤 ----------
        overlap_voxels = np.logical_and(comp, colon_mask > 0.5).sum()
        overlap_ratio = overlap_voxels / float(voxels)
        if overlap_ratio < overlap_threshold:
            continue
        # -----------------------------------------------

        vol_mm3 = voxels * 1.0
        vol_cm3 = vol_mm3 / 1000.0

        idx = np.where(comp)
        x_span = idx[0].max() - idx[0].min() + 1
        y_span = idx[1].max() - idx[1].min() + 1
        longest = max(x_span, y_span)
        perp = min(x_span, y_span)

        # ---------- slice 映射回原始 CT ----------
        z_mean_resampled = float(np.mean(idx[2]))
        slice_idx_original = None
        if original_spacing is not None and original_ct_shape is not None:
            sz = float(original_spacing[2])
            tz = float(target_spacing[2])
            z_mm = z_mean_resampled * tz
            slice_idx_original = int(round(z_mm / sz))
            slice_idx_original = max(
                0, min(slice_idx_original, original_ct_shape[2] - 1)
            )
        # ----------------------------------------

        if ct is not None:
            lesion_hu = ct[comp > 0]
            mean_hu = float(lesion_hu.mean())
            std_hu = float(lesion_hu.std()) if lesion_hu.size > 1 else 0.0
        else:
            mean_hu, std_hu = None, None

        if organ_hu is not None and mean_hu is not None:
            if mean_hu < organ_hu:
                enh = "Hypoattenuating"
            elif mean_hu > organ_hu:
                enh = "Hyperattenuating"
            else:
                enh = "Isoattenuating"
        else:
            enh = "Lesion attenuation relative to colon cannot be determined"

        if not header_written:
            text += "Colon lesions:\n"
            header_written = True

        text += f"Colon lesion {comp_label}:\n"
        slice_str = f"{slice_idx_original}" if slice_idx_original is not None else "unknown"
        text += (
            f"Size: {np.round(longest / 10.0, 1)} x {np.round(perp / 10.0, 1)} cm "
            f"(approx., slice {slice_str}). "
        )
        text += f"Volume: {np.round(vol_cm3, 1)} cm^3.\n"
        if mean_hu is not None:
            text += (
                f"Enhancement relative to colon: {enh} "
                f"(HU value is {np.round(mean_hu, 1)} +/- {np.round(std_hu, 1)}).\n"
            )
        text += "\n"

    return text.strip()


# ---------------------- 3. 主入口 ---------------------- #

def process_colon_case(
    ct_path,
    colon_mask_path,
    colon_lesion_mask_path=None,
):
    print("Processing colon case:")
    print(f"  CT:           {ct_path}")
    print(f"  Colon mask:   {colon_mask_path}")
    print(f"  Lesion mask:  {colon_lesion_mask_path}")

    ct_img = load_canonical(ct_path)
    original_spacing = ct_img.header.get_zooms()
    original_ct_shape = ct_img.shape
    ct = ct_img.get_fdata()

    spacing = original_spacing
    target_spacing = (1.0, 1.0, 1.0)

    colon = load_canonical(colon_mask_path).get_fdata().astype("uint8")
    colon, _ = resample_image(colon, spacing, target_spacing, order=0)
    ct, _ = resample_image(ct, spacing, target_spacing)
    colon = (colon > 0.5).astype("float32")

    lesion = None
    if colon_lesion_mask_path is not None and os.path.isfile(colon_lesion_mask_path):
        m = load_canonical(colon_lesion_mask_path).get_fdata().astype("uint8")
        m, _ = resample_image(m, spacing, target_spacing, order=0)
        lesion = (m > 0.5).astype("float32")

    organ_text, vol = generate_colon_organ_report(colon, spacing, ct=ct)

    lesion_text = analyze_colon_lesion_mask(
        lesion_mask=lesion,
        ct=ct,
        colon_mask=colon,
        original_spacing=original_spacing,
        original_ct_shape=original_ct_shape,
        target_spacing=target_spacing,
    ) if lesion is not None else ""

    if lesion_text:
        full = organ_text.strip() + "\n\n" + lesion_text.strip()
    else:
        full = organ_text.strip()

    return full
