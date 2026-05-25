# process_organs/kidney.py

import os
import numpy as np
from scipy import ndimage

from utils.io import (
    load_canonical,
    resample_image,
    load_segments_kidney,   # 需要在 utils/io.py 里实现
)
from utils.image_process import (
    measure_volume,
    measure_organ_hu,
)


# ---------------------- 1. Organ 级别分析 ---------------------- #

def generate_kidney_organ_report(ct, kidney_right, kidney_left, spacing):
    """
    双肾 organ 级别分析：
      - 右 / 左 / 总体积
      - 是否增大
      - HU 平均值（对整合的双肾）
    """

    vol_right = measure_volume(kidney_right, spacing=spacing, check_border=False)
    vol_left = measure_volume(kidney_left, spacing=spacing, check_border=False)

    if vol_right is None or vol_left is None:
        vol_total = None
    else:
        vol_total = vol_right + vol_left

    union = np.maximum(kidney_right, kidney_left)
    organ_hu, organ_hu_std = measure_organ_hu(union, 0 * union, ct)

    size_right = 'normal'
    size_left = 'normal'

    if vol_right is not None and vol_right / 1000 > (415.2 / 2):
        size_right = 'large'
    if vol_left is not None and vol_left / 1000 > (415.2 / 2):
        size_left = 'large'

    text = "Kidney: \n"

    if vol_total is not None:
        if size_right == 'normal' and size_left == 'normal':
            text += "Normal size "
        else:
            text += "Bilateral kidneys are enlarged "

        text += (
            f"(right kidney volume: {np.round(vol_right/1000, 1)} cm^3; "
            f"left kidney volume: {np.round(vol_left/1000, 1)} cm^3; "
            f"total kidney volume: {np.round(vol_total/1000, 1)} cm^3).\n"
        )

    text += f"Mean HU value: {np.round(organ_hu, 1)} +/- {np.round(organ_hu_std, 1)}.\n"

    return text, vol_right, vol_left, vol_total, organ_hu, organ_hu_std


# ---------------------- 2. Lesion 定位辅助 ---------------------- #

def locate_kidney_lesion_side(component_mask, kidney_segments):
    """
    使用 kidney segments（1: left, 2: right）判定病灶是左肾还是右肾
    """
    if kidney_segments is None:
        return None

    overlapping = kidney_segments[component_mask > 0.5]
    overlapping = overlapping[overlapping > 0]
    if overlapping.size == 0:
        return None

    unique, counts = np.unique(overlapping, return_counts=True)
    label = unique[np.argmax(counts)]

    if label == 1:
        return "left kidney"
    elif label == 2:
        return "right kidney"
    else:
        return None


# ---------------------- 3. Lesion 分析 ---------------------- #

def analyze_kidney_lesion_mask(
    lesion_mask,
    ct,
    union_kidney_mask,
    kidney_segments=None,
    lesion_type="lesion",
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

    kidney_hu, kidney_hu_std = measure_organ_hu(
        union_kidney_mask, 0 * union_kidney_mask, ct
    )

    header_written = False
    text = ""

    for comp_label in range(1, num + 1):
        comp = (labeled == comp_label)
        voxels = int(comp.sum())
        if voxels < min_voxels:
            continue

        # ---------- 新增：肾内 overlap 过滤 ----------
        overlap_voxels = np.logical_and(comp, union_kidney_mask > 0.5).sum()
        overlap_ratio = overlap_voxels / float(voxels)
        if overlap_ratio < overlap_threshold:
            continue
        # -------------------------------------------

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

        lesion_hu = ct[comp > 0]
        mean_hu = float(lesion_hu.mean())
        std_hu = float(lesion_hu.std()) if lesion_hu.size > 1 else 0.0

        if kidney_hu is not None:
            if mean_hu < kidney_hu:
                enh = "Hypoattenuating"
            elif mean_hu > kidney_hu:
                enh = "Hyperattenuating"
            else:
                enh = "Isoattenuating"
        else:
            enh = "Lesion attenuation relative to kidney cannot be determined"

        side = locate_kidney_lesion_side(comp, kidney_segments)

        if not header_written:
            if lesion_type == "malignant tumor":
                text += "Kidney malignant tumors:\n"
            elif lesion_type == "cyst":
                text += "Kidney cysts:\n"
            else:
                text += "Kidney lesions:\n"
            header_written = True

        text += f"Kidney {lesion_type} {comp_label}: \n"
        if side is not None:
            text += f"Location: {side}.\n"

        slice_str = f"{slice_idx_original}" if slice_idx_original is not None else "unknown"
        text += (
            f"Size: {np.round(longest/10.0, 1)} x {np.round(perp/10.0, 1)} cm "
            f"(approx., slice {slice_str}). "
        )
        text += f"Volume: {np.round(vol_cm3, 1)} cm^3.\n"
        text += (
            f"Enhancement relative to kidneys: {enh} "
            f"(HU value is {np.round(mean_hu, 1)} +/- {np.round(std_hu, 1)}).\n\n"
        )

    return text.strip()


def generate_kidney_lesion_report(
    ct,
    kidney_right,
    kidney_left,
    kidney_segments=None,
    tumor_mask=None,
    cyst_mask=None,
    lesion_mask=None,
    original_spacing=None,
    original_ct_shape=None,
    target_spacing=(1.0, 1.0, 1.0),
):
    union = np.maximum(kidney_right, kidney_left)
    parts = []

    if tumor_mask is not None:
        t = analyze_kidney_lesion_mask(
            tumor_mask, ct, union,
            kidney_segments=kidney_segments,
            lesion_type="malignant tumor",
            original_spacing=original_spacing,
            original_ct_shape=original_ct_shape,
            target_spacing=target_spacing,
        )
        if t:
            parts.append(t)

    if cyst_mask is not None:
        t = analyze_kidney_lesion_mask(
            cyst_mask, ct, union,
            kidney_segments=kidney_segments,
            lesion_type="cyst",
            original_spacing=original_spacing,
            original_ct_shape=original_ct_shape,
            target_spacing=target_spacing,
        )
        if t:
            parts.append(t)

    if lesion_mask is not None:
        t = analyze_kidney_lesion_mask(
            lesion_mask, ct, union,
            kidney_segments=kidney_segments,
            lesion_type="lesion",
            original_spacing=original_spacing,
            original_ct_shape=original_ct_shape,
            target_spacing=target_spacing,
        )
        if t:
            parts.append(t)

    return "\n\n".join(parts) if parts else ""


# ---------------------- 4. 主入口：process_kidney_case ---------------------- #

def process_kidney_case(
    ct_path,
    kidney_right_mask_path,
    kidney_left_mask_path=None,
    kidney_segments_dir=None,
    tumor_mask_path=None,
    cyst_mask_path=None,
    lesion_mask_path=None,
):
    print("Processing kidney case:")
    print(f"  CT:                 {ct_path}")
    print(f"  Right kidney mask:  {kidney_right_mask_path}")
    print(f"  Left  kidney mask:  {kidney_left_mask_path}")
    print(f"  Segment dir:        {kidney_segments_dir}")
    print(f"  Tumor mask:         {tumor_mask_path}")
    print(f"  Cyst mask:          {cyst_mask_path}")
    print(f"  Lesion mask:        {lesion_mask_path}")

    ct_img = load_canonical(ct_path)
    original_spacing = ct_img.header.get_zooms()
    original_ct_shape = ct_img.shape
    ct = ct_img.get_fdata()

    spacing = original_spacing
    target_spacing = (1.0, 1.0, 1.0)

    right = load_canonical(kidney_right_mask_path).get_fdata().astype("uint8")

    if kidney_left_mask_path is None:
        if "kidney_right" in kidney_right_mask_path:
            kidney_left_mask_path = kidney_right_mask_path.replace(
                "kidney_right", "kidney_left"
            )

    if kidney_left_mask_path is None or not os.path.isfile(kidney_left_mask_path):
        raise ValueError(f"Left kidney mask not found for {kidney_right_mask_path}")

    left = load_canonical(kidney_left_mask_path).get_fdata().astype("uint8")

    right, _ = resample_image(right, spacing, target_spacing, order=0)
    left, _ = resample_image(left, spacing, target_spacing, order=0)
    ct, _ = resample_image(ct, spacing, target_spacing)

    right = (right > 0.5).astype("float32")
    left = (left > 0.5).astype("float32")

    kidney_segments = None
    if kidney_segments_dir is not None and os.path.isdir(kidney_segments_dir):
        try:
            kidney_segments = load_segments_kidney(kidney_segments_dir, spacing)
        except Exception as e:
            print(f"[WARN] Failed to load kidney segments from {kidney_segments_dir}: {e}")

    def _load_lesion(path):
        if path is None or not os.path.isfile(path):
            return None
        m = load_canonical(path).get_fdata().astype("uint8")
        m, _ = resample_image(m, spacing, target_spacing, order=0)
        return (m > 0.5).astype("float32")

    tumor = _load_lesion(tumor_mask_path)
    cyst = _load_lesion(cyst_mask_path)
    lesion = _load_lesion(lesion_mask_path)

    organ_text, *_ = generate_kidney_organ_report(
        ct=ct,
        kidney_right=right,
        kidney_left=left,
        spacing=spacing,
    )

    lesion_text = generate_kidney_lesion_report(
        ct=ct,
        kidney_right=right,
        kidney_left=left,
        kidney_segments=kidney_segments,
        tumor_mask=tumor,
        cyst_mask=cyst,
        lesion_mask=lesion,
        original_spacing=original_spacing,
        original_ct_shape=original_ct_shape,
        target_spacing=target_spacing,
    )

    return (
        organ_text.strip() + "\n\n" + lesion_text.strip()
        if lesion_text else organ_text.strip()
    )
