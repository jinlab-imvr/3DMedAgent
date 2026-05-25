# process_organs/liver.py

import os
import numpy as np
from scipy import ndimage

from utils.io import load_canonical, resample_image, load_segments_liver
from utils.image_process import (
    measure_volume,
    measure_organ_hu
)

# ---------------------- 1. Organ 级别分析 ---------------------- #

def generate_liver_organ_report(ct, liver_mask, spacing, phase=None, spleen_hu=None):
    vol = measure_volume(liver_mask, spacing=spacing, check_border=False)
    organ_hu, organ_hu_std = measure_organ_hu(liver_mask, 0 * liver_mask, ct)

    att = 'normal'
    size = None
    if vol is not None:
        size = 'normal'
        if vol / 1000 > 3000:
            size = 'large'
        if organ_hu is not None and organ_hu <= 40:
            att = 'fatty'

    text = "Liver: \n"

    if vol is not None:
        if size == 'normal':
            text += "Normal size "
        elif size == 'large':
            text += "Liver is enlarged "
        elif size == 'massive':
            text += "Liver is massively enlarged "

        text += f"(volume: {np.round(vol / 1000, 1)} cm^3).\n"

    if phase == 'Plain' and organ_hu is not None:
        if att == 'fatty':
            text += f"Fatty infiltration (Mean HU value: {np.round(organ_hu, 1)} +/- {np.round(organ_hu_std, 1)}).\n"
        else:
            text += f"Normal attenuation (Mean HU value: {np.round(organ_hu, 1)} +/- {np.round(organ_hu_std, 1)}).\n"
    else:
        text += f"Mean HU value: {np.round(organ_hu, 1)} +/- {np.round(organ_hu_std, 1)}.\n"

    return text, vol, organ_hu, organ_hu_std, att


# ---------------------- 2. Segment 定位辅助 ---------------------- #

def locate_lesion_segments(lesion_mask, liver_segments):
    if liver_segments is None:
        return None

    overlapping = liver_segments[lesion_mask > 0.5]
    overlapping = overlapping[overlapping > 0]
    if overlapping.size == 0:
        return None

    unique, counts = np.unique(overlapping, return_counts=True)
    order = np.argsort(counts)[::-1]
    return [int(s) for s in unique[order]]


# ---------------------- 3. Lesion 级别分析 ---------------------- #

def analyze_liver_lesion_mask(
    lesion_mask,
    ct,
    liver_mask,
    liver_hu_mean,
    liver_hu_std,
    liver_segments=None,
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

    text = ""
    header_written = False

    for comp_label in range(1, num + 1):
        comp = (labeled == comp_label)
        voxels = int(comp.sum())
        if voxels < min_voxels:
            continue

        # ---------- 新增：肝内 overlap 过滤 ----------
        overlap_voxels = np.logical_and(comp, liver_mask > 0.5).sum()
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
        perpendicular = min(x_span, y_span)

        # slice 映射回原始 CT
        z_mean_resampled = float(np.mean(idx[2]))
        slice_idx_original = None
        if original_spacing is not None and original_ct_shape is not None:
            sz = float(original_spacing[2])
            tz = float(target_spacing[2])
            z_mm = z_mean_resampled * tz
            slice_idx_original = int(round(z_mm / sz))
            slice_idx_original = max(0, min(slice_idx_original, original_ct_shape[2] - 1))

        lesion_hu = ct[comp > 0]
        mean_hu = float(lesion_hu.mean())
        std_hu = float(lesion_hu.std()) if lesion_hu.size > 1 else 0.0

        if liver_hu_mean is not None:
            if mean_hu < liver_hu_mean:
                enh = "Hypoattenuating"
            elif mean_hu > liver_hu_mean:
                enh = "Hyperattenuating"
            else:
                enh = "Isoattenuating"
        else:
            enh = "Lesion attenuation relative to liver cannot be determined"

        seg_desc = ""
        if liver_segments is not None:
            segs = locate_lesion_segments(comp, liver_segments)
            if segs:
                seg_desc = "hepatic segment " + "/".join(str(s) for s in segs)

        if not header_written:
            if lesion_type == "malignant tumor":
                text += "Liver malignant tumors:\n"
            elif lesion_type == "cyst":
                text += "Liver cysts:\n"
            else:
                text += "Liver lesions:\n"
            header_written = True

        text += f"Liver {lesion_type} {comp_label}:\n"
        if seg_desc:
            text += f"Location: {seg_desc}.\n"

        slice_str = f"{slice_idx_original}" if slice_idx_original is not None else "unknown"

        text += (
            f"Size: {np.round(longest / 10.0, 1)} x {np.round(perpendicular / 10.0, 1)} cm "
            f"(approx., slice {slice_str}). "
        )
        text += f"Volume: {np.round(vol_cm3, 1)} cm^3.\n"
        text += (
            f"Enhancement relative to liver: {enh} "
            f"(HU value is {np.round(mean_hu, 1)} +/- {np.round(std_hu, 1)}).\n\n"
        )

    return text.strip()


def generate_liver_lesion_report(
    ct,
    spacing,
    liver_mask,
    liver_segments=None,
    tumor_mask=None,
    cyst_mask=None,
    lesion_mask=None,
    original_spacing=None,
    original_ct_shape=None,
    target_spacing=(1.0, 1.0, 1.0),
):
    liver_hu_mean, liver_hu_std = measure_organ_hu(liver_mask, 0 * liver_mask, ct)

    parts = []

    if tumor_mask is not None:
        t = analyze_liver_lesion_mask(
            tumor_mask, ct, liver_mask,
            liver_hu_mean, liver_hu_std,
            liver_segments, "malignant tumor",
            original_spacing=original_spacing,
            original_ct_shape=original_ct_shape,
            target_spacing=target_spacing,
        )
        if t:
            parts.append(t)

    if cyst_mask is not None:
        t = analyze_liver_lesion_mask(
            cyst_mask, ct, liver_mask,
            liver_hu_mean, liver_hu_std,
            liver_segments, "cyst",
            original_spacing=original_spacing,
            original_ct_shape=original_ct_shape,
            target_spacing=target_spacing,
        )
        if t:
            parts.append(t)

    if lesion_mask is not None:
        t = analyze_liver_lesion_mask(
            lesion_mask, ct, liver_mask,
            liver_hu_mean, liver_hu_std,
            liver_segments, "lesion",
            original_spacing=original_spacing,
            original_ct_shape=original_ct_shape,
            target_spacing=target_spacing,
        )
        if t:
            parts.append(t)

    return "\n\n".join(parts) if parts else ""


# ---------------------- 4. 主入口 ---------------------- #

def process_liver_case(
    ct_path,
    liver_mask_path,
    liver_segments_dir=None,
    tumor_mask_path=None,
    cyst_mask_path=None,
    lesion_mask_path=None,
    phase=None,
    spleen_hu=None,
):
    ct_img = load_canonical(ct_path)
    original_spacing = ct_img.header.get_zooms()
    original_ct_shape = ct_img.shape
    ct = ct_img.get_fdata()

    spacing = original_spacing
    target_spacing = (1.0, 1.0, 1.0)

    liver = load_canonical(liver_mask_path).get_fdata().astype("uint8")

    liver, _ = resample_image(liver, spacing, target_spacing, order=0)
    ct, _ = resample_image(ct, spacing, target_spacing)
    liver = (liver > 0.5).astype("float32")

    liver_segments = None
    if liver_segments_dir and os.path.isdir(liver_segments_dir):
        try:
            liver_segments = load_segments_liver(liver_segments_dir, spacing)
        except Exception as e:
            print(f"[WARN] Failed to load liver segments: {e}")

    def _load_mask(p):
        if p is None or not os.path.isfile(p):
            return None
        m = load_canonical(p).get_fdata().astype("uint8")
        m, _ = resample_image(m, spacing, target_spacing, order=0)
        return (m > 0.5).astype("float32")

    tumor_mask = _load_mask(tumor_mask_path)
    cyst_mask = _load_mask(cyst_mask_path)
    lesion_mask = _load_mask(lesion_mask_path)

    organ_text, *_ = generate_liver_organ_report(
        ct, liver, spacing, phase, spleen_hu
    )

    lesion_text = generate_liver_lesion_report(
        ct=ct,
        spacing=spacing,
        liver_mask=liver,
        liver_segments=liver_segments,
        tumor_mask=tumor_mask,
        cyst_mask=cyst_mask,
        lesion_mask=lesion_mask,
        original_spacing=original_spacing,
        original_ct_shape=original_ct_shape,
        target_spacing=target_spacing,
    )

    return (
        organ_text.strip() + "\n\n" + lesion_text.strip()
        if lesion_text else organ_text.strip()
    )
