# process_organs/pancreas.py

import os
import numpy as np
from scipy import ndimage

from utils.io import (
    load_canonical,
    resample_image,
    load_segments_pancreas,  # 需要在 utils/io.py 里实现
)
from utils.image_process import (
    measure_volume,
    measure_organ_hu,
)


# ---------------------- 1. Organ 级别分析 ---------------------- #

def generate_pancreas_organ_report(ct, pancreas_mask, spacing, phase=None, spleen_hu=None):
    """
    Pancreas organ 分析：
      - 体积
      - 大小是否增大（>83 ml）
      - 是否脂肪胰（P/S <= 0.7）
      - HU 均值/标准差
    """

    vol = measure_volume(pancreas_mask, spacing=spacing, check_border=False)
    organ_hu, organ_hu_std = measure_organ_hu(pancreas_mask, 0 * pancreas_mask, ct)

    att = 'normal'
    if vol is not None:
        if vol / 1000 > 83:
            size = 'large'
        else:
            size = 'normal'
        if spleen_hu is not None and organ_hu is not None:
            if organ_hu / spleen_hu <= 0.7:
                att = 'fatty'
    else:
        size = None

    text = "Pancreas: \n"

    if vol is not None:
        if size == 'normal':
            text += "Normal size "
        elif size == 'large':
            text += "Pancreas is enlarged "
        text += f"(volume: {np.round(vol/1000, 1)} cm^3).\n"

    if phase is not None and phase == 'Plain' and spleen_hu is not None and organ_hu is not None:
        if att == 'fatty':
            text += (
                f"Fatty infiltration, mean HU value: {np.round(organ_hu,1)} +/- {np.round(organ_hu_std,1)}, "
                f"pancreatic index (P/S): {np.round(organ_hu/spleen_hu, 2)}.\n"
            )
        else:
            text += (
                f"Normal attenuation, mean HU value: {np.round(organ_hu,1)} +/- {np.round(organ_hu_std,1)}, "
                f"pancreatic index (P/S): {np.round(organ_hu/spleen_hu, 2)}.\n"
            )
    else:
        text += f"Mean HU value: {np.round(organ_hu,1)} +/- {np.round(organ_hu_std,1)}.\n"

    return text, vol, organ_hu, organ_hu_std, att


# ---------------------- 2. Segment 辅助 ---------------------- #

SEG_LABELS_PANCREAS = {
    1: 'head',
    2: 'body',
    3: 'tail',
}


def locate_pancreas_segments(component_mask, pancreas_segments):
    if pancreas_segments is None:
        return None

    overlapping = pancreas_segments[component_mask > 0.5]
    overlapping = overlapping[overlapping > 0]
    if overlapping.size == 0:
        return None

    unique, counts = np.unique(overlapping, return_counts=True)
    ids = unique[np.argsort(counts)[::-1]]
    descs = []
    for sid in ids:
        name = SEG_LABELS_PANCREAS.get(int(sid))
        if name:
            descs.append(name)
    return descs if descs else None


# ---------------------- 3. Lesion 分析 ---------------------- #

def analyze_pancreas_lesion_mask(
    lesion_mask,
    ct,
    pancreas_mask,
    pancreas_segments=None,
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

    organ_hu, organ_hu_std = measure_organ_hu(pancreas_mask, 0 * pancreas_mask, ct)

    header_written = False
    text = ""

    for comp_label in range(1, num + 1):
        comp = (labeled == comp_label)
        voxels = int(comp.sum())
        if voxels < min_voxels:
            continue

        # ---------- 新增：pancreas 内 overlap 过滤 ----------
        overlap_voxels = np.logical_and(comp, pancreas_mask > 0.5).sum()
        overlap_ratio = overlap_voxels / float(voxels)
        if overlap_ratio < overlap_threshold:
            continue
        # --------------------------------------------------

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

        if organ_hu is not None:
            if mean_hu < organ_hu:
                enh = "Hypoattenuating"
            elif mean_hu > organ_hu:
                enh = "Hyperattenuating"
            else:
                enh = "Isoattenuating"
        else:
            enh = "Lesion attenuation relative to pancreas cannot be determined"

        loc = locate_pancreas_segments(comp, pancreas_segments)

        if not header_written:
            if lesion_type in ["PDAC", "PNET"]:
                text += f"Pancreatic {lesion_type}s:\n"
            elif lesion_type == "cyst":
                text += "Pancreatic cysts:\n"
            elif lesion_type == "malignant tumor":
                text += "Pancreatic malignant tumors:\n"
            else:
                text += "Pancreatic lesions:\n"
            header_written = True

        text += f"Pancreas {lesion_type} {comp_label}: \n"
        if loc:
            text += "Location: pancreas " + "/".join(loc) + ".\n"

        slice_str = f"{slice_idx_original}" if slice_idx_original is not None else "unknown"
        text += (
            f"Size: {np.round(longest/10.0, 1)} x {np.round(perp/10.0, 1)} cm "
            f"(approx., slice {slice_str}). "
        )
        text += f"Volume: {np.round(vol_cm3, 1)} cm^3.\n"
        text += (
            f"Enhancement relative to pancreas: {enh} "
            f"(HU value is {np.round(mean_hu,1)} +/- {np.round(std_hu,1)}).\n\n"
        )

    return text.strip()


def generate_pancreas_lesion_report(
    ct,
    pancreas_mask,
    pancreas_segments=None,
    pdac_mask=None,
    pnet_mask=None,
    tumor_mask=None,
    cyst_mask=None,
    lesion_mask=None,
    original_spacing=None,
    original_ct_shape=None,
    target_spacing=(1.0, 1.0, 1.0),
):
    parts = []

    if pdac_mask is not None:
        t = analyze_pancreas_lesion_mask(
            pdac_mask, ct, pancreas_mask,
            pancreas_segments=pancreas_segments,
            lesion_type="PDAC",
            original_spacing=original_spacing,
            original_ct_shape=original_ct_shape,
            target_spacing=target_spacing,
        )
        if t:
            parts.append(t)

    if pnet_mask is not None:
        t = analyze_pancreas_lesion_mask(
            pnet_mask, ct, pancreas_mask,
            pancreas_segments=pancreas_segments,
            lesion_type="PNET",
            original_spacing=original_spacing,
            original_ct_shape=original_ct_shape,
            target_spacing=target_spacing,
        )
        if t:
            parts.append(t)

    if tumor_mask is not None:
        t = analyze_pancreas_lesion_mask(
            tumor_mask, ct, pancreas_mask,
            pancreas_segments=pancreas_segments,
            lesion_type="malignant tumor",
            original_spacing=original_spacing,
            original_ct_shape=original_ct_shape,
            target_spacing=target_spacing,
        )
        if t:
            parts.append(t)

    if cyst_mask is not None:
        t = analyze_pancreas_lesion_mask(
            cyst_mask, ct, pancreas_mask,
            pancreas_segments=pancreas_segments,
            lesion_type="cyst",
            original_spacing=original_spacing,
            original_ct_shape=original_ct_shape,
            target_spacing=target_spacing,
        )
        if t:
            parts.append(t)

    if lesion_mask is not None:
        t = analyze_pancreas_lesion_mask(
            lesion_mask, ct, pancreas_mask,
            pancreas_segments=pancreas_segments,
            lesion_type="lesion",
            original_spacing=original_spacing,
            original_ct_shape=original_ct_shape,
            target_spacing=target_spacing,
        )
        if t:
            parts.append(t)

    if not parts:
        return ""

    return "\n\n".join(parts)


# ---------------------- 4. 主入口：process_pancreas_case ---------------------- #

def process_pancreas_case(
    ct_path,
    pancreas_mask_path,
    pancreas_segments_dir=None,
    pdac_mask_path=None,
    pnet_mask_path=None,
    tumor_mask_path=None,
    cyst_mask_path=None,
    lesion_mask_path=None,
    phase=None,
    spleen_hu=None,
):
    print("Processing pancreas case:")
    print(f"  CT:               {ct_path}")
    print(f"  Pancreas mask:    {pancreas_mask_path}")
    print(f"  Segment dir:      {pancreas_segments_dir}")
    print(f"  PDAC mask:        {pdac_mask_path}")
    print(f"  PNET mask:        {pnet_mask_path}")
    print(f"  Tumor mask:       {tumor_mask_path}")
    print(f"  Cyst mask:        {cyst_mask_path}")
    print(f"  Lesion mask:      {lesion_mask_path}")

    ct_img = load_canonical(ct_path)
    original_spacing = ct_img.header.get_zooms()
    original_ct_shape = ct_img.shape
    ct = ct_img.get_fdata()

    spacing = original_spacing
    target_spacing = (1.0, 1.0, 1.0)

    pancreas = load_canonical(pancreas_mask_path).get_fdata().astype("uint8")

    pancreas, _ = resample_image(pancreas, original_spacing=spacing,
                                 target_spacing=target_spacing, order=0)
    ct, _ = resample_image(ct, original_spacing=spacing,
                           target_spacing=target_spacing)
    pancreas = (pancreas > 0.5).astype("float32")

    pancreas_segments = None
    if pancreas_segments_dir is not None and os.path.isdir(pancreas_segments_dir):
        try:
            pancreas_segments = load_segments_pancreas(pancreas_segments_dir, spacing)
        except Exception as e:
            print(f"[WARN] Failed to load pancreas segments from {pancreas_segments_dir}: {e}")

    def _load(path):
        if path is None or (not os.path.isfile(path)):
            return None
        m = load_canonical(path).get_fdata().astype("uint8")
        m, _ = resample_image(m, original_spacing=spacing,
                              target_spacing=target_spacing, order=0)
        return (m > 0.5).astype("float32")

    pdac = _load(pdac_mask_path)
    pnet = _load(pnet_mask_path)
    tumor = _load(tumor_mask_path)
    cyst = _load(cyst_mask_path)
    lesion = _load(lesion_mask_path)

    organ_text, vol, organ_hu, organ_hu_std, att = generate_pancreas_organ_report(
        ct=ct,
        pancreas_mask=pancreas,
        spacing=spacing,
        phase=phase,
        spleen_hu=spleen_hu,
    )

    lesion_text = generate_pancreas_lesion_report(
        ct=ct,
        pancreas_mask=pancreas,
        pancreas_segments=pancreas_segments,
        pdac_mask=pdac,
        pnet_mask=pnet,
        tumor_mask=tumor,
        cyst_mask=cyst,
        lesion_mask=lesion,
        original_spacing=original_spacing,
        original_ct_shape=original_ct_shape,
        target_spacing=target_spacing,
    )

    if lesion_text:
        full = organ_text.strip() + "\n\n" + lesion_text.strip()
    else:
        full = organ_text.strip()

    return full