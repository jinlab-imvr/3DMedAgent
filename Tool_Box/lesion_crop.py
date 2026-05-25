# Tool_Box/lesion_crop.py

import os
from typing import Dict, Optional

import numpy as np
import nibabel as nib
import cv2

# 复用你已有的函数
from .slice_selection import extract_lesion_slices_from_report


# organ -> lesion mask 文件名映射
ORGAN_LESION_MASK_MAP = {
    "liver": "hepatic tumor.nii.gz",
    "pancreas": "pancreatic tumor.nii.gz",
    "kidney": "kidney cyst.nii.gz",
    # 其他器官暂不提供
}


def get_lesion_mask_path(case_dir: str, organ_name: str) -> Optional[str]:
    """
    根据 organ 找到对应的 lesion mask 路径。
    organ_name: 'liver' / 'pancreas' / 'kidney'
    """
    organ = organ_name.lower()
    if organ not in ORGAN_LESION_MASK_MAP:
        return None

    fname = ORGAN_LESION_MASK_MAP[organ]
    path = os.path.join(case_dir, fname)
    if os.path.exists(path):
        return path
    return None


def crop_and_zoom_lesion(
    volume: np.ndarray,         # (H, W, D)，主程序已经读取好的 CT
    case_dir: str,              # 当前 case 的文件夹，用来找 mask
    organ_name: str,            # 'liver' / 'pancreas' / 'kidney'
    report_text: str,           # 原始结构化报告文本
    output_size: int = 256,
    margin_ratio: float = 0.5,
) -> Optional[Dict]:

    # -------- 1. 基本检查 --------
    if volume is None or volume.ndim != 3:
        return None

    H, W, D = volume.shape

    # -------- 2. 从报告中抽取该器官的 lesion slices --------
    lesion_indices = extract_lesion_slices_from_report(
        report_text=report_text,
        organ_name=organ_name,
    )

    if not lesion_indices:
        # 报告里没提到这个器官的病灶
        return None

    # clip 到合法范围，假设报告里的 index 对应 D 轴
    lesion_indices = [
        int(np.clip(i, 0, D - 1)) for i in lesion_indices
    ]
    lesion_indices = sorted(set(lesion_indices))

    # -------- 3. 读取 organ 对应的 lesion mask --------
    mask_path = get_lesion_mask_path(case_dir, organ_name)
    if mask_path is None:
        return None

    try:
        mask_nii = nib.load(mask_path)
        mask = mask_nii.get_fdata()
    except Exception:
        return None

    # 强制要求和 volume 形状一致 (H, W, D)
    if mask.shape != volume.shape:
        return None

    mask = mask > 0  # 二值化

    # -------- 4. 在报告给出的 slices 中，选出 mask 面积最大的那一层 --------
    best_slice = None
    best_area = 0

    for idx in lesion_indices:
        area = mask[:, :, idx].sum()
        if area > best_area:
            best_area = area
            best_slice = idx

    if best_slice is None or best_area == 0:
        # 报告有 slice，但 mask 上没有有效病灶
        return None

    # -------- 5. 在该 slice 上计算 lesion bbox，并加 margin --------
    m2d = mask[:, :, best_slice]        # (H, W)
    ys, xs = np.where(m2d)
    if ys.size == 0 or xs.size == 0:
        return None

    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()

    bw = x2 - x1
    bh = y2 - y1

    # 加一点上下文 margin
    x1 = max(0, int(x1 - margin_ratio * bw))
    x2 = min(W, int(x2 + margin_ratio * bw))
    y1 = max(0, int(y1 - margin_ratio * bh))
    y2 = min(H, int(y2 + margin_ratio * bh))

    # -------- 6. 生成 full slice 和 crop + zoom --------
    ct_slice = volume[:, :, best_slice].astype(np.float32)  # (H, W)

    # full slice：直接 resize 到 output_size
    full_slice = cv2.resize(
        ct_slice,
        (output_size, output_size),
        interpolation=cv2.INTER_LINEAR,
    )

    # lesion crop：先裁剪，再 resize
    lesion_crop = ct_slice[y1:y2, x1:x2]
    lesion_crop = cv2.resize(
        lesion_crop,
        (output_size, output_size),
        interpolation=cv2.INTER_LINEAR,
    )
    result = {
        "slice_index": int(best_slice),
        "full_slice": full_slice,
        "lesion_crop": lesion_crop,
        "bbox": (int(x1), int(y1), int(x2), int(y2)),
    }
    return result
