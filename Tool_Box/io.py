# Tool_Box/slice_io.py

import re
import os
import json
import cv2
import time
import pandas as pd
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from typing import List, Optional

# =========================================================
# CT Windowing (Abdominal default)
# =========================================================

def apply_ct_window(
    vol: np.ndarray,
    wl: float = 40.0,
    ww: float = 400.0,
) -> np.ndarray:
    """
    Apply CT windowing on HU volume.

    Args:
        vol: float ndarray (HU values)
        wl: window level
        ww: window width

    Returns:
        float ndarray in [0, 1]
    """
    low = wl - ww / 2.0
    high = wl + ww / 2.0
    vol = np.clip(vol, low, high)
    vol = (vol - low) / (high - low + 1e-6)
    return vol


# =========================================================
# Slice saving (assumes volume already windowed + uint8)
# =========================================================

def save_slices_from_volume(
    volume,
    slice_indices: List[int],
    out_dir: str,
    image_id: Optional[str] = None,
    report_path: Optional[str] = None,
    log_fn=print,
):
    """
    volume: (H, W, D) uint8, already windowed
    slice_indices: list of z indices
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = []

    for idx in sorted(slice_indices):
        try:
            img = volume[:, :, idx]
        except Exception as e:
            log_fn(
                f"[slice] index error image_id={image_id} "
                f"slice={idx} vol_shape={volume.shape} err={e}"
            )
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        p = os.path.join(out_dir, f"slice_{idx:03d}.png")

        ok = cv2.imwrite(p, img_rgb)
        if not ok:
            log_fn(
                f"[slice] write failed image_id={image_id} "
                f"slice={idx} path={p} report={report_path}"
            )
            continue

        if not os.path.exists(p) or os.path.getsize(p) == 0:
            log_fn(
                f"[slice] file empty image_id={image_id} "
                f"slice={idx} path={p} report={report_path}"
            )
            continue

        paths.append(p)

    return paths


# =========================================================
# Report reader
# =========================================================

def read_report_safely(
    report_path: str,
    read_retries: int = 5,
    read_sleep: float = 0.4,
    log_fn=print,
):
    last_err = None
    for attempt in range(read_retries):
        try:
            df = pd.read_csv(report_path)

            if df is None or df.shape[0] == 0:
                last_err = "empty dataframe"
                time.sleep(read_sleep * (attempt + 1))
                continue

            if "Report" not in df.columns:
                last_err = "missing 'Report' column"
                time.sleep(read_sleep * (attempt + 1))
                continue

            val = df.iloc[0]["Report"]
            if not isinstance(val, str) or len(val.strip()) == 0:
                last_err = "Report cell empty"
                time.sleep(read_sleep * (attempt + 1))
                continue

            return val

        except Exception as e:
            last_err = repr(e)
            time.sleep(read_sleep * (attempt + 1))

    log_fn(f"[report] bad report file: {report_path} reason={last_err}")
    return None


# =========================================================
# NIfTI canonical loading
# =========================================================

def load_canonical(src):
    """
    Load a NIfTI file, convert to RAS canonical orientation,
    enforce axis=2 as slice (S) axis.
    """
    if isinstance(src, (str, os.PathLike)):
        img = nib.load(src)
    elif isinstance(src, nib.spatialimages.SpatialImage):
        img = src
    else:
        raise TypeError("Expected path or NIfTI image.")

    img = nib.as_closest_canonical(img)

    cur_ornt = nib.orientations.io_orientation(img.affine)
    ras_ornt = nib.orientations.axcodes2ornt(("R", "A", "S"))
    transform = nib.orientations.ornt_transform(cur_ornt, ras_ornt)

    data = img.get_fdata()
    data_ras = nib.orientations.apply_orientation(data, transform)

    inv_aff = nib.orientations.inv_ornt_aff(transform, img.shape)
    new_affine = img.affine @ inv_aff

    out = nib.Nifti1Image(data_ras, new_affine, header=img.header)
    out.set_sform(new_affine, code=1)
    out.set_qform(new_affine, code=1)

    return out


# =========================================================
# Main loader for VLM pipeline (RECOMMENDED)
# =========================================================

def load_nii_keep_z(
    nii_path: str,
    wl: float = 40.0,
    ww: float = 400.0,
):
    """
    Load NIfTI, enforce canonical orientation,
    apply CT windowing (abdominal default),
    return uint8 volume (H, W, D).
    """
    img = load_canonical(nii_path)
    vol = img.get_fdata()

    if vol.ndim != 3:
        raise ValueError(f"Invalid NIfTI shape: {vol.shape}")

    # CT windowing (CRITICAL)
    vol = apply_ct_window(vol, wl=wl, ww=ww)

    return (vol * 255.0).astype(np.uint8)


# =========================================================
# Legacy / utility (NOT recommended for CT slices now)
# =========================================================

def load_and_preprocess_nii(
    nii_path: str,
    target_shape=(256, 256, 60),
):
    """
    NOTE:
    This function performs resize + global normalization.
    It is NOT recommended for CT visualization / VLM input.
    """
    vol = nib.load(nii_path).get_fdata()
    if vol.ndim != 3:
        raise ValueError(f"Invalid NIfTI shape: {vol.shape}")

    a, b, c = vol.shape
    depth_axis = np.argmin([a, b, c])
    if depth_axis != 2:
        vol = np.moveaxis(vol, depth_axis, 2)

    h, w, d = vol.shape
    th, tw, td = target_shape
    zoom_factors = (th / h, tw / w, td / d)

    vol = zoom(vol, zoom_factors, order=1, mode="nearest")
    vmin, vmax = vol.min(), vol.max()
    if vmax > vmin:
        vol = (vol - vmin) / (vmax - vmin)

    return (vol * 255).astype(np.uint8)


def safe_json_loads(text: str):
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def slice_to_uint8(img2d: np.ndarray) -> np.ndarray:
    """
    Legacy helper.
    NOT recommended for CT slices after windowing.
    """
    x = img2d.astype(np.float32)
    lo = np.percentile(x, 1.0)
    hi = np.percentile(x, 99.0)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(x)), float(np.max(x))
        if hi <= lo:
            return np.zeros_like(x, dtype=np.uint8)
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo + 1e-6)
    return (x * 255.0).astype(np.uint8)


def save_gray_png(img2d: np.ndarray, out_path: str):
    """
    Legacy helper.
    """
    u8 = slice_to_uint8(img2d)
    cv2.imwrite(out_path, u8)

def save_slice_with_contour_overlay(
    volume: np.ndarray,               # (H,W,D) uint8, 已 windowed
    slice_idx: int,
    mask2d_u8: np.ndarray,            # (H,W) uint8 0/1
    out_dir: str,
    image_id: Optional[str] = None,
    report_path: Optional[str] = None,
    thickness: int = 2,
):
    """
    保存两张图：
    1) 原始 slice_{idx:03d}.png
    2) 叠加 contour 的 slice_{idx:03d}_overlay.png

    Returns:
        (raw_path, overlay_path) 失败则 (None, None)
    """
    os.makedirs(out_dir, exist_ok=True)

    try:
        img = volume[:, :, slice_idx]
    except Exception as e:
        if image_id is not None:
            print(f"[slice_overlay] index error image_id={image_id} slice={slice_idx} err={e}", flush=True)
        return None, None

    if mask2d_u8 is None or mask2d_u8.shape != img.shape:
        if mask2d_u8.T.shape == img.shape:
            mask2d_u8 = mask2d_u8.T
        elif image_id is not None:
            print(
                f"[slice_overlay] mask shape mismatch image_id={image_id} slice={slice_idx} "
                f"mask_shape={None if mask2d_u8 is None else mask2d_u8.shape} img_shape={img.shape} "
                f"report={report_path}",
                flush=True
            )
        return None, None

    # raw
    raw_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    raw_path = os.path.join(out_dir, f"slice_{slice_idx:03d}.png")

    ok = cv2.imwrite(raw_path, raw_rgb)
    if not ok or (not os.path.exists(raw_path)) or os.path.getsize(raw_path) == 0:
        if image_id is not None:
            print(f"[slice_overlay] write raw failed image_id={image_id} slice={slice_idx} path={raw_path}", flush=True)
        return None, None

    # overlay (contour only)
    overlay_rgb = raw_rgb.copy()
    m = (mask2d_u8 > 0).astype(np.uint8)

    if int(m.sum()) > 0:
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 绿色轮廓（不填充，信息泄漏更小）
        cv2.drawContours(overlay_rgb, contours, -1, (0, 255, 0), thickness)

    overlay_path = os.path.join(out_dir, f"slice_{slice_idx:03d}_overlay.png")
    ok2 = cv2.imwrite(overlay_path, overlay_rgb)
    if not ok2 or (not os.path.exists(overlay_path)) or os.path.getsize(overlay_path) == 0:
        if image_id is not None:
            print(f"[slice_overlay] write overlay failed image_id={image_id} slice={slice_idx} path={overlay_path}", flush=True)
        return raw_path, None

    return raw_path, overlay_path

# =========================
# Slice-memory helpers
# =========================
def load_region_slices(memory_path: str, log_fn=print) -> dict:
    """
    Load region_slices from {image_id}_slice_memory.json
    Returns: dict like {"liver":[s,e], ...}. If missing/bad, returns {}.
    """
    if not memory_path or (not os.path.exists(memory_path)):
        return {}
    try:
        with open(memory_path, "r") as f:
            mem = json.load(f)
        rs = mem.get("region_slices", {})
        return rs if isinstance(rs, dict) else {}
    except Exception as e:
        log_fn(f"[memory] failed to load: {memory_path} err={e}")
        return {}


def format_slice_and_memory_note(selected_slice_indices: list[int], region_slices: dict) -> str:
    """
    Produce a short note appended into the prompt.
    """
    selected_slice_indices = [int(x) for x in selected_slice_indices] if selected_slice_indices else []
    lines = []
    lines.append(f"Selected slice index (z): {selected_slice_indices if selected_slice_indices else 'N/A'}")

    if isinstance(region_slices, dict) and len(region_slices) > 0:
        lines.append("Slice-memory organ ranges (z index, inclusive):")
        for k in sorted(region_slices.keys()):
            v = region_slices.get(k)
            if isinstance(v, (list, tuple)) and len(v) == 2:
                lines.append(f"- {k}: [{int(v[0])}, {int(v[1])}]")
            else:
                lines.append(f"- {k}: {v}")
    else:
        lines.append("Slice-memory organ ranges: N/A (missing or empty)")

    return "\n".join(lines)
