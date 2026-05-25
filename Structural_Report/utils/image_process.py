# utils/image_process.py

import numpy as np
import nibabel as nib
import scipy.ndimage as ndimage
from scipy.spatial.distance import pdist, squareform
from skimage import measure
from skimage.transform import rotate

def count_unconnected_objects(binary_array, erode=0):
    """
    Count connected components in binary image.
    """
    if erode > 0:
        binary_array = ndimage.binary_erosion(
            binary_array, structure=np.ones((erode, erode))
        )
    labeled, num = ndimage.label(binary_array)
    return num


def is_binary_array(array):
    """
    Check whether array is binary.
    """
    return np.array_equal(array, array.astype(bool))


def plot_slices(images, titles, save=None):
    """
    Plot slices (from original code).
    """
    import matplotlib.pyplot as plt
    n = len(images)
    plt.figure(figsize=(4 * n, 4))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.title(titles[i])
        plt.axis("off")
    if save:
        plt.savefig(save, bbox_inches="tight")
    plt.close()


def plot_slices_union(ct_slice, mask_slice):
    """
    Overlay CT and mask.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4, 4))
    plt.imshow(ct_slice, cmap="gray")
    plt.imshow(mask_slice, cmap="jet", alpha=0.5)
    plt.axis("off")
    plt.close()


def plot_slices_overlay(ct_slice, mask_slice, save=None):
    """
    Overlay CT and mask, optional save.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4, 4))
    plt.imshow(ct_slice, cmap="gray")
    plt.imshow(mask_slice, cmap="jet", alpha=0.4)
    plt.axis("off")
    if save:
        plt.savefig(save, bbox_inches="tight")
    plt.close()


def plot_slice_overlay(slice_image, mask, save=None):
    """
    Draw overlay for single slice.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4, 4))
    plt.imshow(slice_image, cmap="gray")
    plt.imshow(mask, cmap="jet", alpha=0.4)
    plt.axis("off")
    if save:
        plt.savefig(save, bbox_inches="tight")
    plt.close()


def print_slice(slice_image, cmap="gray"):
    """
    Display slice inline.
    """
    import matplotlib.pyplot as plt
    plt.imshow(slice_image, cmap=cmap)
    plt.axis("off")
    plt.show()


def measure_diameter(binary_image):
    """
    Compute longest contour diameter in 2D mask.
    """
    contours = measure.find_contours(binary_image, 0.5)
    if not contours:
        return 0, None, None

    contour = contours[0]
    dist = squareform(pdist(contour))
    idx = np.unravel_index(np.argmax(dist), dist.shape)
    p1 = contour[idx[0]]
    p2 = contour[idx[1]]
    return dist[idx], p1, p2


def rotate_image(binary_image, angle):
    """
    Rotate binary mask by angle while preserving connectivity.
    """
    rotated = rotate(binary_image, angle, resize=True,
                     order=0, preserve_range=True, mode="constant", cval=0)
    return (rotated > 0.5).astype(np.uint8)


def measure_vertical_span(binary_image):
    """
    Vertical span of mask along y-axis.
    """
    ys, _ = np.where(binary_image == 1)
    return ys.max() - ys.min() if len(ys) > 0 else 0


def measure_volume(array_3d, spacing, check_border=False):
    """
    Volume measurement with optional border check.
    """
    if check_border:
        eroded = ndimage.binary_erosion(array_3d, structure=np.ones((1, 1, 1)))
        dilated = ndimage.binary_dilation(eroded, structure=np.ones((1, 1, 1)))

        if (
            np.any(dilated[0, :, :]) or np.any(dilated[-1, :, :]) or
            np.any(dilated[:, 0, :]) or np.any(dilated[:, -1, :]) or
            np.any(dilated[:, :, 0]) or np.any(dilated[:, :, -1])
        ):
            return None

    return np.sum(array_3d > 0.5)


def measure_organ_hu(
    organ,
    tumor,
    ct,
    trim_percent=0.05,              # 去掉两端离群
    valid_range=(-2000, 5000),       # 过滤明显无效值
    erode_iters=0,                   # mask 腐蚀
    normalize=False,                 # <<< 新增：是否做归一化
    norm_range=(-1024,3071),    # <<< 新增：目标归一化范围
):
    # ---------------------------
    # 1) organ mask
    # ---------------------------
    organ_mask = organ > 0
    if erode_iters and erode_iters > 0:
        organ_mask = ndimage.binary_erosion(
            organ_mask, iterations=int(erode_iters)
        )

    if tumor is not None:
        organ_mask &= (tumor == 0)

    values = ct[organ_mask]
    if values.size == 0:
        return None, None

    # ---------------------------
    # 2) 过滤明显无效值（padding / 极端异常）
    # ---------------------------
    if valid_range is not None:
        lo_v, hi_v = valid_range
        values = values[(values >= lo_v) & (values <= hi_v)]
        if values.size == 0:
            return None, None

    # ---------------------------
    # 3) trimmed statistics
    # ---------------------------
    if trim_percent and trim_percent > 0:
        p = float(trim_percent)
        p = max(0.0, min(0.49, p))
        lo = np.quantile(values, p)
        hi = np.quantile(values, 1.0 - p)
        values = values[(values >= lo) & (values <= hi)]
        if values.size == 0:
            return None, None

    # ---------------------------
    # 4) 可选：归一化到常用 CT 范围
    # ---------------------------
    if normalize:
        vmin = values.min()
        vmax = values.max()
        if vmax <= vmin:
            return None, None

        tgt_min, tgt_max = norm_range
        values = (values - vmin) / (vmax - vmin)
        values = values * (tgt_max - tgt_min) + tgt_min

    return float(values.mean()), float(values.std(ddof=0))


def get_tumor_segment(segments, tumor_mask):
    """
    Identify tumor segment based on liver/pancreas/kidney segments.
    """
    if segments is None:
        return None

    overlap = segments * (tumor_mask > 0)
    uniq = np.unique(overlap)
    uniq = uniq[uniq > 0]
    return uniq.tolist() if len(uniq) > 0 else None


def analyze_nth_largest_connected_component(
    array_3d, ns=[1], th=None, erode=0, ct=None, segments=None, resize_factor=1
):
    """
    原始逻辑完整保留，用于分析 3D lesion components。
    """
    struct = np.ones((3, 3, 3), dtype=int)
    labeled, num = ndimage.label(array_3d, structure=struct)

    sizes = ndimage.sum(array_3d, labeled, range(1, num + 1))
    sizes = np.array(sizes)

    sorted_idx = np.argsort(sizes)[::-1]
    results = []

    for n in ns:
        if n > len(sorted_idx):
            continue
        idx = sorted_idx[n - 1]
        mask_n = (labeled == (idx + 1))

        result = {
            "mask": mask_n,
            "size": sizes[idx],
        }

        results.append(result)

    return results


def compute_ct_metaHU(ct_path: str) -> str:
    """
    Compute scan-level HU intensity range and percentiles.

    Parameters
    ----------
    ct_path : str
        Path to CT NIfTI file (.nii or .nii.gz)

    Returns
    -------
    str
        Formatted multi-line string describing HU distribution
    """

    # -------------------------
    # Load CT
    # -------------------------
    img = nib.load(ct_path)
    data = img.get_fdata(dtype=np.float32)

    # Flatten & clean
    hu = data.reshape(-1)
    hu = hu[np.isfinite(hu)]  # remove NaN / Inf

    if hu.size == 0:
        raise ValueError("Empty or invalid CT volume")

    # -------------------------
    # Basic range
    # -------------------------
    hu_min = float(np.min(hu))
    hu_max = float(np.max(hu))

    # -------------------------
    # Robust percentiles
    # -------------------------
    percentiles = {
        "p1": 1,
        "p5": 5,
        "p50": 50,
        "p95": 95,
        "p99": 99,
    }

    pct_values = {
        name: float(np.percentile(hu, q))
        for name, q in percentiles.items()
    }

    # -------------------------
    # Formatting (VLM-friendly)
    # -------------------------
    lines = []
    lines.append("[CT_HU_META]")
    lines.append(
        f"HU_range: min={hu_min:.1f}, max={hu_max:.1f}"
    )
    lines.append(
        "HU_percentiles: "
        + ", ".join(
            f"{k}={v:.1f}" for k, v in pct_values.items()
        )
    )

    return "\n".join(lines)