import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# ============================================================
# Backend selection
# ============================================================
SEG_BACKEND = "totalseg"   # "biomedparse" | "vista3d" | "totalseg"

# ============================================================
# ------------------ GPU device limit ------------------------
# ============================================================
# Set this before importing torch. The value is a physical GPU id; inside this
# process it becomes logical cuda:0 because only that GPU is visible. TotalSeg
# is an exception in this environment: setting CUDA_VISIBLE_DEVICES hides CUDA
# from torch, so TotalSegmentator receives the physical device id via -d gpu:X.
SEGMENTATION_CUDA_DEVICE_ID = os.environ.get("SEGMENTATION_CUDA_DEVICE_ID", "6")
if SEG_BACKEND != "totalseg":
    os.environ["CUDA_VISIBLE_DEVICES"] = SEGMENTATION_CUDA_DEVICE_ID

import sys
import json
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import time
import nibabel as nib
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# ============================================================
# ------------------ CPU thread limits ------------------------
# ============================================================
torch.set_num_threads(4)
torch.set_num_interop_threads(4)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.cuda.set_device(DEVICE)
if SEG_BACKEND == "totalseg":
    TOTALSEG_DEVICE = f"gpu:{SEGMENTATION_CUDA_DEVICE_ID}"
    print(f"Using TotalSegmentator device: {TOTALSEG_DEVICE} (CUDA_VISIBLE_DEVICES not set)")
else:
    TOTALSEG_DEVICE = None
    print(f"Using device: {DEVICE} (CUDA_VISIBLE_DEVICES={SEGMENTATION_CUDA_DEVICE_ID})")


# ============================================================
# ------------------ paths & imports -------------------------
# ============================================================
PROJECT_ROOT = "/mnt/blobdata/project/3DMedAgent"
DATA_ROOT = "/mnt/blobdata/data/DeepTumorVQA"

BIOPARSE_ROOT = os.path.join(PROJECT_ROOT, "SegAgent/BiomedParse")
sys.path.insert(0, BIOPARSE_ROOT)

VISTA3D_ROOT = os.path.join(PROJECT_ROOT, "SegAgent/VISTA3d")
sys.path.insert(0, VISTA3D_ROOT)

from SegAgent import BiomedParse_Segmentator, Vista3D_Segmentator, run_totalsegmentator_cli

# ============================================================
# ------------------ load label_dict.json --------------------
# ============================================================
BIO_LABEL_DICT_PATH = os.path.join(BIOPARSE_ROOT, "label_dict.json")
with open(BIO_LABEL_DICT_PATH, "r") as f:
    _bio_map = json.load(f)
    BIO_ID2ORGAN = {int(k): v for k, v in _bio_map.items()}
    BIO_ORGAN2ID = {v: int(k) for k, v in _bio_map.items()}

VISTA3D_LABEL_DICT_PATH = os.path.join(VISTA3D_ROOT, "label_dict.json")
with open(VISTA3D_LABEL_DICT_PATH, "r") as f:
    VISTA3D_ORGAN2ID = json.load(f)
    VISTA3D_ID2ORGAN = {v: k for k, v in VISTA3D_ORGAN2ID.items()}

# ============================================================
# ------------------ VQA data --------------------------------
# ============================================================
vqa_file = os.path.join(PROJECT_ROOT, "VQA/DeepTumorVQA_sampled-v3.csv")
vqa_data = pd.read_csv(vqa_file)

image_ids = vqa_data["Image ID"].dropna().unique()
print(f"Total unique Image IDs: {len(image_ids)}")

imageid_to_dataset = (
    vqa_data
    .dropna(subset=["Image ID", "dataset"])
    .drop_duplicates("Image ID")
    .set_index("Image ID")["dataset"]
    .to_dict()
)

# ============================================================
# ------------------ dirs ------------------------------------
# ============================================================
img_dir = os.path.join(DATA_ROOT, "data")
save_dir = os.path.join(DATA_ROOT, "Subset-v3/segmentations/VISTA3D")

# img_dir = os.path.join(DATA_ROOT, "data")
# save_dir = os.path.join(DATA_ROOT, "Healthy_set/segmentations/VISTA3D")
# ============================================================
# ------------------ segmentator init ------------------------
# ============================================================
if SEG_BACKEND == "biomedparse":
    segmentator = BiomedParse_Segmentator(
        config_dir=os.path.join(BIOPARSE_ROOT, "configs/model"),
        checkpoint_path=os.path.join(PROJECT_ROOT, "BiomedParse/checkpoints/biomedparse_v2.ckpt"),
        device=DEVICE,
    )
elif SEG_BACKEND == "vista3d":
    segmentator = Vista3D_Segmentator(
        config_file=os.path.join(VISTA3D_ROOT, "configs/infer.yaml"),
        device=str(DEVICE),
    )
elif SEG_BACKEND == "totalseg":
    segmentator = None
else:
    raise ValueError(f"Unknown SEG_BACKEND: {SEG_BACKEND}")

# ============================================================
# ------------------ organ list ------------------------------
# ============================================================
object_list = [
    "liver",
    "spleen",
    "pancreas",
    "colon",
    "left kidney",
    "right kidney",
    "pancreatic tumor",
    "hepatic tumor",
    "left kidney cyst",
    "right kidney cyst",
]

# ============================================================
# ------------------ organ → label_id mapping ----------------
# ============================================================
if SEG_BACKEND == "biomedparse":
    organ2label = BIO_ORGAN2ID
elif SEG_BACKEND == "vista3d":
    organ2label = VISTA3D_ORGAN2ID
else:
    organ2label = None

# ============================================================
# ------------------ helper: save binary mask ----------------
# ============================================================
def save_binary_mask(mask, affine, header, save_path):
    nii = nib.Nifti1Image(mask.astype(np.uint8, copy=False), affine, header)
    nii.set_data_dtype(np.uint8)
    nib.save(nii, save_path)

# ============================================================
# ------------------ helper: backend prompt ------------------
# ============================================================
def map_organs_to_backend_prompt(organs, backend):
    if backend == "biomedparse":
        return organs
    elif backend == "vista3d":
        return [VISTA3D_ORGAN2ID[o] for o in organs]
    elif backend == "totalseg":
        return organs
    else:
        raise RuntimeError("Unreachable backend")

# ============================================================
# ------------------ robust nib.load for blob ----------------
# ============================================================
def robust_load_affine_header(
    input_path: str,
    retries: int = 6,
    sleep: float = 0.35,
    tag: str = "",
):
    """
    Blob/NFS 上常见的 transient 问题：
    - Empty file
    - Temporary read failure
    这里做 best-effort 重试，失败则抛出异常交给上层 fallback。
    """
    last_err = None
    for t in range(retries):
        try:
            ref = nib.load(input_path)
            return ref.affine, ref.header
        except Exception as e:
            last_err = e
            time.sleep(sleep * (t + 1))
    raise last_err

# ============================================================
# ------------------ CPU prefetch loader ---------------------
# ============================================================
def preload_ref_nii(input_path: str):
    # best-effort retry inside thread
    return robust_load_affine_header(input_path, retries=6, sleep=0.35, tag="prefetch")

# ============================================================
# ------------------ main loop --------------------------------
# ============================================================
START_IDX = 0
MAX_CASES = None
SKIP_EXISTING = True
LOG_EVERY = 100

print(
    "Run controls: "
    f"START_IDX={START_IDX}, "
    f"MAX_CASES={MAX_CASES if MAX_CASES is not None else 'all'}, "
    f"SKIP_EXISTING={SKIP_EXISTING}, "
    f"LOG_EVERY={LOG_EVERY}"
)

executor = ThreadPoolExecutor(max_workers=2)
future = None
next_payload = None  # (input_path, out_case_dir, backend_prompt, img_id, dataset)
stats = {"attempted": 0, "succeeded": 0, "failed": 0, "skipped": 0}

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def expected_liver_segment_paths(out_case_dir: str):
    return [
        os.path.join(out_case_dir, f"liver_segment_{segment_idx}.nii.gz")
        for segment_idx in range(1, 9)
    ]


def log_progress(prefix: str, index: int | None = None):
    index_part = f" index={index}" if index is not None else ""
    print(
        f"[{prefix}] time={now_str()}{index_part} "
        f"attempted={stats['attempted']} "
        f"succeeded={stats['succeeded']} "
        f"failed={stats['failed']} "
        f"skipped={stats['skipped']}",
        flush=True,
    )


def should_log_case():
    return MAX_CASES is not None

def build_case_payload(img_id: str):
    dataset = imageid_to_dataset[img_id]
    input_path = os.path.join(img_dir, dataset, "img", f"{img_id}.nii.gz")
    out_case_dir = os.path.join(save_dir, dataset, img_id)
    os.makedirs(out_case_dir, exist_ok=True)
    backend_prompt = map_organs_to_backend_prompt(object_list, SEG_BACKEND)
    return input_path, out_case_dir, backend_prompt, dataset


def case_is_complete(out_case_dir: str):
    if SEG_BACKEND == "totalseg":
        return all(
            os.path.exists(path) and os.path.getsize(path) > 0
            for path in expected_liver_segment_paths(out_case_dir)
        )

    if not os.path.exists(os.path.join(out_case_dir, "all.nii.gz")):
        return False

    expected_mask_paths = [
        os.path.join(out_case_dir, f"{organ}.nii.gz")
        for organ in object_list
    ]
    expected_mask_paths.extend([
        os.path.join(out_case_dir, "kidney.nii.gz"),
        os.path.join(out_case_dir, "kidney cyst.nii.gz"),
        os.path.join(out_case_dir, "kidney_cyst.nii.gz"),
    ])
    return any(os.path.exists(path) for path in expected_mask_paths)


def print_liver_segment_stats(input_path: str, out_case_dir: str):
    ref_shape = nib.load(input_path).shape
    liver_path = os.path.join(out_case_dir, "liver.nii.gz")
    liver_mask = None
    if os.path.exists(liver_path):
        liver_mask = nib.load(liver_path).get_fdata() > 0

    print(f"[SMOKE] CT shape={ref_shape}", flush=True)
    nonempty_segments = 0
    segment_union = np.zeros(ref_shape, dtype=bool)
    for segment_idx, path in enumerate(expected_liver_segment_paths(out_case_dir), start=1):
        if not os.path.exists(path):
            print(f"[SMOKE][MISSING] liver_segment_{segment_idx}: {path}", flush=True)
            continue

        nii = nib.load(path)
        data = nii.get_fdata()
        unique, counts = np.unique(data, return_counts=True)
        mask = data > 0
        voxel_count = int(mask.sum())
        nonempty_segments += int(voxel_count > 0)
        if mask.shape == ref_shape:
            segment_union |= mask
        ratio = voxel_count / mask.size if mask.size else 0.0
        unique_counts = list(zip(unique.tolist(), counts.tolist()))
        print(
            f"[SMOKE] liver_segment_{segment_idx}: "
            f"shape={mask.shape} dtype={nii.get_data_dtype()} "
            f"unique_counts={unique_counts} nonzero={voxel_count} ratio={ratio:.6f}",
            flush=True,
        )

    print(f"[SMOKE] nonempty_liver_segments={nonempty_segments}/8", flush=True)
    if liver_mask is not None and liver_mask.shape == segment_union.shape:
        overlap = int((segment_union & liver_mask).sum())
        union_voxels = int(segment_union.sum())
        liver_voxels = int(liver_mask.sum())
        print(
            f"[SMOKE] segment_union_voxels={union_voxels} "
            f"liver_voxels={liver_voxels} "
            f"overlap_with_liver={overlap}",
            flush=True,
        )

# ---- initialize prefetch for first item (START_IDX) ----
if START_IDX < len(image_ids) and SEG_BACKEND in ["biomedparse", "vista3d"]:
    first_img_id = image_ids[START_IDX]
    input_path, out_case_dir, backend_prompt, dataset = build_case_payload(first_img_id)
    future = executor.submit(preload_ref_nii, input_path)
    next_payload = (input_path, out_case_dir, backend_prompt, first_img_id, dataset)

image_iter = tqdm(
    image_ids,
    desc=f"Segmenting images ({SEG_BACKEND})",
    disable=MAX_CASES is None,
)
for i, img_id in enumerate(image_iter):
    if i < START_IDX:
        continue

    if MAX_CASES is not None and stats["attempted"] >= MAX_CASES:
        print(f"Reached SEGMENTATION_MAX_CASES={MAX_CASES}; stopping.")
        break

    dataset = imageid_to_dataset[img_id]
    input_path = os.path.join(img_dir, dataset, "img", f"{img_id}.nii.gz")
    
    out_case_dir = os.path.join(save_dir, dataset, img_id)
    os.makedirs(out_case_dir, exist_ok=True)
    backend_prompt = map_organs_to_backend_prompt(object_list, SEG_BACKEND)

    if SKIP_EXISTING and case_is_complete(out_case_dir):
        stats["skipped"] += 1
        future = None
        next_payload = None
        if should_log_case() or stats["skipped"] % LOG_EVERY == 0:
            print(f"[SKIP] Existing complete output for {img_id} ({dataset})", flush=True)
            log_progress("PROGRESS", i)
        continue

    stats["attempted"] += 1
    if should_log_case() or stats["attempted"] % LOG_EVERY == 1:
        print(
            f"[CASE] time={now_str()} index={i} attempted={stats['attempted']} "
            f"image_id={img_id} dataset={dataset}",
            flush=True,
        )

    # --------------------------------------------------------
    # Prefetch: get affine/header (best-effort)
    # IMPORTANT: prefetch failures must NOT crash the loop
    # --------------------------------------------------------
    if SEG_BACKEND in ["biomedparse", "vista3d"]:
        affine = header = None

        # 1) try prefetch result if it matches current
        if future is not None and next_payload is not None and next_payload[3] == img_id:
            try:
                affine, header = future.result()
            except Exception as e:
                print(f"[WARN] Prefetch failed for {img_id} ({dataset}): {repr(e)}")

        # 2) fallback: sync robust load
        if affine is None or header is None:
            try:
                affine, header = robust_load_affine_header(input_path, retries=6, sleep=0.35, tag="sync")
            except Exception as e:
                print(f"[ERROR] Cannot load affine/header for {img_id} ({dataset}): {repr(e)}")
                stats["failed"] += 1
                # schedule next prefetch before continue (so pipeline doesn't stall)
                next_i = i + 1
                if next_i < len(image_ids):
                    try:
                        next_img_id = image_ids[next_i]
                        next_input_path, next_out_case_dir, next_backend_prompt, next_dataset = build_case_payload(next_img_id)
                        future = executor.submit(preload_ref_nii, next_input_path)
                        next_payload = (next_input_path, next_out_case_dir, next_backend_prompt, next_img_id, next_dataset)
                    except Exception as e2:
                        print(f"[WARN] Failed to schedule next prefetch after load error: {repr(e2)}")
                        future = None
                        next_payload = None
                else:
                    future = None
                    next_payload = None
                continue

        # 3) schedule prefetch for next
        next_i = i + 1
        if next_i < len(image_ids):
            next_img_id = image_ids[next_i]
            try:
                next_input_path, next_out_case_dir, next_backend_prompt, next_dataset = build_case_payload(next_img_id)
                future = executor.submit(preload_ref_nii, next_input_path)
                next_payload = (next_input_path, next_out_case_dir, next_backend_prompt, next_img_id, next_dataset)
            except Exception as e:
                print(f"[WARN] Failed to schedule prefetch for next ({next_img_id}): {repr(e)}")
                future = None
                next_payload = None
        else:
            future = None
            next_payload = None

    else:
        # totalseg: no prefetch path
        try:
            affine, header = robust_load_affine_header(input_path, retries=6, sleep=0.35, tag="totalseg_sync")
        except Exception as e:
            print(f"[ERROR] Cannot load affine/header for {img_id} ({dataset}): {repr(e)}")
            stats["failed"] += 1
            continue

    # --------------------------------------------------------
    # Run segmentation with guard
    # --------------------------------------------------------
    try:
        if SEG_BACKEND == "biomedparse":
            all_mask = segmentator.segment(
                input_path=input_path,
                output_path=os.path.join(out_case_dir, "all.nii.gz"),
                object_list=backend_prompt,
                norm_range=(0, 255),
            )
        elif SEG_BACKEND == "vista3d":
            all_mask = segmentator.segment(
                input_path=input_path,
                output_path=os.path.join(out_case_dir, "all.nii.gz"),
                object_list=backend_prompt,
                save_mask=True,
            )
        elif SEG_BACKEND == "totalseg":
            run_totalsegmentator_cli(
                input_path=input_path,
                out_case_dir=out_case_dir,
                organ_list=object_list,
                device=TOTALSEG_DEVICE,
            )
    except Exception as e:
        print(f"[ERROR] Segmentation failed for {img_id} ({dataset}): {repr(e)}")
        stats["failed"] += 1
        continue

    # --------------------------------------------------------
    # Save binary masks (guarded)
    # --------------------------------------------------------
    if SEG_BACKEND in ["biomedparse", "vista3d"]:
        try:
            needed_label_ids = {organ2label[o] for o in object_list}
            present = np.unique(all_mask)
            label_masks = {lid: (all_mask == lid) for lid in present if lid in needed_label_ids}

            for organ in object_list:
                lid = organ2label[organ]
                m = label_masks.get(lid, None)
                if m is None:
                    continue
                save_binary_mask(m, affine, header, os.path.join(out_case_dir, f"{organ}.nii.gz"))

            # merged kidney
            if "left kidney" in organ2label and "right kidney" in organ2label:
                lk = organ2label["left kidney"]
                rk = organ2label["right kidney"]
                if lk in label_masks and rk in label_masks:
                    save_binary_mask(label_masks[lk] | label_masks[rk], affine, header,
                                     os.path.join(out_case_dir, "kidney.nii.gz"))

            # merged kidney cyst (vista3d)
            if SEG_BACKEND == "vista3d":
                lkc = organ2label["left kidney cyst"]
                rkc = organ2label["right kidney cyst"]
                if lkc in label_masks and rkc in label_masks:
                    save_binary_mask(label_masks[lkc] | label_masks[rkc], affine, header,
                                     os.path.join(out_case_dir, "kidney cyst.nii.gz"))

        except Exception as e:
            print(f"[ERROR] Mask saving failed for {img_id} ({dataset}): {repr(e)}")
            stats["failed"] += 1
            continue

    if SEG_BACKEND == "totalseg":
        try:
            if not case_is_complete(out_case_dir):
                missing = [
                    os.path.basename(path)
                    for path in expected_liver_segment_paths(out_case_dir)
                    if not os.path.exists(path) or os.path.getsize(path) == 0
                ]
                raise RuntimeError(f"Missing liver segment outputs: {missing}")

            if should_log_case():
                print_liver_segment_stats(input_path, out_case_dir)

            left_path = os.path.join(out_case_dir, "kidney_left.nii.gz")
            right_path = os.path.join(out_case_dir, "kidney_right.nii.gz")

            if os.path.exists(left_path) and os.path.exists(right_path):
                left_nii = nib.load(left_path)
                right_nii = nib.load(right_path)
                kidney_mask = np.logical_or(left_nii.get_fdata() > 0, right_nii.get_fdata() > 0)
                save_binary_mask(kidney_mask, affine, header, os.path.join(out_case_dir, "kidney.nii.gz"))

            left_cyst_path = os.path.join(out_case_dir, "kidney_cyst_left.nii.gz")
            right_cyst_path = os.path.join(out_case_dir, "kidney_cyst_right.nii.gz")
            if os.path.exists(left_cyst_path) and os.path.exists(right_cyst_path):
                left_nii = nib.load(left_cyst_path)
                right_nii = nib.load(right_cyst_path)
                kidney_cyst_mask = np.logical_or(left_nii.get_fdata() > 0, right_nii.get_fdata() > 0)
                save_binary_mask(kidney_cyst_mask, affine, header, os.path.join(out_case_dir, "kidney_cyst.nii.gz"))
        except Exception as e:
            print(f"[ERROR] TotalSeg postprocess failed for {img_id} ({dataset}): {repr(e)}")
            stats["failed"] += 1
            continue

    stats["succeeded"] += 1
    if should_log_case():
        print(f"[OK] Completed {img_id} ({dataset})", flush=True)
    elif stats["succeeded"] % LOG_EVERY == 0:
        log_progress("PROGRESS", i)

executor.shutdown(wait=True)
print(
    "Segmentation summary: "
    f"attempted={stats['attempted']}, "
    f"succeeded={stats['succeeded']}, "
    f"failed={stats['failed']}, "
    f"skipped={stats['skipped']}",
    flush=True,
)
