#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Precompute end-to-end slice-level CT-CLIP organ lesion scores.

This script does not read precomputed embeddings. For each unique CT case it
loads the NIfTI volume, runs one CT-CLIP visual forward pass, and writes
slice-level organ/lesion probabilities that match the option-specific weighting
style used by notebook Method 5.
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from clip_encode import CTCLIPEmbeddingExtractor
from clip_utils import load_organ_mask


DEFAULT_MODEL_PATH = "/mnt/blobdata/project/CT-CLIP/models/CT-CLIP_v2.pt"
DEFAULT_CSV_PATH = "/mnt/blobdata/project/3DMedAgent/VQA/DeepTumorVQA_sampled-v3.csv"
DEFAULT_DATA_ROOT = "/mnt/blobdata/data/DeepTumorVQA/data"
DEFAULT_MASK_DIR = "/mnt/blobdata/data/DeepTumorVQA/Subset-v3/segmentations/VISTA3D"
DEFAULT_OUTPUT_DIR = "/mnt/blobdata/data/DeepTumorVQA/Subset-v3/clip_detail_slice"

CANONICAL_ORGANS = ("liver", "kidney", "colon", "pancreas", "spleen")
LESION_TYPES = ("lesion", "tumor", "cyst")
TOTAL_SLICES = 240
TEMPORAL_PATCHES = 24
TEMPORAL_PATCH_SIZE = 10
SPATIAL_PATCHES = 24
SPATIAL_PATCH_SIZE = 20


def zero_slice_result() -> Dict[str, object]:
    return {
        "global_probability": 0.0,
        "top_slices": [],
        "slice_probabilities": [0.0] * TOTAL_SLICES,
        "slice_organ_areas": [0] * TOTAL_SLICES,
        "slice_organ_patch_counts": [0] * TOTAL_SLICES,
    }


def slice_percent(slice_index: int) -> float:
    return round(100.0 * int(slice_index) / (TOTAL_SLICES - 1), 4)


def slice_patch_area_grid(organ_mask: torch.Tensor) -> torch.Tensor:
    """
    Convert full-resolution organ mask [240,480,480] to per-slice patch areas
    [240,24,24], matching CT-CLIP spatial patches.
    """
    if tuple(organ_mask.shape) != (TOTAL_SLICES, 480, 480):
        raise ValueError(f"Unexpected organ mask shape: {tuple(organ_mask.shape)}")

    mask = organ_mask.float().contiguous()
    return mask.view(
        TOTAL_SLICES,
        SPATIAL_PATCHES,
        SPATIAL_PATCH_SIZE,
        SPATIAL_PATCHES,
        SPATIAL_PATCH_SIZE,
    ).sum(dim=(2, 4))


class SliceLevelScorer:
    """CT-CLIP scorer for slice-level Method-5-style probabilities."""

    def __init__(self, model_path: str, device: str):
        self.extractor = CTCLIPEmbeddingExtractor(
            model_path=model_path,
            device=device,
            use_fp16=False,
        )
        self.device = self.extractor.device
        self.clip_model = self.extractor.clip_model
        self.tokenizer = self.extractor.tokenizer
        self._text_cache: Dict[Tuple[str, str], torch.Tensor] = {}

    def extract_case_embedding(self, nii_path: str) -> torch.Tensor:
        volume_tensor = self.extractor.load_and_preprocess_volume(nii_path)
        return self.extractor.extract_embedding(volume_tensor)

    def text_latents(self, organ: str, lesion_type: str) -> torch.Tensor:
        key = (organ, lesion_type)
        if key in self._text_cache:
            return self._text_cache[key]

        positive_text = f"{lesion_type.capitalize()} in {organ} is present."
        negative_text = f"{lesion_type.capitalize()} in {organ} is absent."
        text_tensor = self.tokenizer(
            [negative_text, positive_text],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            text_outputs = self.clip_model.text_transformer(**text_tensor)
            text_embeds = text_outputs.last_hidden_state[:, 0, :]
            latents = self.clip_model.to_text_latent(text_embeds)
            latents = F.normalize(latents, dim=-1)

        self._text_cache[key] = latents
        return latents

    def normalized_patches(self, enc_image: torch.Tensor) -> torch.Tensor:
        if tuple(enc_image.shape) != (1, 24, 24, 24, 512):
            raise ValueError(f"Unexpected embedding shape: {tuple(enc_image.shape)}")
        patches = enc_image.to(self.device).float().squeeze(0).reshape(-1, 512)
        return F.normalize(patches, dim=-1)

    def score_patches(
        self,
        normalized_patches: torch.Tensor,
        organ: str,
        lesion_type: str,
    ) -> torch.Tensor:
        text_latents = self.text_latents(organ, lesion_type)
        similarity = normalized_patches @ text_latents.T
        probability = F.softmax(similarity, dim=1)[:, 1]
        return probability.reshape(TEMPORAL_PATCHES, SPATIAL_PATCHES, SPATIAL_PATCHES)


def score_slices(
    scores_3d: torch.Tensor,
    slice_areas: torch.Tensor,
    threshold: float,
    top_k: int,
) -> Dict[str, object]:
    """Aggregate patch scores into 240 option-specific slice probabilities."""
    device = scores_3d.device
    temporal_indices = torch.arange(TOTAL_SLICES, device=device) // TEMPORAL_PATCH_SIZE
    slice_scores = scores_3d[temporal_indices]  # [240,24,24]
    areas = slice_areas.to(device).float()

    valid_mask = areas > 0
    high_conf_mask = valid_mask & (slice_scores > threshold)

    all_area = areas.sum(dim=(1, 2))
    all_weighted = (slice_scores * areas * valid_mask.float()).sum(dim=(1, 2))
    high_area = (areas * high_conf_mask.float()).sum(dim=(1, 2))
    high_weighted = (slice_scores * areas * high_conf_mask.float()).sum(dim=(1, 2))

    probabilities = torch.zeros(TOTAL_SLICES, dtype=torch.float32, device=device)
    has_high = high_area > 0
    has_any = all_area > 0
    probabilities[has_high] = high_weighted[has_high] / high_area[has_high]
    fallback = has_any & ~has_high
    probabilities[fallback] = all_weighted[fallback] / all_area[fallback]

    organ_patch_counts = valid_mask.sum(dim=(1, 2)).to(torch.int64)
    organ_areas = all_area.to(torch.int64)

    probabilities_cpu = [round(float(v), 6) for v in probabilities.detach().cpu().tolist()]
    organ_areas_cpu = [int(v) for v in organ_areas.detach().cpu().tolist()]
    patch_counts_cpu = [int(v) for v in organ_patch_counts.detach().cpu().tolist()]

    candidate_indices = [idx for idx, value in enumerate(probabilities_cpu) if value > 0]
    candidate_indices = sorted(
        candidate_indices,
        key=lambda idx: probabilities_cpu[idx],
        reverse=True,
    )[:top_k]
    top_slices = [
        {
            "slice_index": int(idx),
            "z_percent": slice_percent(idx),
            "temporal_patch_idx": int(idx // TEMPORAL_PATCH_SIZE),
            "probability": probabilities_cpu[idx],
            "organ_area": organ_areas_cpu[idx],
            "organ_patch_count": patch_counts_cpu[idx],
        }
        for idx in candidate_indices
    ]

    return {
        "global_probability": max(probabilities_cpu) if probabilities_cpu else 0.0,
        "top_slices": top_slices,
        "slice_probabilities": probabilities_cpu,
        "slice_organ_areas": organ_areas_cpu,
        "slice_organ_patch_counts": patch_counts_cpu,
    }


def process_case(
    dataset: str,
    image_id: str,
    scorer: SliceLevelScorer,
    data_root: str,
    mask_dir: str,
    threshold: float,
    top_k: int,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    nii_path = os.path.join(data_root, dataset, "img", f"{image_id}.nii.gz")
    if not os.path.exists(nii_path):
        raise FileNotFoundError(f"Missing NIfTI: {nii_path}")

    enc_image = scorer.extract_case_embedding(nii_path)
    normalized_patches = scorer.normalized_patches(enc_image)

    case_json: Dict[str, object] = {
        "image_id": image_id,
        "dataset": dataset,
        "organs": {},
    }

    missing_masks: List[str] = []
    for organ in CANONICAL_ORGANS:
        organ_result: Dict[str, object] = {"mask_available": True}
        mask = load_organ_mask(dataset, image_id, organ, mask_dir)
        if mask is None:
            missing_masks.append(organ)
            organ_result["mask_available"] = False
            for lesion_type in LESION_TYPES:
                organ_result[lesion_type] = zero_slice_result()
            case_json["organs"][organ] = organ_result
            continue

        areas = slice_patch_area_grid(mask)
        for lesion_type in LESION_TYPES:
            scores_3d = scorer.score_patches(normalized_patches, organ, lesion_type)
            organ_result[lesion_type] = score_slices(
                scores_3d=scores_3d,
                slice_areas=areas,
                threshold=threshold,
                top_k=top_k,
            )

        case_json["organs"][organ] = organ_result

    log_entry = {
        "image_id": image_id,
        "dataset": dataset,
        "status": "success",
        "missing_masks": missing_masks,
    }
    return case_json, log_entry


def iter_unique_cases(csv_path: str, max_samples: Optional[int]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"Image ID", "dataset"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    cases = df[["Image ID", "dataset"]].drop_duplicates().reset_index(drop=True)
    if max_samples is not None:
        cases = cases.head(max_samples)
    return cases


def save_case_json(case_json: Dict[str, object], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    image_id = str(case_json["image_id"])
    output_path = os.path.join(output_dir, f"{image_id}.json")
    tmp_path = f"{output_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(case_json, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, output_path)
    return output_path


def build_summary(
    total_cases: int,
    success_count: int,
    skipped_count: int,
    failed_count: int,
    threshold: float,
    top_k: int,
    status: str,
) -> Dict[str, object]:
    return {
        "updated_at": datetime.now().isoformat(),
        "status": status,
        "total_cases": int(total_cases),
        "processed": int(success_count + skipped_count + failed_count),
        "success": int(success_count),
        "skipped": int(skipped_count),
        "failed": int(failed_count),
        "organs": list(CANONICAL_ORGANS),
        "lesion_types": list(LESION_TYPES),
        "threshold": float(threshold),
        "top_k": int(top_k),
        "score_granularity": "slice_level",
        "preprocessing": "ctclip_standard_clip_normalize_pad_minus_one",
    }


def write_log(output_dir: str, summary: Dict[str, object], results: List[Dict[str, object]]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "preprocess_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute end-to-end slice-level CT-CLIP organ lesion scores."
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--csv_path", default=DEFAULT_CSV_PATH)
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--mask_dir", default=DEFAULT_MASK_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("End-to-end slice-level CT-CLIP detail preprocessing")
    print("=" * 80)
    print(f"CSV: {args.csv_path}")
    print(f"Data root: {args.data_root}")
    print(f"Masks: {args.mask_dir}")
    print(f"Output: {output_dir}")
    print(f"Device: {args.device}")
    print(f"Max samples: {args.max_samples if args.max_samples is not None else 'all'}")
    print(f"Threshold: {args.threshold}")
    print(f"Top K: {args.top_k}")
    print(f"Log every: {args.log_every} cases")
    print(f"Overwrite: {args.overwrite}")
    print("=" * 80)

    cases = iter_unique_cases(args.csv_path, args.max_samples)
    scorer = SliceLevelScorer(args.model_path, args.device)

    results: List[Dict[str, object]] = []
    success_count = 0
    skipped_count = 0
    failed_count = 0

    for _, row in tqdm(cases.iterrows(), total=len(cases), desc="Processing cases"):
        image_id = str(row["Image ID"])
        dataset = str(row["dataset"])
        output_path = os.path.join(output_dir, f"{image_id}.json")

        if os.path.exists(output_path) and not args.overwrite:
            skipped_count += 1
            result = {
                "image_id": image_id,
                "dataset": dataset,
                "status": "skipped",
                "reason": "output_exists",
            }
            results.append(result)
            if args.verbose:
                print(f"Skipped {image_id}: output exists")
        else:
            try:
                case_json, result = process_case(
                    dataset=dataset,
                    image_id=image_id,
                    scorer=scorer,
                    data_root=args.data_root,
                    mask_dir=args.mask_dir,
                    threshold=args.threshold,
                    top_k=args.top_k,
                )
                save_path = save_case_json(case_json, output_dir)
                result["output_path"] = save_path
                results.append(result)
                success_count += 1
                if args.verbose:
                    print(f"Saved {image_id}: {save_path}")
            except Exception as exc:
                failed_count += 1
                result = {
                    "image_id": image_id,
                    "dataset": dataset,
                    "status": "failed",
                    "error": str(exc),
                }
                results.append(result)
                print(f"Failed {image_id}: {exc}")

        processed_count = success_count + skipped_count + failed_count
        if args.log_every > 0 and processed_count % args.log_every == 0:
            summary = build_summary(
                total_cases=len(cases),
                success_count=success_count,
                skipped_count=skipped_count,
                failed_count=failed_count,
                threshold=args.threshold,
                top_k=args.top_k,
                status="running",
            )
            write_log(output_dir, summary, results)
            tqdm.write(
                f"Progress log: {processed_count}/{len(cases)} "
                f"(success={success_count}, skipped={skipped_count}, failed={failed_count})"
            )

    summary = build_summary(
        total_cases=len(cases),
        success_count=success_count,
        skipped_count=skipped_count,
        failed_count=failed_count,
        threshold=args.threshold,
        top_k=args.top_k,
        status="completed",
    )
    write_log(output_dir, summary, results)

    print("\nDone")
    print(f"Total: {len(cases)}")
    print(f"Success: {success_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Failed: {failed_count}")
    print(f"Log: {os.path.join(output_dir, 'preprocess_log.json')}")


if __name__ == "__main__":
    main()
