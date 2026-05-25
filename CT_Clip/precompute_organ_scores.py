#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Precompute Method-5-style organ-section lesion scores from CT-CLIP embeddings.

For each unique (dataset, Image ID) in the sampled DeepTumorVQA CSV, this
script loads a precomputed CT-CLIP volume embedding and VISTA3D organ masks,
then writes one JSON file per case.
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
from clip_utils import load_organ_mask, load_precomputed_embedding


DEFAULT_MODEL_PATH = "/mnt/blobdata/project/CT-CLIP/models/CT-CLIP_v2.pt"
DEFAULT_CSV_PATH = "/mnt/blobdata/project/3DMedAgent/VQA/DeepTumorVQA_sampled-v3.csv"
DEFAULT_EMBEDDING_DIR = "/mnt/blobdata/data/DeepTumorVQA/Subset-v3/clip_embedding"
DEFAULT_MASK_DIR = "/mnt/blobdata/data/DeepTumorVQA/Subset-v3/segmentations/VISTA3D"
DEFAULT_OUTPUT_DIR = "/mnt/blobdata/data/DeepTumorVQA/Subset-v3/clip_detail"

CANONICAL_ORGANS = ("liver", "kidney", "colon", "pancreas", "spleen")
LESION_TYPES = ("lesion", "tumor", "cyst")
TOTAL_SLICES = 240
TEMPORAL_PATCHES = 24
TEMPORAL_PATCH_SIZE = 10
SPATIAL_PATCHES = 24
SPATIAL_PATCH_SIZE = 20


def section_ranges(section_index: int) -> Dict[str, List[float]]:
    """Return inclusive slice indices and percent interval for one section."""
    slice_start = section_index * TEMPORAL_PATCH_SIZE
    slice_end = min((section_index + 1) * TEMPORAL_PATCH_SIZE - 1, TOTAL_SLICES - 1)
    percent_start = 100.0 * slice_start / TOTAL_SLICES
    percent_end = 100.0 * (slice_end + 1) / TOTAL_SLICES
    return {
        "slice_index_range": [int(slice_start), int(slice_end)],
        "z_percent_range": [round(percent_start, 4), round(percent_end, 4)],
    }


def build_empty_lesion_result() -> Dict[str, object]:
    return {
        "global_probability": 0.0,
        "max_section_index": None,
        "sections": [],
    }


def patch_area_grid(organ_mask: torch.Tensor) -> torch.Tensor:
    """
    Convert a full-resolution organ mask [240, 480, 480] into patch areas
    [24, 24, 24], matching CT-CLIP temporal and spatial patches.
    """
    if organ_mask.shape != (TOTAL_SLICES, 480, 480):
        raise ValueError(f"Unexpected organ mask shape: {tuple(organ_mask.shape)}")

    mask = organ_mask.float().contiguous()
    area = mask.view(
        TEMPORAL_PATCHES,
        TEMPORAL_PATCH_SIZE,
        SPATIAL_PATCHES,
        SPATIAL_PATCH_SIZE,
        SPATIAL_PATCHES,
        SPATIAL_PATCH_SIZE,
    ).sum(dim=(1, 3, 5))
    return area


class Method5TextScorer:
    """CT-CLIP text scorer for Method-5-style patch probabilities."""

    def __init__(self, model_path: str, device: str):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        extractor = CTCLIPEmbeddingExtractor(
            model_path=model_path,
            device=str(self.device),
            use_fp16=False,
        )
        self.clip_model = extractor.clip_model
        self.tokenizer = extractor.tokenizer
        self._text_cache: Dict[Tuple[str, str], torch.Tensor] = {}

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

    def normalize_patches(self, enc_image: torch.Tensor) -> torch.Tensor:
        if enc_image.shape != (1, 24, 24, 24, 512):
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
        return probability.reshape(24, 24, 24)


def score_sections(
    scores_3d: torch.Tensor,
    areas_3d: torch.Tensor,
    threshold: float,
) -> Dict[str, object]:
    """Aggregate patch scores into 24 Method-5-style section scores."""
    sections: List[Dict[str, object]] = []
    all_probabilities: List[float] = []

    for section_index in range(TEMPORAL_PATCHES):
        ranges = section_ranges(section_index)
        section_scores = scores_3d[section_index]
        section_areas = areas_3d[section_index]
        valid_mask = section_areas > 0
        organ_area = int(section_areas.sum().item())
        organ_patch_count = int(valid_mask.sum().item())

        if organ_area == 0 or organ_patch_count == 0:
            probability = 0.0
        else:
            high_conf_mask = valid_mask & (section_scores > threshold)
            if high_conf_mask.any():
                selected_scores = section_scores[high_conf_mask]
                selected_areas = section_areas[high_conf_mask]
            else:
                selected_scores = section_scores[valid_mask]
                selected_areas = section_areas[valid_mask]

            probability = float((selected_scores * selected_areas).sum().item() / selected_areas.sum().item())

        probability = round(probability, 6)
        all_probabilities.append(probability)
        if probability > 0:
            sections.append(
                {
                    "section_index": section_index,
                    "slice_index_range": ranges["slice_index_range"],
                    "z_percent_range": ranges["z_percent_range"],
                    "organ_area": organ_area,
                    "organ_patch_count": organ_patch_count,
                    "probability": probability,
                }
            )

    max_probability = max(all_probabilities) if all_probabilities else 0.0
    max_section_index = all_probabilities.index(max_probability) if max_probability > 0 else None

    return {
        "global_probability": round(max_probability, 6),
        "max_section_index": max_section_index,
        "sections": sections,
    }


def process_case(
    dataset: str,
    image_id: str,
    scorer: Method5TextScorer,
    embedding_dir: str,
    mask_dir: str,
    threshold: float,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    embedding = load_precomputed_embedding(image_id, embedding_dir)
    if embedding is None:
        raise FileNotFoundError(f"Missing or invalid embedding for {image_id}")

    normalized_patches = scorer.normalize_patches(embedding)
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
                organ_result[lesion_type] = build_empty_lesion_result()
            case_json["organs"][organ] = organ_result
            continue

        areas_3d = patch_area_grid(mask).to(scorer.device)
        for lesion_type in LESION_TYPES:
            scores_3d = scorer.score_patches(normalized_patches, organ, lesion_type)
            organ_result[lesion_type] = score_sections(scores_3d, areas_3d, threshold)

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
    status: str,
) -> Dict[str, object]:
    return {
        "updated_at": datetime.now().isoformat(),
        "status": status,
        "total_cases": int(total_cases),
        "processed": int(success_count + skipped_count + failed_count),
        "success": success_count,
        "skipped": skipped_count,
        "failed": failed_count,
        "organs": list(CANONICAL_ORGANS),
        "lesion_types": list(LESION_TYPES),
        "threshold": threshold,
    }


def write_log(output_dir: str, summary: Dict[str, object], results: List[Dict[str, object]]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "preprocess_log.json")
    payload = {"summary": summary, "results": results}
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute Method-5-style organ-section scores from CT-CLIP embeddings."
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--csv_path", default=DEFAULT_CSV_PATH)
    parser.add_argument("--embedding_dir", default=DEFAULT_EMBEDDING_DIR)
    parser.add_argument("--mask_dir", default=DEFAULT_MASK_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("Method-5 organ-section preprocessing")
    print("=" * 80)
    print(f"CSV: {args.csv_path}")
    print(f"Embeddings: {args.embedding_dir}")
    print(f"Masks: {args.mask_dir}")
    print(f"Output: {output_dir}")
    print(f"Device: {args.device}")
    print(f"Max samples: {args.max_samples if args.max_samples is not None else 'all'}")
    print(f"Threshold: {args.threshold}")
    print(f"Log every: {args.log_every} cases")
    print(f"Overwrite: {args.overwrite}")
    print("=" * 80)

    cases = iter_unique_cases(args.csv_path, args.max_samples)
    scorer = Method5TextScorer(args.model_path, args.device)

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
                    embedding_dir=args.embedding_dir,
                    mask_dir=args.mask_dir,
                    threshold=args.threshold,
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
