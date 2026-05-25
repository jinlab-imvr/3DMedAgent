#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Precompute unified global CT-CLIP disease probabilities.

For each unique DeepTumorVQA CT case, this script loads the NIfTI volume,
runs one CT-CLIP visual forward pass end-to-end, and writes one JSON file
under clip_global. Each organ gets global tumor, cyst, and lesion
probabilities, where lesion = max(tumor, cyst).
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


CANONICAL_ORGANS = ("liver", "kidney", "colon", "pancreas", "spleen")
BASE_LESION_TYPES = ("tumor", "cyst")

DEFAULT_MODEL_PATH = "/mnt/blobdata/project/CT-CLIP/models/CT-CLIP_v2.pt"
DEFAULT_CSV_PATH = "/mnt/blobdata/project/3DMedAgent/VQA/DeepTumorVQA_sampled-v3.csv"
DEFAULT_DATA_ROOT = "/mnt/blobdata/data/DeepTumorVQA/data"
DEFAULT_OUTPUT_DIR = "/mnt/blobdata/data/DeepTumorVQA/Subset-v3/clip_global"

EXPECTED_EMBEDDING_SHAPE = (1, 24, 24, 24, 512)
TEMPORAL_PATCHES = 24


class GlobalDiseaseScorer:
    """Scores tumor/cyst global probabilities from end-to-end CT-CLIP forward."""

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

    def text_latents(self, organ: str, lesion_type: str) -> torch.Tensor:
        key = (organ, lesion_type)
        if key in self._text_cache:
            return self._text_cache[key]

        lesion_title = lesion_type.capitalize()
        positive_text = f"{lesion_title} in {organ} is present."
        negative_text = f"{lesion_title} in {organ} is not present."
        text_tensor = self.tokenizer(
            [positive_text, negative_text],
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

    def extract_global_latent(self, nii_path: str) -> torch.Tensor:
        """
        Match CTCLIP.forward global image path:
        [1,24,24,24,512] -> temporal mean -> [1,24*24*512] -> [1,512].
        """
        volume_tensor = self.extractor.load_and_preprocess_volume(nii_path)
        with torch.no_grad():
            enc_image = self.clip_model.visual_transformer(
                volume_tensor.to(self.device),
                return_encoded_tokens=True,
            )
            if tuple(enc_image.shape) != EXPECTED_EMBEDDING_SHAPE:
                raise ValueError(f"Unexpected embedding shape: {tuple(enc_image.shape)}")
            image_features = enc_image.mean(dim=1).reshape(1, -1)
            latents = self.clip_model.to_visual_latent(image_features)
            latents = F.normalize(latents, dim=-1)
        return latents

    def score_lesion_type(
        self,
        image_latent: torch.Tensor,
        organ: str,
        lesion_type: str,
    ) -> float:
        """Return global positive probability using the same path as CTCLIP.forward."""
        text_latents = self.text_latents(organ, lesion_type)
        with torch.no_grad():
            similarity = (image_latent @ text_latents.T).squeeze(0)
            similarity = similarity * self.clip_model.temperature.exp()
            probability = F.softmax(similarity, dim=0)[0]
        return round(float(probability.detach().cpu().item()), 6)

    def score_case(self, image_id: str, dataset: str, nii_path: str) -> Dict[str, object]:
        image_latent = self.extract_global_latent(nii_path)
        organs: Dict[str, Dict[str, float]] = {}

        for organ in CANONICAL_ORGANS:
            tumor_prob = self.score_lesion_type(image_latent, organ, "tumor")
            cyst_prob = self.score_lesion_type(image_latent, organ, "cyst")
            lesion_prob = round(max(tumor_prob, cyst_prob), 6)

            organs[organ] = {
                "tumor": tumor_prob,
                "cyst": cyst_prob,
                "lesion": lesion_prob,
            }

        return {
            "image_id": image_id,
            "dataset": dataset,
            "organs": organs,
        }


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
    output_path = os.path.join(output_dir, f"{case_json['image_id']}.json")
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
        "base_lesion_types": list(BASE_LESION_TYPES),
        "aggregation": "ctclip_forward_temporal_mean_then_global_projection",
        "lesion_rule": "max(tumor,cyst)",
    }


def write_log(output_dir: str, summary: Dict[str, object], results: List[Dict[str, object]]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "classification_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute unified global tumor/cyst/lesion probabilities with end-to-end CT-CLIP forward."
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--csv_path", default=DEFAULT_CSV_PATH)
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("CT-CLIP unified global classification (end-to-end)")
    print("=" * 80)
    print(f"CSV: {args.csv_path}")
    print(f"Data root: {args.data_root}")
    print(f"Output: {output_dir}")
    print(f"Device: {args.device}")
    print(f"Max samples: {args.max_samples if args.max_samples is not None else 'all'}")
    print(f"Log every: {args.log_every} cases")
    print(f"Overwrite: {args.overwrite}")
    print("=" * 80)

    cases = iter_unique_cases(args.csv_path, args.max_samples)
    scorer = GlobalDiseaseScorer(args.model_path, args.device)

    results: List[Dict[str, object]] = []
    success_count = 0
    skipped_count = 0
    failed_count = 0

    for _, row in tqdm(cases.iterrows(), total=len(cases), desc="Classifying cases"):
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
                nii_path = os.path.join(args.data_root, dataset, "img", f"{image_id}.nii.gz")
                if not os.path.exists(nii_path):
                    raise FileNotFoundError(f"Missing NIfTI: {nii_path}")
                case_json = scorer.score_case(image_id, dataset, nii_path)
                save_path = save_case_json(case_json, output_dir)
                success_count += 1
                result = {
                    "image_id": image_id,
                    "dataset": dataset,
                    "status": "success",
                    "output_path": save_path,
                }
                results.append(result)
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
        status="completed",
    )
    write_log(output_dir, summary, results)

    print("\nDone")
    print(f"Total: {len(cases)}")
    print(f"Success: {success_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Failed: {failed_count}")
    print(f"Log: {os.path.join(output_dir, 'classification_log.json')}")


if __name__ == "__main__":
    main()
