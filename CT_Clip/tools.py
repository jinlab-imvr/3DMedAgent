#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Notebook-aligned runtime CT-CLIP tools."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage


PROJECT_CTCLIP_ROOT = os.environ.get("CTCLIP_PROJECT_ROOT", "models/CT-CLIP")
for _path in (
    PROJECT_CTCLIP_ROOT,
    os.path.join(PROJECT_CTCLIP_ROOT, "CT_CLIP"),
    os.path.join(PROJECT_CTCLIP_ROOT, "transformer_maskgit"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    from .clip_utils import (
        load_organ_mask as _load_organ_mask,
        load_organ_subregion_mask as _load_organ_subregion_mask,
    )
except ImportError:
    from clip_utils import (  # type: ignore
        load_organ_mask as _load_organ_mask,
        load_organ_subregion_mask as _load_organ_subregion_mask,
    )

from ct_clip import CTCLIP
from transformer_maskgit.MaskGITTransformer import CTViT
from transformers import BertModel, BertTokenizer


_DATA_ROOT = os.environ.get("DEEPTUMORVQA_DATA_ROOT", "data/DeepTumorVQA")
_SUBSET_ROOT = os.path.join(_DATA_ROOT, "Subset-v3")

DEFAULT_MODEL_PATH = os.environ.get(
    "CTCLIP_MODEL_PATH",
    os.path.join(PROJECT_CTCLIP_ROOT, "models", "CT-CLIP_v2.pt"),
)
DEFAULT_EMBEDDING_DIR = os.environ.get(
    "CTCLIP_EMBEDDING_DIR",
    os.path.join(_SUBSET_ROOT, "clip_embedding"),
)
DEFAULT_MASK_DIR = os.environ.get(
    "CTCLIP_MASK_DIR",
    os.path.join(_SUBSET_ROOT, "segmentations", "VISTA3D"),
)
DEFAULT_DATA_ROOT = os.environ.get("CTCLIP_DATA_ROOT", os.path.join(_DATA_ROOT, "data"))
DEFAULT_REPORT_DIR = os.environ.get(
    "CTCLIP_REPORT_DIR",
    os.path.join(_DATA_ROOT, "structured_report", "VISTA3D"),
)
DEFAULT_CLIP_DETAIL_DIR = os.environ.get(
    "CTCLIP_DETAIL_DIR",
    os.path.join(_SUBSET_ROOT, "clip_detail"),
)

EMBEDDING_SHAPE = (1, 24, 24, 24, 512)
TOTAL_SLICES = 240
SPATIAL_SIZE = 480
TEMPORAL_PATCHES = 24
SPATIAL_PATCHES = 24
TEMPORAL_PATCH_SIZE = 10
SPATIAL_PATCH_SIZE = 20


def _row_get(row: Any, key: str, default: Any = "") -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    try:
        value = row[key]
    except Exception:
        return default
    return default if value is None else value


def _is_na(value: Any) -> bool:
    try:
        return bool(value != value)
    except Exception:
        return False


def _to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None or _is_na(value) or value == "":
        return default
    try:
        return float(value)
    except Exception:
        return default


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    return value


def normalize_organ_name(organ: Any) -> str:
    organ = str(organ or "").strip().lower()
    return {
        "pancreatic": "pancreas",
        "left kidney": "kidney",
        "right kidney": "kidney",
        "renal": "kidney",
        "hepatic": "liver",
    }.get(organ, organ)


def normalize_lesion_type(lesion: Any) -> str:
    lesion = str(lesion or "").strip().lower()
    return {
        "tumour": "tumor",
        "mass": "tumor",
        "cysts": "cyst",
        "tumors": "tumor",
    }.get(lesion, lesion if lesion in {"tumor", "cyst", "lesion", "normal"} else "lesion")


def get_report_organ_heading(organ: str) -> str:
    organ = normalize_organ_name(organ)
    return {
        "liver": "Liver",
        "kidney": "Kidney",
        "pancreas": "Pancreas",
        "spleen": "Spleen",
        "colon": "Colon",
    }.get(organ, organ.capitalize())


def parse_numeric_mcq_options(mcq_question: Any) -> Dict[str, float]:
    matches = re.findall(r"\b([A-Z]):\s*([-+]?\d+(?:\.\d+)?)", str(mcq_question or ""))
    return {letter: float(value) for letter, value in matches}


def parse_counting_mcq_options(mcq_question: Any) -> Dict[str, int]:
    return {letter: int(round(value)) for letter, value in parse_numeric_mcq_options(mcq_question).items()}


def parse_hu_difference_options(mcq_question: Any) -> Dict[str, float]:
    return parse_numeric_mcq_options(mcq_question)


def counting_lesion_types(lesion_type: Any) -> List[str]:
    lesion_type = normalize_lesion_type(lesion_type)
    return [lesion_type] if lesion_type != "normal" else ["tumor", "cyst", "lesion"]


def hu_lesion_types(lesion_type: Any) -> List[str]:
    lesion_type = normalize_lesion_type(lesion_type)
    if lesion_type == "tumor":
        return ["tumor"]
    if lesion_type == "cyst":
        return ["cyst"]
    return ["tumor", "cyst"]


def compute_patch_area_grid(organ_mask: torch.Tensor) -> torch.Tensor:
    if tuple(organ_mask.shape) != (TOTAL_SLICES, SPATIAL_SIZE, SPATIAL_SIZE):
        raise ValueError(f"Mask shape mismatch: {tuple(organ_mask.shape)}")
    mask = organ_mask.float().contiguous()
    return mask.reshape(
        TEMPORAL_PATCHES,
        TEMPORAL_PATCH_SIZE,
        SPATIAL_PATCHES,
        SPATIAL_PATCH_SIZE,
        SPATIAL_PATCHES,
        SPATIAL_PATCH_SIZE,
    ).sum(dim=(1, 3, 5))


@dataclass
class CTClipRuntimeContext:
    model_path: str = DEFAULT_MODEL_PATH
    embedding_dir: str = DEFAULT_EMBEDDING_DIR
    mask_dir: str = DEFAULT_MASK_DIR
    data_root: str = DEFAULT_DATA_ROOT
    report_dir: str = DEFAULT_REPORT_DIR
    clip_detail_dir: str = DEFAULT_CLIP_DETAIL_DIR
    device: str = "cuda"

    def __post_init__(self) -> None:
        self.device_obj = torch.device(self.device if torch.cuda.is_available() else "cpu")
        if self.device_obj.type == "cuda":
            torch.cuda.set_device(self.device_obj)

        print("🔧 初始化CT-CLIP runtime tool模型...")
        print(f"   Device: {self.device_obj}")
        self.tokenizer = BertTokenizer.from_pretrained(
            "microsoft/BiomedVLP-CXR-BERT-specialized",
            do_lower_case=True,
            local_files_only=True,
        )
        text_encoder = BertModel.from_pretrained(
            "microsoft/BiomedVLP-CXR-BERT-specialized",
            local_files_only=True,
        )
        text_encoder.resize_token_embeddings(len(self.tokenizer))

        image_encoder = CTViT(
            dim=512,
            codebook_size=8192,
            image_size=480,
            patch_size=20,
            temporal_patch_size=10,
            spatial_depth=4,
            temporal_depth=4,
            dim_head=32,
            heads=8,
        )
        self.clip_model = CTCLIP(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            dim_image=294912,
            dim_text=768,
            dim_latent=512,
            extra_latent_projection=False,
            use_mlm=False,
            downsample_image_embeds=False,
            use_all_token_embeds=False,
            image_size=(32, 480, 480),
        )
        print(f"📦 加载模型权重: {self.model_path}")
        state_dict = torch.load(self.model_path, map_location="cpu")
        missing_keys, unexpected_keys = self.clip_model.load_state_dict(state_dict, strict=False)
        del state_dict
        print(f"   Missing keys: {len(missing_keys)}, unexpected keys: {len(unexpected_keys)}")
        self.clip_model = self.clip_model.to(self.device_obj)
        self.clip_model.eval()
        print("✅ Runtime tool模型加载完成!")

        self._text_cache: Dict[Tuple[str, str], torch.Tensor] = {}
        self._embedding_cache: Dict[str, torch.Tensor] = {}
        self._mask_cache: Dict[Tuple[str, str, str], torch.Tensor] = {}
        self._volume_cache: Dict[Tuple[str, str], Tuple[np.ndarray, Tuple[float, ...]]] = {}
        self._organ_hu_cache: Dict[Tuple[str, str], Optional[Dict[str, Any]]] = {}
        self._detail_cache: Dict[str, Optional[Dict[str, Any]]] = {}

    @property
    def device_torch(self) -> torch.device:
        return self.device_obj

    def load_embedding(self, image_id: str) -> torch.Tensor:
        image_id = str(image_id)
        if image_id in self._embedding_cache:
            return self._embedding_cache[image_id].to(self.device_obj)

        path = os.path.join(self.embedding_dir, f"{image_id}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing embedding: {path}")
        data = torch.load(path, map_location="cpu")
        if isinstance(data, dict):
            embedding = data.get("embedding", data.get("enc_image"))
        else:
            embedding = data
        if embedding is None:
            raise ValueError(f"Unknown embedding format: {path}")
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.as_tensor(embedding)
        embedding = embedding.float()
        if tuple(embedding.shape) != EMBEDDING_SHAPE:
            raise ValueError(f"Unexpected embedding shape for {image_id}: {tuple(embedding.shape)}")
        self._embedding_cache[image_id] = embedding
        return embedding.to(self.device_obj)

    def load_mask(self, dataset: str, image_id: str, organ: str) -> torch.Tensor:
        organ = normalize_organ_name(organ)
        key = (str(dataset), str(image_id), organ)
        if key not in self._mask_cache:
            mask = _load_organ_mask(str(dataset), str(image_id), organ, self.mask_dir)
            if mask is None:
                raise FileNotFoundError(f"Missing organ mask for {dataset}/{image_id}/{organ}")
            self._mask_cache[key] = mask.float()
        return self._mask_cache[key].to(self.device_obj)

    def text_latents(self, organ: str, lesion_type: str) -> torch.Tensor:
        organ = normalize_organ_name(organ)
        lesion_type = normalize_lesion_type(lesion_type)
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
        ).to(self.device_obj)
        with torch.no_grad():
            text_outputs = self.clip_model.text_transformer(**text_tensor)
            text_embeds = text_outputs.last_hidden_state[:, 0, :]
            latents = self.clip_model.to_text_latent(text_embeds)
            latents = F.normalize(latents, dim=-1)
        self._text_cache[key] = latents
        return latents

    def compute_patch_scores(
        self,
        enc_image: torch.Tensor,
        organ_mask: torch.Tensor,
        organ: str,
        lesion_types: Iterable[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if tuple(enc_image.shape) != EMBEDDING_SHAPE:
            raise ValueError(f"Embedding shape mismatch: {tuple(enc_image.shape)}")
        patch_features = enc_image.squeeze(0).reshape(-1, 512).float()
        patch_features = F.normalize(patch_features, dim=-1)

        score_vectors: List[torch.Tensor] = []
        scores_by_type: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for lesion_type in lesion_types:
                lesion_type = normalize_lesion_type(lesion_type)
                text_latents = self.text_latents(organ, lesion_type)
                similarity = patch_features @ text_latents.T
                probabilities = F.softmax(similarity, dim=1)
                positive_scores = probabilities[:, 1]
                scores_by_type[lesion_type] = positive_scores.reshape(24, 24, 24)
                score_vectors.append(positive_scores)

        if not score_vectors:
            raise ValueError("No lesion types provided")
        final_scores = torch.stack(score_vectors, dim=0).max(dim=0)[0] if len(score_vectors) > 1 else score_vectors[0]
        scores_3d = final_scores.reshape(24, 24, 24)
        organ_patch_area = compute_patch_area_grid(organ_mask)
        organ_patch_mask = organ_patch_area > 0
        return scores_3d, organ_patch_area, organ_patch_mask, scores_by_type

    def load_aligned_ct_volume(self, dataset: str, image_id: str) -> Tuple[np.ndarray, Tuple[float, ...]]:
        key = (str(dataset), str(image_id))
        if key in self._volume_cache:
            return self._volume_cache[key]

        nii_path = os.path.join(self.data_root, str(dataset), "img", f"{image_id}.nii.gz")
        if not os.path.exists(nii_path):
            raise FileNotFoundError(f"Missing CT file: {nii_path}")
        nii = nib.load(nii_path)
        volume_data = nii.get_fdata()
        voxel_spacing = tuple(float(v) for v in nii.header.get_zooms())

        volume_tensor = torch.tensor(volume_data, dtype=torch.float32)
        target_shape = (480, 480, 240)
        h, w, d = volume_tensor.shape
        dh, dw, dd = target_shape
        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end = min(d_start + dd, d)
        volume_tensor = volume_tensor[h_start:h_end, w_start:w_end, d_start:d_end]

        pad_h_before = (dh - volume_tensor.size(0)) // 2
        pad_h_after = dh - volume_tensor.size(0) - pad_h_before
        pad_w_before = (dw - volume_tensor.size(1)) // 2
        pad_w_after = dw - volume_tensor.size(1) - pad_w_before
        pad_d_before = (dd - volume_tensor.size(2)) // 2
        pad_d_after = dd - volume_tensor.size(2) - pad_d_before
        volume_tensor = F.pad(
            volume_tensor,
            (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after),
            value=-1024,
        )
        volume_dhw = volume_tensor.permute(2, 0, 1).numpy()
        self._volume_cache[key] = (volume_dhw, voxel_spacing)
        return self._volume_cache[key]

    def load_organ_hu_from_report(self, image_id: str, organ: str) -> Optional[Dict[str, Any]]:
        organ = normalize_organ_name(organ)
        key = (str(image_id), organ)
        if key in self._organ_hu_cache:
            return self._organ_hu_cache[key]

        report_path = os.path.join(self.report_dir, f"{image_id}_report.csv")
        if not os.path.exists(report_path):
            self._organ_hu_cache[key] = None
            return None
        with open(report_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows or "Report" not in rows[0]:
            self._organ_hu_cache[key] = None
            return None
        heading = re.escape(get_report_organ_heading(organ))
        pattern = rf"{heading}:.*?Mean HU value:\s*([-+]?\d+(?:\.\d+)?)\s*\+/-\s*([-+]?\d+(?:\.\d+)?)"
        match = re.search(pattern, rows[0]["Report"], flags=re.DOTALL)
        if not match:
            self._organ_hu_cache[key] = None
            return None
        self._organ_hu_cache[key] = {
            "mean_hu": float(match.group(1)),
            "std_hu": float(match.group(2)),
            "report_path": report_path,
        }
        return self._organ_hu_cache[key]

    def load_clip_detail(self, image_id: str) -> Optional[Dict[str, Any]]:
        image_id = str(image_id)
        if image_id not in self._detail_cache:
            path = os.path.join(self.clip_detail_dir, f"{image_id}.json")
            if not os.path.exists(path):
                self._detail_cache[image_id] = None
            else:
                with open(path, "r", encoding="utf-8") as f:
                    self._detail_cache[image_id] = json.load(f)
        return self._detail_cache[image_id]


def count_lesions_with_3d_clustering(
    scores_3d: torch.Tensor,
    mask_3d: torch.Tensor,
    threshold: float,
    min_size: int,
    connectivity: int,
) -> Tuple[int, np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    scores_masked = scores_3d.clone()
    scores_masked[~mask_3d] = 0.0
    binary_mask = (scores_masked > threshold).detach().cpu().numpy()
    total_high_conf = int(binary_mask.sum())

    if total_high_conf == 0:
        return 0, np.zeros_like(binary_mask, dtype=int), [], {
            "threshold": threshold,
            "connectivity": connectivity,
            "min_size": min_size,
            "total_high_conf_patches": 0,
            "num_components_before_filter": 0,
            "num_components_after_filter": 0,
            "component_sizes": [],
            "avg_component_size": 0.0,
            "max_component_size": 0,
            "min_component_size": 0,
        }

    structure = ndimage.generate_binary_structure(3, connectivity)
    labeled_array, num_features = ndimage.label(binary_mask, structure=structure)
    scores_np = scores_3d.detach().cpu().numpy()

    components_info: List[Dict[str, Any]] = []
    filtered_labels: List[int] = []
    for label_id in range(1, int(num_features) + 1):
        component_mask = labeled_array == label_id
        component_size = int(component_mask.sum())
        if component_size >= min_size:
            component_scores = scores_np[component_mask]
            center = ndimage.center_of_mass(component_mask)
            components_info.append({
                "id": len(filtered_labels) + 1,
                "size": component_size,
                "center_location": [int(c) for c in center],
                "avg_score": float(component_scores.mean()),
                "max_score": float(component_scores.max()),
                "min_score": float(component_scores.min()),
                "std_score": float(component_scores.std()),
            })
            filtered_labels.append(label_id)

    filtered_labeled_array = np.zeros_like(labeled_array)
    for new_label, old_label in enumerate(filtered_labels, start=1):
        filtered_labeled_array[labeled_array == old_label] = new_label

    component_sizes = [c["size"] for c in components_info]
    details = {
        "threshold": threshold,
        "connectivity": connectivity,
        "min_size": min_size,
        "total_high_conf_patches": total_high_conf,
        "num_components_before_filter": int(num_features),
        "num_components_after_filter": len(filtered_labels),
        "component_sizes": component_sizes,
        "avg_component_size": float(np.mean(component_sizes)) if component_sizes else 0.0,
        "max_component_size": int(max(component_sizes)) if component_sizes else 0,
        "min_component_size": int(min(component_sizes)) if component_sizes else 0,
    }
    return len(filtered_labels), filtered_labeled_array, components_info, details


def run_counting_tool(
    row: Any,
    context: CTClipRuntimeContext,
    threshold: float = 0.6,
    min_size: int = 3,
    connectivity: int = 2,
) -> Dict[str, Any]:
    image_id = str(_row_get(row, "Image ID"))
    dataset = str(_row_get(row, "dataset"))
    organ = normalize_organ_name(_row_get(row, "organ"))
    lesion_type = normalize_lesion_type(_row_get(row, "lesion"))
    mcq_question = _row_get(row, "multiple-choice question")
    correct_option = str(_row_get(row, "correct option", "")).strip().upper()
    options = parse_counting_mcq_options(mcq_question)

    base = {
        "tool_name": "ctclip_counting",
        "available": False,
        "image_id": image_id,
        "dataset": dataset,
        "organ": organ,
        "lesion_type": lesion_type,
        "lesion_types_checked": counting_lesion_types(lesion_type),
        "parameters": {"threshold": threshold, "min_size": min_size, "connectivity": connectivity},
        "options": options,
        "correct_option": correct_option,
    }
    if not options:
        return {**base, "failure_reason": "could_not_parse_counting_options"}

    try:
        enc_image = context.load_embedding(image_id)
        organ_mask = context.load_mask(dataset, image_id, organ)
        scores_3d, _, mask_3d, _ = context.compute_patch_scores(
            enc_image,
            organ_mask,
            organ,
            counting_lesion_types(lesion_type),
        )
        count, _, components, details = count_lesions_with_3d_clustering(
            scores_3d,
            mask_3d,
            threshold=threshold,
            min_size=min_size,
            connectivity=connectivity,
        )
        diffs = {opt: abs(value - count) for opt, value in options.items()}
        min_diff = min(diffs.values())
        recommended_options = [opt for opt, diff in diffs.items() if diff == min_diff]
        recommended_option = recommended_options[0] if recommended_options else None
        correct_count = options.get(correct_option)
        valid_scores = scores_3d[mask_3d]
        score_statistics = {
            "mean": float(valid_scores.mean().item()) if valid_scores.numel() > 0 else 0.0,
            "std": float(valid_scores.std().item()) if valid_scores.numel() > 1 else 0.0,
            "max": float(valid_scores.max().item()) if valid_scores.numel() > 0 else 0.0,
            "min": float(valid_scores.min().item()) if valid_scores.numel() > 0 else 0.0,
        }
        result = {
            **base,
            "available": True,
            "failure_reason": "",
            "predicted_count": int(count),
            "correct_count": int(correct_count) if correct_count is not None else None,
            "count_diff": abs(int(correct_count) - int(count)) if correct_count is not None else None,
            "recommended_option": recommended_option,
            "recommended_options": recommended_options,
            "option_count_differences": diffs,
            "components": components,
            "clustering_details": details,
            "score_statistics": score_statistics,
        }
        return _jsonify(result)
    except Exception as exc:
        return {**base, "available": False, "failure_reason": str(exc)}


def load_method5_clip_detail_prior(
    image_id: str,
    organ: str,
    lesion_types: Iterable[str],
    context: CTClipRuntimeContext,
    top_sections: int = 3,
) -> Dict[str, Any]:
    organ = normalize_organ_name(organ)
    detail_path = os.path.join(context.clip_detail_dir, f"{image_id}.json")
    detail = context.load_clip_detail(image_id)
    if not detail:
        return {
            "available": False,
            "detail_path": detail_path,
            "section_indices": [],
            "sections": [],
            "fallback_reason": "missing_clip_detail_json",
        }
    organ_detail = (detail.get("organs") or {}).get(organ)
    if not organ_detail:
        return {
            "available": False,
            "detail_path": detail_path,
            "section_indices": [],
            "sections": [],
            "fallback_reason": "missing_organ_in_clip_detail",
        }

    section_by_index: Dict[int, Dict[str, Any]] = {}
    for lesion_type in lesion_types:
        lesion_detail = organ_detail.get(normalize_lesion_type(lesion_type))
        if not lesion_detail:
            continue
        for section in lesion_detail.get("sections", []):
            section_index = int(section["section_index"])
            probability = float(section.get("probability", 0.0))
            current = section_by_index.get(section_index)
            if current is None or probability > current["probability"]:
                section_by_index[section_index] = {
                    "section_index": section_index,
                    "probability": probability,
                    "source_type": normalize_lesion_type(lesion_type),
                    "slice_index_range": section.get("slice_index_range"),
                    "z_percent_range": section.get("z_percent_range"),
                    "organ_area": section.get("organ_area"),
                    "organ_patch_count": section.get("organ_patch_count"),
                }

    sections = sorted(section_by_index.values(), key=lambda item: item["probability"], reverse=True)
    sections = sections[: int(top_sections)]
    return {
        "available": bool(sections),
        "detail_path": detail_path,
        "section_indices": [int(s["section_index"]) for s in sections],
        "sections": sections,
        "fallback_reason": None if sections else "empty_clip_detail_sections",
    }


def build_temporal_prior_mask(section_indices: Iterable[int], device: torch.device) -> torch.Tensor:
    mask = torch.zeros(24, dtype=torch.bool, device=device)
    for section_index in section_indices:
        section_index = int(section_index)
        if 0 <= section_index < 24:
            mask[section_index] = True
    return mask


def build_ordered_lesion_candidate_patches(
    scores_3d: torch.Tensor,
    organ_patch_area: torch.Tensor,
    top_percent: float = 0.005,
    max_patches: int = 8,
    selection_strategy: str = "topk_score_component",
    temporal_prior_mask: Optional[torch.Tensor] = None,
    connectivity: int = 2,
) -> Tuple[List[Tuple[torch.Tensor, Dict[str, Any]]], Dict[str, Any]]:
    base_organ_patch_mask = organ_patch_area > 0
    if base_organ_patch_mask.sum().item() == 0:
        return [], {"selection_mode": "failed", "failure_reason": "empty_organ_mask"}

    prior_fallback_reason = None
    if temporal_prior_mask is not None and temporal_prior_mask.any().item():
        prior_mask_3d = temporal_prior_mask[:, None, None].expand(24, 24, 24)
        organ_patch_mask = base_organ_patch_mask & prior_mask_3d
        if organ_patch_mask.sum().item() == 0:
            organ_patch_mask = base_organ_patch_mask
            prior_fallback_reason = "empty_prior_organ_overlap"
    else:
        organ_patch_mask = base_organ_patch_mask
        if temporal_prior_mask is not None:
            prior_fallback_reason = "empty_temporal_prior_mask"

    valid_scores = scores_3d[organ_patch_mask]
    valid_flat_indices = organ_patch_mask.flatten().nonzero(as_tuple=False).flatten()
    num_valid = int(valid_scores.numel())
    num_top = max(1, int(math.ceil(num_valid * top_percent)))
    num_top = min(num_top, int(max_patches), num_valid)
    top_values, top_positions = torch.topk(valid_scores, k=num_top)
    top_flat_indices = valid_flat_indices[top_positions]
    flat_scores = scores_3d.flatten()
    flat_area = organ_patch_area.flatten()

    def make_single_patch_candidate(flat_idx: torch.Tensor, mode: str, rank: int = 0):
        selected_flat = torch.zeros(24 * 24 * 24, dtype=torch.bool, device=scores_3d.device)
        selected_flat[flat_idx] = True
        selected_mask = selected_flat.reshape(24, 24, 24)
        return selected_mask, {
            "selection_mode": mode,
            "num_valid_organ_patches": num_valid,
            "num_candidate_patches": 1,
            "candidate_rank": int(rank),
            "num_components": 1,
            "selected_component_label": 1,
            "selected_component_area": float(flat_area[flat_idx].item()),
            "selected_component_patch_count": 1,
            "selected_component_mean_score": float(flat_scores[flat_idx].item()),
            "component_patch_counts": [1],
            "prior_fallback_reason": prior_fallback_reason,
            "top_score_min": float(top_values.min().item()),
            "top_score_max": float(top_values.max().item()),
        }

    if selection_strategy == "top1_patch":
        return [make_single_patch_candidate(top_flat_indices[0], "top1_patch")], {}

    candidate_mask_flat = torch.zeros(24 * 24 * 24, dtype=torch.bool, device=scores_3d.device)
    candidate_mask_flat[top_flat_indices] = True
    candidate_np = candidate_mask_flat.reshape(24, 24, 24).detach().cpu().numpy()
    structure = ndimage.generate_binary_structure(3, connectivity)
    labeled, num_components = ndimage.label(candidate_np, structure=structure)

    candidates: List[Tuple[float, torch.Tensor, Dict[str, Any]]] = []
    if num_components > 0:
        labeled_t = torch.tensor(labeled, device=scores_3d.device)
        component_patch_counts: List[int] = []
        for label_id in range(1, int(num_components) + 1):
            component_mask = labeled_t == label_id
            component_area = float(organ_patch_area[component_mask].sum().item())
            component_mean_score = float(scores_3d[component_mask].mean().item())
            component_max_score = float(scores_3d[component_mask].max().item())
            component_patch_count = int(component_mask.sum().item())
            component_patch_counts.append(component_patch_count)
            if selection_strategy == "topk_area_component":
                component_metric = component_area
            elif selection_strategy == "topk_max_score_component":
                component_metric = component_max_score
            else:
                component_metric = component_mean_score
            candidates.append((
                component_metric,
                component_mask,
                {
                    "selection_mode": selection_strategy,
                    "num_valid_organ_patches": num_valid,
                    "num_candidate_patches": num_top,
                    "num_components": int(num_components),
                    "selected_component_label": int(label_id),
                    "selected_component_area": component_area,
                    "selected_component_patch_count": component_patch_count,
                    "selected_component_mean_score": component_mean_score,
                    "selected_component_max_score": component_max_score,
                    "component_patch_counts": list(component_patch_counts),
                    "prior_fallback_reason": prior_fallback_reason,
                    "top_score_min": float(top_values.min().item()),
                    "top_score_max": float(top_values.max().item()),
                },
            ))

    if not candidates:
        return [make_single_patch_candidate(top_flat_indices[0], "top1_patch_fallback")], {}

    candidates = sorted(candidates, key=lambda item: item[0], reverse=True)
    ordered = []
    for rank, (_, mask, details) in enumerate(candidates):
        details = dict(details)
        details["candidate_rank"] = int(rank)
        ordered.append((mask, details))
    return ordered, {}


def patch_mask_to_organ_filtered_voxels(
    patch_mask: torch.Tensor,
    organ_mask: torch.Tensor,
) -> Tuple[np.ndarray, Dict[str, int]]:
    patch_mask_np = patch_mask.detach().cpu().numpy().astype(bool)
    patch_region = np.zeros((240, 480, 480), dtype=bool)
    selected_indices = np.argwhere(patch_mask_np)
    for d_idx, h_idx, w_idx in selected_indices:
        d_start, d_end = d_idx * 10, min((d_idx + 1) * 10, 240)
        h_start, h_end = h_idx * 20, min((h_idx + 1) * 20, 480)
        w_start, w_end = w_idx * 20, min((w_idx + 1) * 20, 480)
        patch_region[d_start:d_end, h_start:h_end, w_start:w_end] = True
    organ_np = organ_mask.detach().cpu().numpy() > 0
    organ_filtered_region = patch_region & organ_np
    return organ_filtered_region, {
        "lesion_voxels_before_clip": int(organ_filtered_region.sum()),
        "selected_patch_voxels_before_organ_filter": int(patch_region.sum()),
    }


def compute_hu_values_for_candidate(
    patch_mask: torch.Tensor,
    organ_mask: torch.Tensor,
    volume_dhw: np.ndarray,
    hu_clip: Optional[Tuple[float, float]],
) -> Tuple[Optional[np.ndarray], Dict[str, Any], Optional[str]]:
    lesion_region, voxel_details = patch_mask_to_organ_filtered_voxels(patch_mask, organ_mask)
    lesion_values = volume_dhw[lesion_region]
    if lesion_values.size == 0:
        return None, voxel_details, "empty_lesion_voxels_after_organ_filter"
    if hu_clip is None:
        return lesion_values, {
            **voxel_details,
            "lesion_voxels_after_clip": int(lesion_values.size),
            "used_hu_clip": False,
        }, None
    clip_low, clip_high = hu_clip
    clipped_values = lesion_values[(lesion_values >= clip_low) & (lesion_values <= clip_high)]
    if clipped_values.size == 0:
        return None, {
            **voxel_details,
            "lesion_voxels_after_clip": 0,
            "used_hu_clip": True,
        }, "empty_after_hu_clip"
    return clipped_values, {
        **voxel_details,
        "lesion_voxels_after_clip": int(clipped_values.size),
        "used_hu_clip": True,
    }, None


def estimate_lesion_hu_with_clip(
    image_id: str,
    dataset: str,
    organ: str,
    lesion_types: List[str],
    context: CTClipRuntimeContext,
    top_percent: float = 0.005,
    max_patches: int = 8,
    selection_strategy: str = "topk_score_component",
    hu_stat: str = "mean",
    hu_clip: Optional[Tuple[float, float]] = (-200, 300),
    prior_top_sections: int = 3,
    use_method5_prior: bool = True,
    fallback_scan_patches: int = 128,
) -> Dict[str, Any]:
    organ = normalize_organ_name(organ)
    enc_image = context.load_embedding(image_id)
    organ_mask = context.load_mask(dataset, image_id, organ)
    volume_dhw, voxel_spacing = context.load_aligned_ct_volume(dataset, image_id)
    scores_3d, organ_patch_area, organ_patch_mask, _ = context.compute_patch_scores(
        enc_image,
        organ_mask,
        organ,
        lesion_types,
    )

    prior_info = {
        "available": False,
        "section_indices": [],
        "sections": [],
        "fallback_reason": "method5_prior_disabled",
    }
    temporal_prior_mask = None
    if use_method5_prior:
        prior_info = load_method5_clip_detail_prior(
            image_id,
            organ,
            lesion_types,
            context,
            top_sections=prior_top_sections,
        )
        if prior_info["section_indices"]:
            temporal_prior_mask = build_temporal_prior_mask(prior_info["section_indices"], context.device_obj)

    candidates, failure_details = build_ordered_lesion_candidate_patches(
        scores_3d,
        organ_patch_area,
        top_percent=top_percent,
        max_patches=max_patches,
        selection_strategy=selection_strategy,
        temporal_prior_mask=temporal_prior_mask,
    )
    if not candidates:
        return {
            "failed": True,
            "failure_reason": failure_details.get("failure_reason", "candidate_selection_failed"),
            "selection_details": failure_details,
            "prior_info": prior_info,
        }

    rejected_candidates = []
    selected_values = None
    selected_voxel_details = None
    selected_details = None
    for patch_mask, details in candidates:
        values, voxel_details, reject_reason = compute_hu_values_for_candidate(
            patch_mask,
            organ_mask,
            volume_dhw,
            hu_clip,
        )
        if values is not None:
            selected_values = values
            selected_voxel_details = voxel_details
            selected_details = details
            break
        rejected_candidates.append({
            "selection_mode": details.get("selection_mode"),
            "candidate_rank": details.get("candidate_rank"),
            "reason": reject_reason,
            "lesion_voxels_before_clip": voxel_details.get("lesion_voxels_before_clip"),
            "lesion_voxels_after_clip": voxel_details.get("lesion_voxels_after_clip"),
        })

    if selected_values is None:
        organ_patch_mask_flat = (organ_patch_area > 0).flatten()
        valid_scores = scores_3d.flatten()[organ_patch_mask_flat]
        valid_flat_indices = organ_patch_mask_flat.nonzero(as_tuple=False).flatten()
        scan_k = min(int(fallback_scan_patches), int(valid_scores.numel()))
        _, scan_positions = torch.topk(valid_scores, k=scan_k)
        scan_flat_indices = valid_flat_indices[scan_positions]
        for rank, flat_idx in enumerate(scan_flat_indices):
            selected_flat = torch.zeros(24 * 24 * 24, dtype=torch.bool, device=scores_3d.device)
            selected_flat[flat_idx] = True
            patch_mask = selected_flat.reshape(24, 24, 24)
            values, voxel_details, reject_reason = compute_hu_values_for_candidate(
                patch_mask,
                organ_mask,
                volume_dhw,
                hu_clip,
            )
            if values is not None:
                selected_values = values
                selected_voxel_details = voxel_details
                selected_details = {
                    "selection_mode": "top_patch_soft_tissue_fallback",
                    "candidate_rank": int(rank),
                    "num_valid_organ_patches": int(valid_scores.numel()),
                    "num_candidate_patches": 1,
                    "num_components": 1,
                    "selected_component_label": 1,
                    "selected_component_area": float(organ_patch_area.flatten()[flat_idx].item()),
                    "selected_component_patch_count": 1,
                    "selected_component_mean_score": float(scores_3d.flatten()[flat_idx].item()),
                    "component_patch_counts": [1],
                    "prior_fallback_reason": candidates[0][1].get("prior_fallback_reason"),
                    "top_score_min": float(valid_scores.min().item()),
                    "top_score_max": float(valid_scores.max().item()),
                }
                break
            rejected_candidates.append({
                "selection_mode": "top_patch_soft_tissue_fallback",
                "candidate_rank": int(rank),
                "reason": reject_reason,
                "lesion_voxels_before_clip": voxel_details.get("lesion_voxels_before_clip"),
                "lesion_voxels_after_clip": voxel_details.get("lesion_voxels_after_clip"),
            })

    if selected_values is None or selected_voxel_details is None:
        return {
            "failed": True,
            "failure_reason": "no_soft_tissue_voxels_in_candidates",
            "selection_details": candidates[0][1] if candidates else failure_details,
            "prior_info": prior_info,
            "rejected_candidates": rejected_candidates[:20],
        }

    lesion_hu_for_difference = float(np.median(selected_values)) if hu_stat == "median" else float(np.mean(selected_values))
    valid_scores_for_stats = scores_3d[organ_patch_mask]
    score_stats = {
        "score_min": float(valid_scores_for_stats.min().item()),
        "score_max": float(valid_scores_for_stats.max().item()),
        "score_mean": float(valid_scores_for_stats.mean().item()),
        "score_median": float(valid_scores_for_stats.median().item()),
    }
    return {
        "failed": False,
        "lesion_mean_hu": float(np.mean(selected_values)),
        "lesion_hu_for_difference": lesion_hu_for_difference,
        "hu_stat": hu_stat,
        "lesion_median_hu": float(np.median(selected_values)),
        "lesion_std_hu": float(np.std(selected_values)),
        "lesion_min_hu": float(np.min(selected_values)),
        "lesion_max_hu": float(np.max(selected_values)),
        "used_hu_clip": bool(hu_clip is not None),
        "hu_clip": tuple(hu_clip) if hu_clip is not None else None,
        "voxel_spacing": tuple(float(v) for v in voxel_spacing),
        "selection_details": selected_details,
        "prior_info": prior_info,
        "rejected_candidates": rejected_candidates[:20],
        **selected_voxel_details,
        **score_stats,
    }


def run_hu_difference_tool(
    row: Any,
    context: CTClipRuntimeContext,
    top_percent: float = 0.005,
    max_patches: int = 8,
    selection_strategy: str = "topk_score_component",
    hu_stat: str = "mean",
    hu_clip: Optional[Tuple[float, float]] = (-200, 300),
    prior_top_sections: int = 3,
    use_method5_prior: bool = True,
    fallback_scan_patches: int = 128,
) -> Dict[str, Any]:
    image_id = str(_row_get(row, "Image ID"))
    dataset = str(_row_get(row, "dataset"))
    organ = normalize_organ_name(_row_get(row, "organ"))
    lesion_type = normalize_lesion_type(_row_get(row, "lesion"))
    mcq_question = _row_get(row, "multiple-choice question")
    correct_option = str(_row_get(row, "correct option", "")).strip().upper()
    options = parse_hu_difference_options(mcq_question)
    answer_value = _to_float(_row_get(row, "answer"), None)
    lesion_types = hu_lesion_types(lesion_type)
    parameters = {
        "top_percent": top_percent,
        "max_patches": max_patches,
        "selection_strategy": selection_strategy,
        "hu_stat": hu_stat,
        "hu_clip": list(hu_clip) if hu_clip is not None else None,
        "prior_top_sections": prior_top_sections,
        "use_method5_prior": use_method5_prior,
        "fallback_scan_patches": fallback_scan_patches,
    }
    base = {
        "tool_name": "ctclip_hu_difference",
        "available": False,
        "image_id": image_id,
        "dataset": dataset,
        "organ": organ,
        "lesion_type": lesion_type,
        "lesion_types_checked": lesion_types,
        "parameters": parameters,
        "options": options,
        "correct_option": correct_option,
        "answer_value": answer_value,
    }
    if not options:
        return {**base, "failure_reason": "could_not_parse_hu_options"}

    try:
        organ_hu = context.load_organ_hu_from_report(image_id, organ)
        if organ_hu is None:
            return {**base, "available": False, "failure_reason": "missing_organ_hu_in_structured_report"}
        lesion_hu = estimate_lesion_hu_with_clip(
            image_id=image_id,
            dataset=dataset,
            organ=organ,
            lesion_types=lesion_types,
            context=context,
            top_percent=top_percent,
            max_patches=max_patches,
            selection_strategy=selection_strategy,
            hu_stat=hu_stat,
            hu_clip=hu_clip,
            prior_top_sections=prior_top_sections,
            use_method5_prior=use_method5_prior,
            fallback_scan_patches=fallback_scan_patches,
        )
        if lesion_hu.get("failed"):
            return {
                **base,
                "available": False,
                "failure_reason": lesion_hu.get("failure_reason"),
                "prior_info": lesion_hu.get("prior_info"),
                "selection_details": lesion_hu.get("selection_details"),
            }
        predicted_hu_difference = abs(float(lesion_hu["lesion_hu_for_difference"]) - float(organ_hu["mean_hu"]))
        option_errors = {opt: abs(predicted_hu_difference - value) for opt, value in options.items()}
        min_error = min(option_errors.values())
        recommended_options = [opt for opt, error in option_errors.items() if error == min_error]
        recommended_option = recommended_options[0] if recommended_options else None
        single_option = len(options) < 2
        result = {
            **base,
            "available": True,
            "failure_reason": "",
            "single_option": single_option,
            "is_evaluable_mcq": not single_option,
            "predicted_hu_difference": float(predicted_hu_difference),
            "organ_mean_hu": float(organ_hu["mean_hu"]),
            "organ_std_hu": float(organ_hu["std_hu"]),
            "recommended_option": recommended_option,
            "recommended_options": recommended_options,
            "option_errors": option_errors,
            "hu_diff_signed_error": float(predicted_hu_difference) - answer_value if answer_value is not None else None,
            "hu_diff_abs_error": abs(float(predicted_hu_difference) - answer_value) if answer_value is not None else None,
            **lesion_hu,
        }
        return _jsonify(result)
    except Exception as exc:
        return {**base, "available": False, "failure_reason": str(exc)}


def weighted_location_score(scores: torch.Tensor, areas: torch.Tensor, method: str = "hybrid") -> float:
    if scores.numel() == 0 or areas.numel() == 0 or float(areas.sum().item()) == 0:
        return 0.0
    if method == "threshold_area":
        high = scores > 0.5
        return float(areas[high].sum().item() / areas.sum().item()) if high.any() else 0.0
    if method == "hybrid":
        high = scores > 0.5
        if high.any():
            return float((scores[high] * areas[high]).sum().item() / areas[high].sum().item())
    return float((scores * areas).sum().item() / areas.sum().item())


def run_location_tool(tool_call: Dict[str, Any], context: CTClipRuntimeContext) -> Dict[str, Any]:
    inputs = tool_call["inputs"]
    image_id = str(inputs["image_id"])
    dataset = str(inputs["dataset"])
    organ = normalize_organ_name(inputs["target_organ"])
    lesion_type = normalize_lesion_type(inputs["lesion_type"])
    lesion_types = counting_lesion_types(lesion_type)
    locations = inputs.get("locations") or []
    location_type = inputs.get("location_type") or "organ"
    scoring_method = (inputs.get("parameters") or {}).get("scoring_method", "hybrid")

    enc_image = context.load_embedding(image_id)
    organ_mask = context.load_mask(dataset, image_id, organ)
    scores_3d, organ_patch_area, mask_3d, _ = context.compute_patch_scores(
        enc_image,
        organ_mask,
        organ,
        lesion_types,
    )

    distributions: Dict[str, float] = {}
    all_scores: Dict[str, Any] = {}
    for location in locations:
        location_key = str(location)
        if location_type == "slice":
            try:
                percent = float(location)
                temporal_idx = max(0, min(23, int((percent / 100.0) * 24)))
                slice_idx = max(0, min(239, int((percent / 100.0) * 239)))
            except Exception:
                distributions[location_key] = 0.0
                all_scores[location_key] = {"score": 0.0, "reason": "invalid_slice_percent"}
                continue
            valid_mask = mask_3d[temporal_idx]
            valid_scores = scores_3d[temporal_idx][valid_mask]
            valid_areas = organ_patch_area[temporal_idx][valid_mask]
            score = weighted_location_score(valid_scores, valid_areas, scoring_method)
            distributions[location_key] = score
            all_scores[location_key] = {
                "score": score,
                "temporal_patch": int(temporal_idx),
                "slice_idx": int(slice_idx),
                "slice_percent": percent,
                "area": int(valid_mask.sum().item()),
                "reason": "success" if valid_scores.numel() else "no_valid_patches",
            }
            continue

        subregion_mask = _load_organ_subregion_mask(dataset, image_id, location_key, context.mask_dir)
        if subregion_mask is None:
            distributions[location_key] = 0.0
            all_scores[location_key] = {"score": 0.0, "reason": "mask_not_found"}
            continue
        subregion_area = compute_patch_area_grid(subregion_mask.to(context.device_obj).float())
        valid_mask = mask_3d & (subregion_area > 0)
        valid_scores = scores_3d[valid_mask]
        valid_areas = subregion_area[valid_mask]
        score = weighted_location_score(valid_scores, valid_areas, scoring_method)
        distributions[location_key] = score
        all_scores[location_key] = {
            "score": score,
            "mean": float(valid_scores.mean().item()) if valid_scores.numel() else 0.0,
            "max": float(valid_scores.max().item()) if valid_scores.numel() else 0.0,
            "area": int(valid_mask.sum().item()),
            "total_area": float(valid_areas.sum().item()) if valid_areas.numel() else 0.0,
            "reason": "success" if valid_scores.numel() else "no_valid_patches",
        }

    max_score = max(distributions.values()) if distributions else 0.0
    predicted_locations = [loc for loc, score in distributions.items() if abs(score - max_score) < 1e-6]
    predicted_location: Any = predicted_locations[0] if len(predicted_locations) == 1 else predicted_locations
    option_to_location = inputs.get("option_to_location") or {}
    option_scores = {opt: distributions.get(str(loc), 0.0) for opt, loc in option_to_location.items()}
    recommended_options: List[str] = []
    if option_scores:
        option_max = max(option_scores.values())
        recommended_options = [opt for opt, score in option_scores.items() if abs(score - option_max) < 1e-6]

    confidence = "low"
    if len(distributions) > 1:
        sorted_scores = sorted(distributions.values(), reverse=True)
        gap = sorted_scores[0] - sorted_scores[1]
        confidence = "high" if gap > 0.2 else "medium" if gap > 0.1 else "low"

    return _jsonify({
        "tool_name": "ctclip_location",
        "tool_call_id": tool_call.get("tool_call_id"),
        "available": True,
        "failure_reason": "",
        "image_id": image_id,
        "dataset": dataset,
        "organ": organ,
        "lesion_type": lesion_type,
        "lesion_types_checked": lesion_types,
        "locations": locations,
        "location_type": location_type,
        "proxy_warning": inputs.get("proxy_warning"),
        "distributions": distributions,
        "predicted_location": predicted_location,
        "all_scores": all_scores,
        "option_location_scores": option_scores,
        "recommended_options": recommended_options,
        "confidence": confidence,
        "reasoning": f"Highest CT-CLIP lesion score location: {predicted_location}",
    })


def run_count_by_location_tool(tool_call: Dict[str, Any], context: CTClipRuntimeContext) -> Dict[str, Any]:
    inputs = tool_call["inputs"]
    image_id = str(inputs["image_id"])
    dataset = str(inputs["dataset"])
    organ = normalize_organ_name(inputs["target_organ"])
    lesion_type = normalize_lesion_type(inputs["lesion_type"])
    lesion_types = counting_lesion_types(lesion_type)
    locations = inputs.get("locations") or []
    params = inputs.get("parameters") or {}
    threshold = float(params.get("threshold", 0.6))
    min_size = int(params.get("min_size", 3))
    connectivity = int(params.get("connectivity", 2))

    enc_image = context.load_embedding(image_id)
    organ_mask = context.load_mask(dataset, image_id, organ)
    scores_3d, _, mask_3d, _ = context.compute_patch_scores(enc_image, organ_mask, organ, lesion_types)

    location_counts: Dict[str, int] = {}
    all_scores: Dict[str, Any] = {}
    for location in locations:
        location_key = str(location)
        subregion_mask = _load_organ_subregion_mask(dataset, image_id, location_key, context.mask_dir)
        if subregion_mask is None:
            location_counts[location_key] = 0
            all_scores[location_key] = {"count": 0, "reason": "mask_not_found"}
            continue
        subregion_area = compute_patch_area_grid(subregion_mask.to(context.device_obj).float())
        subregion_patch_mask = mask_3d & (subregion_area > 0)
        count, _, components, details = count_lesions_with_3d_clustering(
            scores_3d,
            subregion_patch_mask,
            threshold=threshold,
            min_size=min_size,
            connectivity=connectivity,
        )
        location_counts[location_key] = int(count)
        all_scores[location_key] = {
            "count": int(count),
            "components": components[:5],
            "details": details,
            "reason": "success",
        }

    numeric_options = inputs.get("numeric_options") or {}
    option_texts = inputs.get("options") or {}
    recommended_options: List[str] = []
    option_count_differences: Dict[str, int] = {}
    predicted_count = None
    if len(locations) == 1:
        predicted_count = location_counts.get(str(locations[0]), 0)
        option_count_differences = {
            opt: abs(int(round(float(value))) - int(predicted_count))
            for opt, value in numeric_options.items()
        }
        if option_count_differences:
            min_diff = min(option_count_differences.values())
            recommended_options = [opt for opt, diff in option_count_differences.items() if diff == min_diff]
    elif len(locations) >= 2:
        left, right = str(locations[0]), str(locations[1])
        if location_counts.get(left, 0) == location_counts.get(right, 0):
            relation = "same"
        else:
            relation = left if location_counts.get(left, 0) > location_counts.get(right, 0) else right
        for opt, text in option_texts.items():
            low = str(text).lower()
            if relation == "same" and ("same" in low or "equal" in low):
                recommended_options.append(opt)
            elif relation != "same" and str(relation).lower() in low:
                recommended_options.append(opt)

    return _jsonify({
        "tool_name": "ctclip_count_by_location",
        "tool_call_id": tool_call.get("tool_call_id"),
        "available": True,
        "failure_reason": "",
        "image_id": image_id,
        "dataset": dataset,
        "organ": organ,
        "lesion_type": lesion_type,
        "lesion_types_checked": lesion_types,
        "locations": locations,
        "location_counts": location_counts,
        "predicted_count": predicted_count,
        "recommended_options": recommended_options,
        "option_count_differences": option_count_differences,
        "all_scores": all_scores,
        "parameters": {"threshold": threshold, "min_size": min_size, "connectivity": connectivity},
    })


def run_hu_attenuation_tool(tool_call: Dict[str, Any], context: CTClipRuntimeContext) -> Dict[str, Any]:
    inputs = tool_call["inputs"]
    image_id = str(inputs["image_id"])
    dataset = str(inputs["dataset"])
    organ = normalize_organ_name(inputs["target_organ"])
    lesion_type = normalize_lesion_type(inputs["lesion_type"])
    option_texts = inputs.get("options") or {}
    params = inputs.get("parameters") or {}

    organ_hu = context.load_organ_hu_from_report(image_id, organ)
    if organ_hu is None:
        return {
            "tool_name": "ctclip_hu_attenuation",
            "tool_call_id": tool_call.get("tool_call_id"),
            "available": False,
            "failure_reason": "missing_organ_hu_in_structured_report",
        }

    lesion_hu = estimate_lesion_hu_with_clip(
        image_id=image_id,
        dataset=dataset,
        organ=organ,
        lesion_types=hu_lesion_types(lesion_type),
        context=context,
        top_percent=float(params.get("top_percent", 0.005)),
        max_patches=int(params.get("max_patches", 8)),
        hu_stat=str(params.get("hu_stat", "mean")),
    )
    if lesion_hu.get("failed"):
        return {
            "tool_name": "ctclip_hu_attenuation",
            "tool_call_id": tool_call.get("tool_call_id"),
            "available": False,
            "failure_reason": lesion_hu.get("failure_reason"),
            "prior_info": lesion_hu.get("prior_info"),
            "selection_details": lesion_hu.get("selection_details"),
        }

    diff = float(lesion_hu["lesion_hu_for_difference"]) - float(organ_hu["mean_hu"])
    if diff < -10:
        attenuation = "hypoattenuating"
    elif diff > 10:
        attenuation = "hyperattenuating"
    else:
        attenuation = "isoattenuating"
    recommended_options = [
        opt for opt, text in option_texts.items()
        if attenuation.replace("attenuating", "") in str(text).lower()
    ]

    return _jsonify({
        "tool_name": "ctclip_hu_attenuation",
        "tool_call_id": tool_call.get("tool_call_id"),
        "available": True,
        "failure_reason": "",
        "image_id": image_id,
        "dataset": dataset,
        "organ": organ,
        "lesion_type": lesion_type,
        "predicted_attenuation": attenuation,
        "recommended_options": recommended_options,
        "organ_mean_hu": float(organ_hu["mean_hu"]),
        "lesion_hu_for_difference": float(lesion_hu["lesion_hu_for_difference"]),
        **lesion_hu,
    })


def make_runtime_row_from_tool_call(tool_call: Dict[str, Any], row: Any) -> Dict[str, Any]:
    inputs = tool_call.get("inputs") or {}
    runtime_row = dict(row) if isinstance(row, dict) else {}
    runtime_row["Image ID"] = inputs.get("image_id", _row_get(row, "Image ID"))
    runtime_row["dataset"] = inputs.get("dataset", _row_get(row, "dataset"))
    runtime_row["organ"] = inputs.get("target_organ", _row_get(row, "organ"))
    runtime_row["lesion"] = inputs.get("lesion_type", _row_get(row, "lesion"))
    runtime_row.pop("correct option", None)
    runtime_row.pop("answer", None)
    return runtime_row


def run_runtime_tool_call(
    tool_call: Dict[str, Any],
    row: Any,
    context: CTClipRuntimeContext,
) -> Dict[str, Any]:
    tool_name = str(tool_call.get("tool_name") or "")
    if tool_name == "ctclip_counting":
        result = run_counting_tool(make_runtime_row_from_tool_call(tool_call, row), context)
    elif tool_name == "ctclip_hu_difference":
        result = run_hu_difference_tool(make_runtime_row_from_tool_call(tool_call, row), context)
    elif tool_name == "ctclip_location":
        result = run_location_tool(tool_call, context)
    elif tool_name == "ctclip_count_by_location":
        result = run_count_by_location_tool(tool_call, context)
    elif tool_name == "ctclip_hu_attenuation":
        result = run_hu_attenuation_tool(tool_call, context)
    else:
        result = {
            "tool_name": tool_name,
            "tool_call_id": tool_call.get("tool_call_id"),
            "available": False,
            "failure_reason": "unsupported_tool",
        }
    result["tool_call_id"] = tool_call.get("tool_call_id", result.get("tool_call_id"))
    return _jsonify(result)
