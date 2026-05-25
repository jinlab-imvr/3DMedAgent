#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CT-CLIP通用工具函数
包含embedding加载、mask处理、分数计算等通用功能
"""

import os
import torch
import numpy as np
import nibabel as nib
from typing import Optional, List, Tuple, Dict


def canonicalize_organ_name(organ: Optional[str]) -> str:
    """
    标准化器官名称
    
    Args:
        organ: 原始器官名称
        
    Returns:
        标准化后的器官名称
    """
    if organ is None:
        return ""
    organ = str(organ).lower().strip()
    organ_map = {
        "pancreatic": "pancreas",
        "hepatic": "liver",
        "renal": "kidney",
        "splenic": "spleen",
        "colonic": "colon",
    }
    return organ_map.get(organ, organ)


def parse_lesion_types(lesion_input: Optional[List[str]]) -> List[str]:
    """
    解析和标准化lesion类型
    
    Args:
        lesion_input: lesion类型列表或单个类型
        
    Returns:
        标准化的lesion类型列表
    """
    if lesion_input is None:
        return ["tumor", "cyst", "lesion"]
    
    if isinstance(lesion_input, str):
        lesion_input = [lesion_input]
    
    valid_types = ["cyst", "tumor", "lesion"]
    result = []
    
    for lesion in lesion_input:
        lesion = str(lesion).lower().strip()
        if lesion in valid_types:
            result.append(lesion)
    
    # 去重
    result = list(set(result))
    
    # 如果为空，返回默认值
    if not result:
        result = ["tumor", "cyst"]
    
    return result


def load_precomputed_embedding(image_id: str, embedding_dir: str) -> Optional[torch.Tensor]:
    """
    加载预先提取的CT-CLIP embedding
    
    Args:
        image_id: 图像ID
        embedding_dir: embedding存储目录
        
    Returns:
        embedding tensor [1, 24, 24, 24, 512] 或 None
    """
    embedding_path = os.path.join(embedding_dir, f"{image_id}.pt")
    
    if not os.path.exists(embedding_path):
        print(f"❌ Embedding not found: {embedding_path}")
        return None
    
    try:
        data = torch.load(embedding_path, map_location='cpu')
        
        # 处理不同的保存格式
        if isinstance(data, dict):
            if 'embedding' in data:
                enc_image = data['embedding']
            elif 'enc_image' in data:
                enc_image = data['enc_image']
            else:
                print(f"❌ Unknown dict format in {embedding_path}")
                return None
        else:
            enc_image = data
        
        # 确保格式正确
        if not isinstance(enc_image, torch.Tensor):
            enc_image = torch.tensor(enc_image)
        
        # 转换为float32
        if enc_image.dtype == torch.float16:
            enc_image = enc_image.float()
        
        # 检查形状
        if enc_image.shape != (1, 24, 24, 24, 512):
            print(f"❌ Unexpected embedding shape: {enc_image.shape}")
            return None
        
        return enc_image
        
    except Exception as e:
        print(f"❌ Error loading embedding: {e}")
        return None


def load_organ_mask(dataset: str, image_id: str, organ: str, mask_dir: str) -> Optional[torch.Tensor]:
    """
    加载器官mask
    
    Args:
        dataset: 数据集名称
        image_id: 图像ID
        organ: 器官名称
        mask_dir: mask存储目录
        
    Returns:
        mask tensor [240, 480, 480] 或 None
    """
    organ = canonicalize_organ_name(organ)
    mask_path = os.path.join(mask_dir, dataset, image_id, f"{organ}.nii.gz")
    
    if not os.path.exists(mask_path):
        print(f"❌ Organ mask not found: {mask_path}")
        return None
    
    try:
        nii = nib.load(mask_path)
        mask_data = nii.get_fdata()
        
        # 转换为tensor
        mask_tensor = torch.tensor(mask_data, dtype=torch.float32)
        
        # Resize到目标形状 (480, 480, 240) - 与volume相同
        target_shape = (480, 480, 240)
        h, w, d = mask_tensor.shape
        dh, dw, dd = target_shape
        
        # Center crop/pad
        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end = min(d_start + dd, d)
        
        mask_tensor = mask_tensor[h_start:h_end, w_start:w_end, d_start:d_end]
        
        # Padding if necessary
        pad_h_before = (dh - mask_tensor.size(0)) // 2
        pad_h_after = dh - mask_tensor.size(0) - pad_h_before
        pad_w_before = (dw - mask_tensor.size(1)) // 2
        pad_w_after = dw - mask_tensor.size(1) - pad_w_before
        pad_d_before = (dd - mask_tensor.size(2)) // 2
        pad_d_after = dd - mask_tensor.size(2) - pad_d_before
        
        mask_tensor = torch.nn.functional.pad(
            mask_tensor,
            (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after),
            value=0
        )
        
        # Permute to (depth, height, width) - [D, H, W] = [240, 480, 480]
        mask_tensor = mask_tensor.permute(2, 0, 1)
        
        # 二值化
        mask_tensor = (mask_tensor > 0.5).float()
        
        return mask_tensor
        
    except Exception as e:
        print(f"❌ Error loading organ mask: {e}")
        return None


def load_organ_subregion_mask(
    dataset: str, 
    image_id: str, 
    subregion: str, 
    mask_dir: str
) -> Optional[torch.Tensor]:
    """
    加载器官子区域mask（如left kidney, liver segment等）
    
    Args:
        dataset: 数据集名称
        image_id: 图像ID
        subregion: 子区域名称，如 "left", "right", "5", "3"（直接使用MCQ解析出的原始值）
        mask_dir: mask存储目录
        
    Returns:
        mask tensor [240, 480, 480] 或 None
    """
    # 根据subregion值确定mask文件名（参考notebook的简单直接方法）
    # MCQ解析出的原始值：
    # - kidney: "left", "right"
    # - liver segment: "5", "3", "8", "1" 等数字
    
    subregion = subregion.lower().strip()
    
    # Kidney sub-regions
    if subregion == "left":
        mask_filename = "left kidney.nii.gz"
    elif subregion == "right":
        mask_filename = "right kidney.nii.gz"
    # Liver segments (数字)
    elif subregion.isdigit():
        # 肝段：liver_segment_X.nii.gz
        mask_filename = f"liver_segment_{subregion}.nii.gz"
    # 兼容旧格式
    elif "left" in subregion and "kidney" in subregion:
        mask_filename = "left kidney.nii.gz"
    elif "right" in subregion and "kidney" in subregion:
        mask_filename = "right kidney.nii.gz"
    elif "liver_segment" in subregion:
        mask_filename = f"{subregion}.nii.gz"
    else:
        # Fallback: 尝试将下划线替换为空格
        mask_filename = f"{subregion.replace('_', ' ')}.nii.gz"
    
    mask_path = os.path.join(mask_dir, dataset, image_id, mask_filename)
    
    if not os.path.exists(mask_path):
        print(f"❌ Subregion mask not found: {mask_path}")
        return None
    
    try:
        nii = nib.load(mask_path)
        mask_data = nii.get_fdata()
        
        # 转换为tensor
        mask_tensor = torch.tensor(mask_data, dtype=torch.float32)
        
        # Resize到目标形状 (480, 480, 240)
        target_shape = (480, 480, 240)
        h, w, d = mask_tensor.shape
        dh, dw, dd = target_shape
        
        # Center crop/pad
        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end = min(d_start + dd, d)
        
        mask_tensor = mask_tensor[h_start:h_end, w_start:w_end, d_start:d_end]
        
        # Padding if necessary
        pad_h_before = (dh - mask_tensor.size(0)) // 2
        pad_h_after = dh - mask_tensor.size(0) - pad_h_before
        pad_w_before = (dw - mask_tensor.size(1)) // 2
        pad_w_after = dw - mask_tensor.size(1) - pad_w_before
        pad_d_before = (dd - mask_tensor.size(2)) // 2
        pad_d_after = dd - mask_tensor.size(2) - pad_d_before
        
        mask_tensor = torch.nn.functional.pad(
            mask_tensor,
            (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after),
            value=0
        )
        
        # Permute to (depth, height, width)
        mask_tensor = mask_tensor.permute(2, 0, 1)  # [240, 480, 480]
        
        # 二值化
        mask_tensor = (mask_tensor > 0.5).float()
        
        return mask_tensor
        
    except Exception as e:
        print(f"❌ Error loading subregion mask: {e}")
        return None


def compute_all_patches_scores_fast(
    enc_image: torch.Tensor,
    organ_mask: torch.Tensor,
    organ: str,
    lesion_types: List[str],
    clip_model,
    tokenizer,
    device: torch.device,
    text_prompt_style: str = "not_present"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用矩阵乘法批量计算所有patches的lesion分数
    
    Args:
        enc_image: [1, 24, 24, 24, 512] 全局embedding
        organ_mask: [240, 480, 480] organ mask
        organ: 器官名称
        lesion_types: 要检测的病变类型列表
        clip_model: CT-CLIP模型
        tokenizer: 文本tokenizer
        device: 计算设备
        text_prompt_style: 'not_present' or 'absent'
        
    Returns:
        scores_3d: [24, 24, 24] 每个patch的lesion分数
        mask_3d: [24, 24, 24] 每个patch是否与organ重叠
    """
    # 确认维度
    assert enc_image.shape == (1, 24, 24, 24, 512), f"Embedding shape mismatch: {enc_image.shape}"
    assert organ_mask.shape == (240, 480, 480), f"Mask shape mismatch: {organ_mask.shape}"
    
    # 1. Reshape embedding: [1, 24, 24, 24, 512] → [13824, 512]
    all_patches = enc_image.squeeze(0).reshape(-1, 512)
    all_patches = all_patches.float()
    
    # 2. 构建text prompts并计算text embedding
    max_score_per_lesion = []
    
    for lesion_type in lesion_types:
        positive_text = f"{lesion_type.capitalize()} in {organ} is present."
        
        # 根据text_prompt_style选择negative prompt
        if text_prompt_style == "absent":
            negative_text = f"{lesion_type.capitalize()} in {organ} is absent."
        else:  # default: "not_present"
            negative_text = f"{lesion_type.capitalize()} in {organ} is not present."
        
        text_tensor = tokenizer(
            [negative_text, positive_text],
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            text_outputs = clip_model.text_transformer(**text_tensor)
            text_embeds = text_outputs.last_hidden_state[:, 0, :]  # [2, 768]
            text_latents = clip_model.to_text_latent(text_embeds)  # [2, 512]
            text_latents = torch.nn.functional.normalize(text_latents, dim=-1)
        
        # 3. 批量计算相似度: [13824, 512] @ [512, 2] = [13824, 2]
        all_patches_norm = torch.nn.functional.normalize(all_patches, dim=-1)
        similarity = all_patches_norm @ text_latents.T
        
        # 4. 应用softmax（对每个patch的2个类别）
        similarity = torch.nn.functional.softmax(similarity, dim=1)
        
        # 5. 取positive的概率
        scores = similarity[:, 1]
        max_score_per_lesion.append(scores)
    
    # 6. 对每个patch取所有lesion类型的最大分数
    if len(max_score_per_lesion) > 1:
        all_scores = torch.stack(max_score_per_lesion, dim=0)
        final_scores = all_scores.max(dim=0)[0]
    else:
        final_scores = max_score_per_lesion[0]
    
    # 7. Reshape成3D: [13824] → [24, 24, 24]
    scores_3d = final_scores.reshape(24, 24, 24)
    
    # 8. 构建organ mask的3D patch-level版本
    mask_3d = torch.zeros(24, 24, 24, dtype=torch.bool, device=device)
    
    for d_idx in range(24):
        for h_idx in range(24):
            for w_idx in range(24):
                d_start = d_idx * 10
                d_end = min((d_idx + 1) * 10, 240)
                h_start = h_idx * 20
                h_end = min((h_idx + 1) * 20, 480)
                w_start = w_idx * 20
                w_end = min((w_idx + 1) * 20, 480)
                
                patch_mask = organ_mask[d_start:d_end, h_start:h_end, w_start:w_end]
                if patch_mask.sum() > 0:
                    mask_3d[d_idx, h_idx, w_idx] = True
    
    return scores_3d, mask_3d


def get_top_k_slices(
    scores_3d: torch.Tensor,
    mask_3d: torch.Tensor,
    k: int = 3
) -> List[int]:
    """
    获取分数最高的k个slice索引
    
    Args:
        scores_3d: [24, 24, 24] 分数张量
        mask_3d: [24, 24, 24] mask张量
        k: 返回前k个
        
    Returns:
        slice索引列表（在原始CT的240个slices中的索引）
    """
    # 对每个depth slice计算平均分数
    slice_scores = []
    for d_idx in range(24):
        slice_mask = mask_3d[d_idx, :, :]
        if slice_mask.sum() > 0:
            slice_score = scores_3d[d_idx, :, :][slice_mask].mean().item()
        else:
            slice_score = 0.0
        slice_scores.append((d_idx, slice_score))
    
    # 排序并取top k
    slice_scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, score in slice_scores[:k] if score > 0]
    
    # 转换为原始CT的slice索引（24个patch → 240个slices）
    original_slice_indices = [idx * 10 + 5 for idx in top_indices]  # 取patch中间位置
    
    return original_slice_indices


def slice_range_to_indices(slice_range: str, total_slices: int = 240) -> Tuple[int, int]:
    """
    将百分比值或范围转换为实际slice索引
    
    Args:
        slice_range: 单个百分比 "71.9" 或范围 "0-33" 或 "34-66" (百分比)
        total_slices: 总slice数量
        
    Returns:
        (start_idx, end_idx) 实际索引范围
    """
    try:
        if "-" in slice_range:
            # 范围格式: "0-33"
            parts = slice_range.split("-")
            start_pct = float(parts[0])
            end_pct = float(parts[1])
        else:
            # 单个百分比: "71.9" -> 创建小范围 (±2%)
            percent = float(slice_range)
            start_pct = max(0, percent - 2)
            end_pct = min(100, percent + 2)
        
        start_idx = int(total_slices * start_pct / 100)
        end_idx = int(total_slices * end_pct / 100)
        
        # 确保范围有效
        start_idx = max(0, min(start_idx, total_slices - 1))
        end_idx = max(start_idx + 1, min(end_idx, total_slices))
        
        return start_idx, end_idx
    except Exception as e:
        print(f"❌ Failed to parse slice range '{slice_range}': {e}")
        return 0, total_slices


def parse_location_from_string(location_str: str) -> Tuple[str, str]:
    """
    解析位置字符串，判断是organ还是slice类型
    
    Args:
        location_str: 如 "left_kidney", "liver_segment_1", "0-33"
        
    Returns:
        (location_type, normalized_name)
        - location_type: "organ" | "slice"
        - normalized_name: 标准化的名称
    """
    if "-" in location_str and location_str.replace("-", "").isdigit():
        # 是slice范围
        return "slice", location_str
    else:
        # 是器官子区域
        return "organ", location_str


if __name__ == "__main__":
    # 测试代码
    print("CT-CLIP Utils Module")
    print(f"Organ canonicalization: pancreatic -> {canonicalize_organ_name('pancreatic')}")
    print(f"Lesion types parsing: {parse_lesion_types(['tumor', 'cyst'])}")
    print(f"Slice range conversion: 0-33 -> {slice_range_to_indices('0-33')}")
