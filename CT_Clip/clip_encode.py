#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CT-CLIP Volume Embedding Extraction Script
==========================================
提取CT volume的结构化embedding并保存到磁盘，用于后续快速加载和重复使用。

Embedding shape: [1, 24, 24, 24, 512]
- 24 temporal patches (depth direction, 10 slices per patch)
- 24×24 spatial patches (height×width, 20×20 pixels per patch)
- 512 feature dimensions

存储策略:
- 使用torch.save保存为.pt文件
- 可选float16格式节省空间（~14MB vs ~28MB per volume）
- 包含元数据（image_id, dataset, shape, dtype）
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

# Import CT-CLIP components
from transformer_maskgit.MaskGITTransformer import CTViT
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP


class CTCLIPEmbeddingExtractor:
    """CT-CLIP Volume Embedding提取器"""
    
    def __init__(self, model_path, device='cuda', use_fp16=False):
        """
        初始化模型
        
        Args:
            model_path: CT-CLIP模型权重路径
            device: 'cuda' or 'cpu'
            use_fp16: 是否使用float16保存（节省空间）
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.use_fp16 = use_fp16
        
        print(f"🔧 初始化CT-CLIP模型...")
        print(f"   Device: {self.device}")
        print(f"   FP16 模式: {use_fp16}")
        
        # 初始化tokenizer和text encoder（用于完整模型加载）
        self.tokenizer = BertTokenizer.from_pretrained(
            'microsoft/BiomedVLP-CXR-BERT-specialized', 
            do_lower_case=True
        )
        text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")
        text_encoder.resize_token_embeddings(len(self.tokenizer))
        
        # 初始化image encoder
        image_encoder = CTViT(
            dim=512,
            codebook_size=8192,
            image_size=480,
            patch_size=20,
            temporal_patch_size=10,
            spatial_depth=4,
            temporal_depth=4,
            dim_head=32,
            heads=8
        )
        
        # 初始化CLIP模型
        self.clip_model = CTCLIP(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            dim_image=294912,  # 24*24*512
            dim_text=768,
            dim_latent=512,
            extra_latent_projection=False,
            use_mlm=False,
            downsample_image_embeds=False,
            use_all_token_embeds=False,
            image_size=(32, 480, 480)
        )
        
        # 加载预训练权重
        print(f"📦 加载模型权重: {model_path}")
        state_dict = torch.load(model_path, map_location="cpu")
        missing_keys, unexpected_keys = self.clip_model.load_state_dict(state_dict, strict=False)
        del state_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"   Missing keys: {len(missing_keys)}, unexpected keys: {len(unexpected_keys)}")
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()
        print("✅ 模型加载完成!")
    
    def load_and_preprocess_volume(self, nii_path):
        """
        加载并预处理NIfTI volume
        
        Args:
            nii_path: NIfTI文件路径
        
        Returns:
            volume_tensor: [1, 1, 240, 480, 480] 预处理后的tensor
        """
        if not os.path.exists(nii_path):
            raise FileNotFoundError(f"文件不存在: {nii_path}")
        
        # 加载NIfTI
        nii = nib.load(nii_path)
        volume_data = nii.get_fdata()
        
        # HU值裁剪和归一化
        hu_min, hu_max = -1000, 200
        volume_data = np.clip(volume_data, hu_min, hu_max)
        volume_data = (((volume_data + 400) / 600)).astype(np.float32)
        
        tensor = torch.tensor(volume_data, dtype=torch.float32)
        
        # Resize到目标形状 (480, 480, 240)
        target_shape = (480, 480, 240)
        h, w, d = tensor.shape
        dh, dw, dd = target_shape
        
        # Center crop/pad
        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end = min(d_start + dd, d)
        
        tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]
        
        # Padding if necessary
        pad_h_before = (dh - tensor.size(0)) // 2
        pad_h_after = dh - tensor.size(0) - pad_h_before
        pad_w_before = (dw - tensor.size(1)) // 2
        pad_w_after = dw - tensor.size(1) - pad_w_before
        pad_d_before = (dd - tensor.size(2)) // 2
        pad_d_after = dd - tensor.size(2) - pad_d_before
        
        tensor = torch.nn.functional.pad(
            tensor,
            (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after),
            value=-1
        )
        
        # Permute to (depth, height, width) and add batch/channel dims
        # [H, W, D] -> [D, H, W] -> [1, 1, D, H, W]
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
        
        return tensor
    
    def extract_embedding(self, volume_tensor):
        """
        提取volume的结构化embedding
        
        Args:
            volume_tensor: [1, 1, 240, 480, 480] volume tensor
        
        Returns:
            embedding: [1, 24, 24, 24, 512] 结构化embedding
        """
        volume_tensor = volume_tensor.to(self.device)
        
        with torch.no_grad():
            # 调用visual_transformer获取结构化embedding
            enc_image = self.clip_model.visual_transformer(
                volume_tensor, 
                return_encoded_tokens=True
            )
        
        return enc_image
    
    def save_embedding(self, embedding, save_path, metadata=None):
        """
        保存embedding到磁盘
        
        Args:
            embedding: [1, 24, 24, 24, 512] embedding tensor
            save_path: 保存路径（.pt文件）
            metadata: 元数据字典（可选）
        """
        # 移到CPU并可选转换为float16
        embedding_cpu = embedding.cpu()
        if self.use_fp16:
            embedding_cpu = embedding_cpu.half()
        
        # 准备保存数据
        save_data = {
            'embedding': embedding_cpu,
            'shape': list(embedding.shape),
            'dtype': str(embedding_cpu.dtype),
            'created_at': datetime.now().isoformat(),
        }
        
        # 添加元数据
        if metadata:
            save_data['metadata'] = metadata
        
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存
        torch.save(save_data, save_path)
        
        # 计算文件大小
        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        
        return file_size_mb
    
    def process_single_volume(self, nii_path, image_id, dataset, output_dir):
        """
        处理单个volume并保存embedding
        
        Args:
            nii_path: NIfTI文件路径
            image_id: Image ID
            dataset: 数据集名称
            output_dir: 输出目录
        
        Returns:
            result_dict: 处理结果
        """
        try:
            # 1. 加载和预处理
            volume_tensor = self.load_and_preprocess_volume(nii_path)
            
            # 2. 提取embedding
            embedding = self.extract_embedding(volume_tensor)
            
            # 3. 准备元数据
            metadata = {
                'image_id': image_id,
                'dataset': dataset,
                'nii_path': nii_path,
                'preprocess_pad_value': -1,
            }
            
            # 4. 保存
            save_filename = f"{image_id}.pt"
            save_path = os.path.join(output_dir, save_filename)
            file_size_mb = self.save_embedding(embedding, save_path, metadata)
            
            return {
                'status': 'success',
                'image_id': image_id,
                'dataset': dataset,
                'embedding_shape': list(embedding.shape),
                'save_path': save_path,
                'file_size_mb': file_size_mb,
                'error': None
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'image_id': image_id,
                'dataset': dataset,
                'embedding_shape': None,
                'save_path': None,
                'file_size_mb': None,
                'error': str(e)
            }


def load_embedding(embedding_path, device='cuda'):
    """
    从磁盘加载embedding（工具函数）
    
    Args:
        embedding_path: embedding文件路径（.pt）
        device: 目标设备
    
    Returns:
        embedding: tensor
        metadata: 元数据字典
    """
    data = torch.load(embedding_path, map_location='cpu')
    
    embedding = data['embedding']
    
    # 转换到目标设备和dtype
    if embedding.dtype == torch.float16:
        embedding = embedding.float()  # 转回float32用于计算
    
    embedding = embedding.to(device)
    
    metadata = data.get('metadata', {})
    
    return embedding, metadata


def main():
    parser = argparse.ArgumentParser(description='CT-CLIP Volume Embedding Extraction')
    parser.add_argument('--model_path', type=str, 
                       default='/mnt/blobdata/project/CT-CLIP/models/CT-CLIP_v2.pt',
                       help='CT-CLIP模型路径')
    parser.add_argument('--csv_path', type=str,
                       default='/mnt/blobdata/project/3DMedAgent/VQA/DeepTumorVQA_sampled-v3.csv',
                       help='数据CSV路径')
    parser.add_argument('--data_root', type=str,
                       default='/mnt/blobdata/data/DeepTumorVQA/data',
                       help='数据根目录')
    parser.add_argument('--output_dir', type=str,
                       default='/mnt/blobdata/data/DeepTumorVQA/Subset-v3/clip_embedding',
                       help='输出目录')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备: cuda 或 cpu')
    parser.add_argument('--use_fp16', action='store_true',
                       help='使用float16保存（节省空间）')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='最大处理样本数（默认：全部）')
    parser.add_argument('--verbose', action='store_true',
                       help='显示详细输出（默认：静默模式）')
    parser.add_argument('--overwrite', action='store_true',
                       help='覆盖已存在的embedding文件（默认：跳过已存在文件）')
    
    args = parser.parse_args()
    
    print("="*80)
    print("🚀 CT-CLIP Volume Embedding Extraction")
    print("="*80)
    print(f"模型路径: {args.model_path}")
    print(f"CSV路径: {args.csv_path}")
    print(f"数据根目录: {args.data_root}")
    print(f"输出目录: {args.output_dir}")
    print(f"设备: {args.device}")
    print(f"FP16模式: {args.use_fp16}")
    print(f"最大样本数: {args.max_samples if args.max_samples else '全部'}")
    print(f"详细输出: {args.verbose}")
    print("="*80)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. 初始化提取器
    extractor = CTCLIPEmbeddingExtractor(
        model_path=args.model_path,
        device=args.device,
        use_fp16=args.use_fp16
    )
    
    # 2. 加载CSV
    print(f"\n📄 加载CSV文件...")
    df = pd.read_csv(args.csv_path)
    print(f"   总样本数: {len(df)}")
    
    # 获取唯一的Image ID（避免重复处理）
    required_columns = {'Image ID', 'dataset'}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"CSV缺少必要列: {sorted(missing_columns)}")

    unique_images = df[['Image ID', 'dataset']].drop_duplicates()
    print(f"   唯一Image ID数: {len(unique_images)}")
    
    # 限制样本数（用于测试）
    if args.max_samples:
        unique_images = unique_images.head(args.max_samples)
        print(f"   限制处理: 前 {args.max_samples} 个样本")
    else:
        print(f"   处理模式: 全部 {len(unique_images)} 个样本")
    
    # 3. 批量处理
    print(f"\n⚡ 开始提取embeddings...")
    if not args.verbose:
        print(f"   (静默模式: 仅显示进度条)")
    
    results = []
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    for idx, (_, row) in enumerate(tqdm(unique_images.iterrows(), 
                                        total=len(unique_images),
                                        desc="Processing volumes")):
        image_id = row['Image ID']
        dataset = row['dataset']
        
        # 检查是否已存在
        save_path = os.path.join(args.output_dir, f'{image_id}.pt')
        if os.path.exists(save_path) and not args.overwrite:
            skipped_count += 1
            if args.verbose:
                print(f"⏭️  [{idx+1}/{len(unique_images)}] {image_id}: 已存在，跳过")
            continue
        
        # 构建NIfTI路径
        nii_path = os.path.join(args.data_root, dataset, 'img', f'{image_id}.nii.gz')
        
        # 处理单个volume
        result = extractor.process_single_volume(
            nii_path=nii_path,
            image_id=image_id,
            dataset=dataset,
            output_dir=args.output_dir
        )
        
        results.append(result)
        
        # 只在verbose模式下打印每个样本的进度
        if args.verbose:
            if result['status'] == 'success':
                print(f"✅ [{idx+1}/{len(unique_images)}] {image_id}: "
                      f"shape={result['embedding_shape']}, "
                      f"size={result['file_size_mb']:.2f}MB")
            else:
                print(f"❌ [{idx+1}/{len(unique_images)}] {image_id}: {result['error']}")
        
        # 更新计数
        if result['status'] == 'success':
            success_count += 1
        else:
            failed_count += 1
    
    # 4. 统计结果
    print("\n" + "="*80)
    print("📊 处理完成统计")
    print("="*80)
    
    print(f"✅ 成功: {success_count}/{len(unique_images)}")
    print(f"⏭️  跳过(已存在): {skipped_count}/{len(unique_images)}")
    print(f"❌ 失败: {failed_count}/{len(unique_images)}")
    
    if success_count > 0:
        avg_size = np.mean([r['file_size_mb'] for r in results if r['status'] == 'success'])
        total_size = sum([r['file_size_mb'] for r in results if r['status'] == 'success'])
        print(f"📦 平均文件大小: {avg_size:.2f} MB")
        print(f"💾 本次存储空间: {total_size:.2f} MB")
    
    # 5. 保存处理日志
    log_path = os.path.join(args.output_dir, 'extraction_log.json')
    with open(log_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': vars(args),
            'results': results,
            'summary': {
                'total_unique': len(unique_images),
                'processed': len(results),
                'success': success_count,
                'skipped': skipped_count,
                'failed': failed_count
            }
        }, f, indent=2)
    
    print(f"\n📝 处理日志已保存: {log_path}")
    print("="*80)
    
    # 6. 测试加载（验证保存的文件）
    if success_count > 0:
        print("\n🧪 验证：测试加载第一个保存的embedding...")
        first_success = next(r for r in results if r['status'] == 'success')
        test_path = first_success['save_path']
        
        try:
            embedding, metadata = load_embedding(test_path, device=args.device)
            expected_shape = (1, 24, 24, 24, 512)
            if tuple(embedding.shape) != expected_shape:
                raise ValueError(f"Unexpected embedding shape: {tuple(embedding.shape)} != {expected_shape}")
            print(f"✅ 加载成功!")
            print(f"   Shape: {embedding.shape}")
            print(f"   Dtype: {embedding.dtype}")
            print(f"   Device: {embedding.device}")
            print(f"   Metadata: {metadata}")
        except Exception as e:
            print(f"❌ 加载失败: {e}")


if __name__ == '__main__':
    main()
