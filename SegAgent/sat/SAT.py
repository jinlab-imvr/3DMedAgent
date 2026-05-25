# SegAgent/SegAgent.py
import os
import sys
import json
import uuid
import random
import datetime
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

# ==============================================================
# 导入 SAT 模块（单卡版本）
# ==============================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../SAT')))
from data.inference_dataset import Inference_Dataset, collate_fn
from model.build_model import build_maskformer_single, load_checkpoint
from model.text_encoder import Text_Encoder_Single
from evaluate.inference_engine import inference
from evaluate.params import parse_args


# ==============================================================
# Config
# ==============================================================
@dataclass
class SegAgentConfig:
    gpu_id: Optional[int] = None
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = False

    rcd_dir: str = "./result"
    tmp_dir: str = "./_tmp_jsonl"

    vision_backbone: str = "UNET-L"
    seg_checkpoint: str = ""
    max_queries: int = 256
    batchsize_3d: int = 1
    partial_load: bool = False

    text_encoder_name: str = "ours"
    text_encoder_checkpoint: str = ""
    text_encoder_partial_load: bool = False
    open_bert_layer: int = 12
    open_modality_embed: bool = False

    extra_env: Dict[str, str] = field(default_factory=dict)


# ==============================================================
# Seed Setup
# ==============================================================
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


# ==============================================================
# To solve the key mismatch issue when loading DDP checkpoints
# ==============================================================
def clean_state_dict(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    new_state_dict = {}
    for k, v in state_dict.items():
        # remove DDP prefix
        if k.startswith("module."):
            k = k[len("module."):]
        elif k.startswith("_module."):
            k = k[len("_module."):]
        new_state_dict[k] = v
    return new_state_dict


# ==============================================================
# SegAgent Class
# ==============================================================
class SegAgent:
    def __init__(self, cfg: SegAgentConfig):
        self.cfg = cfg
        set_seed(cfg.seed)

        # environment setup
        if cfg.gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
        for k, v in cfg.extra_env.items():
            os.environ[str(k)] = str(v)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SegAgent] Using device: {self.device}")

        # 输出目录
        Path(cfg.rcd_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.tmp_dir).mkdir(parents=True, exist_ok=True)

        # 初始化参数（保持 inference.py 一致）
        args = parse_args()

        args.vision_backbone = cfg.vision_backbone
        args.max_queries = cfg.max_queries
        args.batchsize_3d = cfg.batchsize_3d
        args.pin_memory = cfg.pin_memory
        args.num_workers = cfg.num_workers
        args.partial_load = cfg.partial_load
        self.args = args

        # ===== 构建模型 =====
        self.model = build_maskformer_single(self.args, self.device)

        # ===== 构建文本编码器 =====
        self.text_encoder = Text_Encoder_Single(
            text_encoder=cfg.text_encoder_name,
            checkpoint=None,  
            partial_load=cfg.text_encoder_partial_load,
            open_bert_layer=cfg.open_bert_layer,
            open_modality_embed=cfg.open_modality_embed,
            device=self.device
        )

        # ===== 加载 segmentation checkpoint =====
        if cfg.seg_checkpoint and os.path.exists(cfg.seg_checkpoint):
            print(f"[SegAgent] Loading segmentation checkpoint: {cfg.seg_checkpoint}")
            try:
                ckpt = torch.load(cfg.seg_checkpoint, map_location=self.device, weights_only=False)
                state_dict = ckpt.get("model_state_dict", ckpt)
                state_dict = clean_state_dict(state_dict)
                missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
                print(f"[SegAgent] Segmentation model loaded ({len(missing)} missing, {len(unexpected)} unexpected keys).")
                print(f"[SegAgent] Missing keys: {list(missing)}")
                print(f"[SegAgent] Unexpected keys: {list(unexpected)}")

            except Exception as e:
                print(f"[SegAgent] Warning: fallback to load_checkpoint due to {e}")
                self.model, _, _ = load_checkpoint(
                    checkpoint=cfg.seg_checkpoint,
                    resume=False,
                    partial_load=cfg.partial_load,
                    model=self.model,
                    device=self.device
                )

        # ===== 加载 text encoder checkpoint =====
        if cfg.text_encoder_checkpoint and os.path.exists(cfg.text_encoder_checkpoint):
            print(f"[SegAgent] Loading text encoder checkpoint: {cfg.text_encoder_checkpoint}")
            try:
                ckpt = torch.load(cfg.text_encoder_checkpoint, map_location=self.device, weights_only=False)
                state_dict = ckpt.get("model_state_dict", ckpt)
                state_dict = clean_state_dict(state_dict)

                new_state_dict = {}
                for k, v in state_dict.items():
                    if not k.startswith("model.") and ("text_tower" in k or "projection_layer" in k or "modality_embed" in k):
                        k = "model." + k
                    new_state_dict[k] = v

                missing, unexpected = self.text_encoder.load_state_dict(new_state_dict, strict=False)
                print(f"[SegAgent] Text encoder loaded ({len(missing)} missing, {len(unexpected)} unexpected keys).")
                print(f"[SegAgent] Missing keys: {list(missing)}")
                print(f"[SegAgent] Unexpected keys: {list(unexpected)}")

            except Exception as e:
                print(f"[SegAgent] Warning: fallback to load_checkpoint due to {e}")
                self.text_encoder, _, _ = load_checkpoint(
                    checkpoint=cfg.text_encoder_checkpoint,
                    resume=False,
                    partial_load=cfg.text_encoder_partial_load,
                    model=self.text_encoder,
                    device=self.device
                )

    # ==============================================================
    # The inference function
    # ==============================================================
    def infer_from_jsonl(self, jsonl_path: str, out_dir: Optional[str] = None) -> str:
        out_root = out_dir or self.cfg.rcd_dir
        Path(out_root).mkdir(parents=True, exist_ok=True)

        testset = Inference_Dataset(jsonl_path, self.cfg.max_queries, self.cfg.batchsize_3d)
        testloader = DataLoader(
            testset,
            batch_size=1,
            pin_memory=self.cfg.pin_memory,
            num_workers=self.cfg.num_workers,
            collate_fn=collate_fn
        )

        inference(
            model=self.model,
            text_encoder=self.text_encoder,
            device=self.device,
            testset=testset,
            testloader=testloader,
            nib_dir=out_root
        )
        return out_root

    # ==============================================================
    # inference on a single image
    # ==============================================================
    def infer_one(self, image_path: str, labels: List[str],
                  modality: str = "ct", dataset_name: str = "AbdomenCT1K",
                  out_dir: Optional[str] = None, case_id: Optional[str] = None) -> str:
        assert Path(image_path).exists(), f"Image not found: {image_path}"
        out_root = out_dir or self.cfg.rcd_dir
        Path(out_root).mkdir(parents=True, exist_ok=True)

        case_id = case_id or Path(image_path).stem
        tmp_jsonl = Path(self.cfg.tmp_dir) / f"{uuid.uuid4().hex}.jsonl"

        with open(tmp_jsonl, "w") as f:
            json.dump({
                "image": image_path,
                "label": labels,
                "modality": modality.lower(),
                "dataset": dataset_name
            }, f, ensure_ascii=False)
            f.write("\n")

        try:
            self.infer_from_jsonl(str(tmp_jsonl), out_dir=out_root)
        finally:
            tmp_jsonl.unlink(missing_ok=True)

        final_dir = Path(out_root) / dataset_name / f"seg_{case_id}"
        return str(final_dir)


if __name__ == "__main__":
    cfg = SegAgentConfig(
        gpu_id=2,
        rcd_dir=YOUR_DATA_DIR,
        vision_backbone="UNET-L",
        seg_checkpoint=YOUR_DATA_DIR,
        text_encoder_name="ours",
        text_encoder_checkpoint=YOUR_DATA_DIR,
        max_queries=256,
        batchsize_3d=1,
        extra_env={"RANK": "0", "WORLD_SIZE": "1"}
    )

    agent = SegAgent(cfg)
    final_dir = agent.infer_one(
        image_path=YOUR_DATA_DIR,
        labels=["liver", "kidney", "spleen", "heart"],
        modality="ct",
        dataset_name="AbdomenCT1K",
        case_id="valid_861_a_1"
    )
    print("Output saved to:", final_dir)
