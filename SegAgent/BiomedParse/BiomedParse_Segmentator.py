import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

import torch
import torch.nn.functional as F
import hydra
from hydra import compose
from hydra.core.global_hydra import GlobalHydra
from skimage import segmentation
from skimage.measure import label
import nibabel as nib

from .utils import process_input, process_output, slice_nms
from .inference import postprocess, merge_multiclass_masks


# BiomedParse 预训练时用的默认器官列表（按你给的示例）
DEFAULT_TEXT_PROMPTS = {
    "1": "liver",
    "2": "kidney",
    "3": "spleen",
    "4": "pancreas",
    '5': 'colon',
    "6": "stomach",
    "7": "heart",
    "8": "left lung",
    "9": "right lung",
    "10": "lung lesion",
    "11": "liver tumor",
    "12": "left kidney",
    "13": "right kidney",
}


class BiomedParse_Segmentator:
    def __init__(
        self,
        config_dir: str | None = None,
        config_name: str = "biomedparse_3D",
        checkpoint_path: str | None = None,
        device: torch.device | str | None = None,
    ):
        # 1. 设备设定
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.config_dir = config_dir
        self.config_name = config_name
        self.checkpoint_path = checkpoint_path

        # 2. 初始化 Hydra（避免重复初始化报错，先 clear）
        GlobalHydra.instance().clear()
        hydra.initialize(config_path=self.config_dir, job_name="biomedparse_segmentator")

        # 3. 载入配置与模型
        cfg = compose(config_name=self.config_name)
        model = hydra.utils.instantiate(cfg, _convert_="object")

        # 4. 加载预训练权重并放到设备上
        model.load_pretrained(self.checkpoint_path)
        self.model = model.to(self.device).eval()

        print(f"[BiomedParse_Segmentator] Using device: {self.device}")
        print(f"[BiomedParse_Segmentator] Config: {self.config_dir}/{self.config_name}.yaml")
        print(f"[BiomedParse_Segmentator] Checkpoint: {self.checkpoint_path}")

    # --------- 内部工具函数 ---------
    def _load_and_preprocess(
        self,
        input_path: str,
        norm_range: tuple[float, float] | None = (0.0, 255.0),
    ):
        nifti = nib.load(input_path)
        data = nifti.get_fdata()  # 原始一般是 (H, W, D)
        affine = nifti.affine

        # 保证 float32 精度
        data = data.astype(np.float32)

        # 把 depth 放到第一维: (H, W, D) -> (D, H, W)
        if data.ndim == 3:
            data = data.transpose(2, 0, 1)
        else:
            raise ValueError(f"Expected 3D volume, got shape {data.shape}")

        # 先归一化到 [0,1]
        v_min = data.min()
        v_max = data.max()
        if v_max > v_min:
            data = (data - v_min) / (v_max - v_min)
        else:
            data = np.zeros_like(data, dtype=np.float32)

        # 再拉伸到指定范围
        if norm_range is not None:
            lo, hi = norm_range
            data = data * (hi - lo) + lo

        # 如果范围是 0-255，就直接转成 uint8（和你原来代码一致）
        if norm_range == (0.0, 255.0) or norm_range == (0, 255):
            data = data.astype(np.uint8)

        return data, affine

    def _build_text_prompts(self, object_list: list[str] | None):
        if not object_list:
            selected_keys = sorted(DEFAULT_TEXT_PROMPTS.keys(), key=lambda x: int(x))
        else:
            selected_keys = []
            for k, v in DEFAULT_TEXT_PROMPTS.items():
                if v in object_list:
                    selected_keys.append(k)
            selected_keys = sorted(selected_keys, key=lambda x: int(x))

            if len(selected_keys) == 0:
                raise ValueError(
                    f"None of the requested objects {object_list} "
                    f"found in DEFAULT_TEXT_PROMPTS: {list(DEFAULT_TEXT_PROMPTS.values())}"
                )

        ids = [int(k) for k in selected_keys]
        text = "[SEP]".join([DEFAULT_TEXT_PROMPTS[k] for k in selected_keys])
        return ids, text

    # --------- 对外接口：分割 ---------
    def segment(
        self,
        input_path: str,
        output_path: str,
        object_list: list[str] | None = None,
        norm_range: tuple[float, float] | None = (0.0, 255.0),
        slice_batch_size: int = 4,
    ):
        # 1. 加载 & 预处理体数据
        data, affine = self._load_and_preprocess(input_path, norm_range=norm_range)
        print("Loaded image shape (D, H, W):", data.shape)

        # 2. 构造 text-prompts
        ids, text = self._build_text_prompts(object_list)
        print("Target objects:", [DEFAULT_TEXT_PROMPTS[str(i)] for i in ids])
        print("Text prompts:", text)

        # 3. 利用 BiomedParse 原始 pipeline 做 slice 级推理
        imgs, pad_width, padded_size, valid_axis = process_input(data, 512)
        imgs = imgs.to(self.device).int()

        input_tensor = {
            "image": imgs.unsqueeze(0),  # Add batch dimension
            "text": [text],
        }

        with torch.no_grad():
            output = self.model(input_tensor, mode="eval", slice_batch_size=slice_batch_size)

        mask_preds = output["predictions"]["pred_gmasks"]
        mask_preds = F.interpolate(
            mask_preds,
            size=(512, 512),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )

        mask_preds = postprocess(mask_preds, output["predictions"]["object_existence"])
        mask_preds = merge_multiclass_masks(mask_preds, ids)
        mask_preds = process_output(mask_preds, pad_width, padded_size, valid_axis)

        print("Processed mask shape (D, H, W):", mask_preds.shape)

        # 4. 保存为 NIfTI（注意转回 (H, W, D)）
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        mask_to_save = mask_preds.transpose(1, 2, 0).astype(np.uint8)
        nib.save(nib.Nifti1Image(mask_to_save, affine), output_path)
        print(f"Saved segmentation mask to: {output_path}")

        return mask_to_save

# --------- Example usage / quick test ---------
def main():
    # 1. 初始化 segmentator
    segmentator = BiomedParse_Segmentator(
        config_dir="configs/model",
        checkpoint_path="/mnt/blobdata/code/3DMedAgent/BiomedParse/checkpoints/biomedparse_v2.ckpt",
    )

    # 2. 跑一遍真实分割
    input_path = "/mnt/blobdata/code/3DMedAgent/RAG/downloads/lung-13/CT_3D/Axial_2.nii.gz"
    output_path = "/mnt/blobdata/code/3DMedAgent/Test_Seg/Report/test_mask/renal2/segmentations/pancreas.nii.gz"

    # 想分割哪些器官就写哪些；None 表示默认全器官
    object_list = ["liver","pancreas","left kidney"]

    mask = segmentator.segment(
        input_path=input_path,
        output_path=output_path,
        object_list=object_list,
        norm_range=(0.0, 255.0),  # 如果以后想换成 (-1024, 3071) 也可以在这里改
    )
    print("BiomedParse_Segmentator finished, final mask shape:", mask.shape)

if __name__ == "__main__":
    main()
