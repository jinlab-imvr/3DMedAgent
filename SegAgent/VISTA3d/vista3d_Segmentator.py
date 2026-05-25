import os
import sys
import torch
import numpy as np
import nibabel as nib

# 指向真正的 Vista3D 仓库根目录（包含 scripts/ 和 vista3d/）
VISTA3D_ROOT = "/mnt/blobdata/project/3DMedAgent/VISTA/vista3d"

if VISTA3D_ROOT not in sys.path:
    sys.path.insert(0, VISTA3D_ROOT)

from scripts.infer import InferClass

class Vista3D_Segmentator:
    """
    Python wrapper for Vista3D inference.
    Interface intentionally aligned with BiomedParse_Segmentator.
    """

    def __init__(
        self,
        config_file: str,
        device: str | torch.device = "cuda:0",
    ):
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # Vista3D 的 InferClass 本身已经封装好模型 / transforms / 保存逻辑
        self.infer_engine = InferClass(
            config_file=config_file
        )

        print("[Vista3D_Segmentator] Initialized")
        print(f"  Config: {config_file}")
        print(f"  Device: {self.device}")

    # --------- 对外接口：分割 ---------
    @torch.no_grad()
    def segment(
        self,
        input_path: str,
        output_path: str,
        object_list: list[int] | None = None,
        save_mask: bool = True,
    ):
        """
        Parameters
        ----------
        input_path : str
            Path to input NIfTI image
        output_path : str
            Desired output NIfTI path
        object_list : list[int] | None
            Vista3D label_prompt, e.g. [1] or [1, 2, 3]
            None means infer everything (NOT recommended for speed)
        save_mask : bool
            Whether to save segmentation to disk

        Returns
        -------
        mask : np.ndarray
            Segmentation mask (H, W, D)
        """
        self.infer_engine.clear_cache()
        if not os.path.isfile(input_path):
            raise FileNotFoundError(input_path)

        # Vista3D 用的是 label_prompt（int list）
        label_prompt = object_list

        # InferClass.infer 会自动保存到 infer.yaml 里配置的 output_path
        pred = self.infer_engine.infer(
            image_file=input_path,
            label_prompt=label_prompt,
            save_mask=save_mask,
        )

        # pred 是 MetaTensor (C, H, W, D) 或 (1, H, W, D)
        pred_np = pred.cpu().numpy()

        # 通常 Vista3D 输出是 (1, H, W, D)
        if pred_np.ndim == 4:
            pred_np = pred_np[0]

        # 如果你想强制把结果保存到指定 output_path（覆盖默认）
        if save_mask:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            affine = nib.load(input_path).affine
            nib.save(
                nib.Nifti1Image(pred_np.astype(np.uint8), affine),
                output_path,
            )

        return pred_np


# --------- Example usage ---------
def main():
    segmentator = Vista3D_Segmentator(
        config_file="configs/infer.yaml",
        device="cuda:0",
    )

    input_path = "/mnt/blobdata/data/DeepTumorVQA/data/autoPET_PETCT/img/BDMAP_00000010.nii.gz"
    output_path = "/mnt/blobdata/code/3DMedAgent/Test_Seg/vista3d_pancreas.nii.gz"

    mask = segmentator.segment(
        input_path=input_path,
        output_path=output_path,
        object_list=[1,3,4,5,14,62],  # Vista3D label id
    )
    print(np.unique(mask,return_counts=True))
    print("Vista3D segmentation finished:", mask.shape)


if __name__ == "__main__":
    main()
