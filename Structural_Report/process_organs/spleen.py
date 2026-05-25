# process_organs/spleen.py

import numpy as np

from utils.io import load_canonical, resample_image
from utils.image_process import measure_volume, measure_organ_hu


def generate_spleen_report(ct, spleen_mask, spacing, phase=None):
    """
    只针对脾脏的 report 逻辑，等价于原始 organ_text 中 clss=='spleen' 的分支。
    返回:
        text: 字符串报告
        vol: 体积（mm^3，可能为 None）
        organ_hu: 平均 HU
        organ_hu_std: HU 标准差
    """

    # 计算体积：原代码是 measure_volume(organ, spacing=spacing, check_border=True)
    vol = measure_volume(spleen_mask, spacing=spacing, check_border=False)

    # HU 统计：原代码是 measure_organ_hu(organ, 0*organ, ct)
    organ_hu, organ_hu_std = measure_organ_hu(spleen_mask, 0 * spleen_mask, ct)

    # 判定大小（完全照原逻辑）
    size = None
    if vol is not None:
        size = 'normal'
        # Taylor A, Dodds W, Erickson S, Stewart E. CT of Acquired Abnormalities
        # of the Spleen. AJR Am J Roentgenol. 1991;157(6):1213-9.
        if vol / 1000 > 314.5:
            size = 'large'
        if vol / 1000 > 430.8:
            size = 'massive'

    text = ''
    text += "Spleen: \n"

    if vol is not None:
        if size == 'normal':
            text += 'Normal size '
        elif size == 'large':
            text += 'Spleen is enlarged '
        elif size == 'massive':
            text += 'Spleen is massively enlarged '
        text += f"(volume: {np.round(vol / 1000, 1)} cm^3).\n"

    # phase 对 spleen 的影响在原 organ_text 里其实走的是 “else: Mean HU ...” 分支
    text += f"Mean HU value: {np.round(organ_hu, 1)} +/- {np.round(organ_hu_std, 1)}.\n"

    return text, vol, organ_hu, organ_hu_std


def process_spleen_case(ct_path, spleen_mask_path, phase=None):
    """
    单个 case 的脾脏处理入口。
    参数:
        ct_path: 原始 CT 的完整路径（.../ct.nii.gz）
        spleen_mask_path: 脾脏分割的完整路径（.../spleen.nii.gz 或 _spleen.nii.gz）
        phase: 可选，和原逻辑保持一致，默认为 None
    返回:
        spleen_report_text: 该 case 的脾脏报告字符串
    """

    print(f"Processing spleen case:")
    print(f"  CT:    {ct_path}")
    print(f"  Mask:  {spleen_mask_path}")

    # 1. 读取脾脏 mask
    spleen = load_canonical(spleen_mask_path).get_fdata().astype('uint8')

    # 2. 读取 CT
    ct_img = load_canonical(ct_path)
    spacing = ct_img.header.get_zooms()
    ct = ct_img.get_fdata()

    # For RadGenome-ChestCT dataset
    # spacing = (1.0, 1.0, 3.0)

    # 3. 重采样到 1mm^3（和原 get_paths 中 spleen 分支完全一致）
    spleen, _ = resample_image(spleen, original_spacing=spacing,
                               target_spacing=(1, 1, 1), order=0)
    ct, _ = resample_image(ct, original_spacing=spacing,
                           target_spacing=(1, 1, 1))
    spleen = spleen.astype('float32')

    # 4. 生成报告
    text, vol, organ_hu, organ_hu_std = generate_spleen_report(
        ct=ct,
        spleen_mask=spleen,
        spacing=spacing,
        phase=phase,
    )

    return text
