# utils/io.py

import os
import numpy as np
import nibabel as nib
import scipy.ndimage as ndimage


def load_canonical(src):
    """
    Load a NIfTI file, convert to RAS canonical orientation,
    and enforce data axis order so that axis=2 corresponds to the slice (S) axis.

    Returns:
        nib.Nifti1Image with data oriented to RAS and consistent axis semantics.
    """
    if isinstance(src, (str, os.PathLike)):
        img = nib.load(src)
    elif isinstance(src, nib.spatialimages.SpatialImage):
        img = src
    else:
        raise TypeError("Expected path or NIfTI image.")

    # 1) Bring image close to RAS (this may change affine and/or data orientation)
    img = nib.as_closest_canonical(img)

    # 2) Explicitly apply orientation transform to guarantee axis semantics
    #    (R, A, S) with axis order aligned to (X, Y, Z)
    cur_ornt = nib.orientations.io_orientation(img.affine)
    ras_ornt = nib.orientations.axcodes2ornt(("R", "A", "S"))
    transform = nib.orientations.ornt_transform(cur_ornt, ras_ornt)

    data = img.get_fdata()
    data_ras = nib.orientations.apply_orientation(data, transform)

    # 3) Update affine to match reoriented data
    # inv_ornt_aff maps from old voxel coords to new voxel coords; we compose it.
    inv_aff = nib.orientations.inv_ornt_aff(transform, img.shape)
    new_affine = img.affine @ inv_aff

    out = nib.Nifti1Image(data_ras, new_affine, header=img.header)

    # 4) Keep header consistent
    out.set_sform(new_affine, code=1)
    out.set_qform(new_affine, code=1)

    return out


def get_orientation_transform(img, orientation=('L', 'P', 'S')):
    """
    Compute transform to reorient NIfTI image to target orientation.
    """
    current_ornt = nib.orientations.io_orientation(img.affine)
    desired_ornt = nib.orientations.axcodes2ornt(orientation)
    return nib.orientations.ornt_transform(current_ornt, desired_ornt)


def apply_transform(data, transform):
    """
    Apply an orientation transform to raw numpy voxel data.
    """
    return nib.orientations.apply_orientation(data, transform)


def resample_image(image, original_spacing, target_spacing=(1, 1, 1), order=1):
    """
    Resample a 3D image array to new voxel spacing.
    Input may be ndarray or Nifti image.
    """
    try:
        image = image.get_fdata()
    except:
        pass

    resize_factor = np.array(original_spacing) / np.array(target_spacing)
    resampled = ndimage.zoom(image, resize_factor, order=order)

    return resampled, resize_factor

def load_segments_liver(segments_path, spacing):
    """
    从 liver 子段掩码目录中加载所有 liver_segment*.nii.gz，
    合成一个多标签 joint 体积，然后重采样到 1mm^3。

    逻辑严格对应原始 CreateAAReports.py 里的 load_segments_liver：
      - 对每个 liver_segment*.nii.gz 做二值化
      - 避免重叠：如果 joint+seg 出现新的 max，就在已有 joint>0 的位置把 seg 置 0
      - 用文件名最后一位数字作为 segment ID（1~8）
      - 最后整体重采样到 1mm isotropic
    """
    joint = None

    for segment in os.listdir(segments_path):
        if 'liver_segment' not in segment:
            continue

        seg_path = os.path.join(segments_path, segment)
        seg = load_canonical(seg_path).get_fdata()
        seg = np.where(seg > 0.5, 1, 0)

        if joint is None:
            joint = seg
        else:
            # 避免段之间 overlap：若 joint+seg 的 max 变大，说明有重叠
            if joint.max() != (joint + seg).max():
                # 把 joint 已经为 1 的地方从 seg 里清掉
                seg = np.where(joint > 0, 0, seg)

            # 用文件名最后一个数字作为 segment id（和原代码保持一致）
            seg_id = int(seg_path[-1 - len('.nii.gz')])
            joint += seg * seg_id

    if joint is None:
        return None

    joint_resampled, _ = resample_image(
        joint,
        original_spacing=spacing,
        target_spacing=(1, 1, 1),
        order=0
    )
    return joint_resampled

def load_segments_pancreas(segments_path,spacing):
    #loads head, body and tail of the pancreas
    joint=None
    
    for i,segment in enumerate(['pancreas_head.nii.gz','pancreas_body.nii.gz','pancreas_tail.nii.gz'],1):
        #print('Segment path:', segment,segments_path)
        seg_path = os.path.join(segments_path, segment)
        seg=load_canonical(seg_path).get_fdata()
        seg = np.where(seg > 0.5, 1, 0)
        if joint is None:
            joint=seg
        else:
            joint+=seg*i
            #threshold to i
            joint=np.where(joint > i, i, joint)

    joint, _ = resample_image(joint,original_spacing=spacing,
                             target_spacing=(1, 1, 1),order=0)
    #print('loaded pancreas segments from:',segments_path)
    return joint

def load_segments_kidney(segments_path, spacing):
    joint = None

    # 统一定义左右肾的候选文件名
    kidney_candidates = {
        1: ["kidney_left.nii.gz", "left kidney.nii.gz"],
        2: ["kidney_right.nii.gz", "right kidney.nii.gz"],
    }

    for label, name_list in kidney_candidates.items():
        seg = None

        # 在候选命名中找到第一个存在的
        for name in name_list:
            seg_path = os.path.join(segments_path, name)
            if os.path.exists(seg_path):
                seg = load_canonical(seg_path).get_fdata()
                break

        # 如果这一侧肾脏不存在，直接跳过
        if seg is None:
            continue

        seg = np.where(seg > 0.5, 1, 0)
        if joint is None:
            joint = seg * label
        else:
            joint = joint + seg * label
            # 防止重叠区域被错误累加
            joint = np.where(joint > label, label, joint)

    joint, _ = resample_image(
        joint,
        original_spacing=spacing,
        target_spacing=(1, 1, 1),
        order=0,
    )

    return joint