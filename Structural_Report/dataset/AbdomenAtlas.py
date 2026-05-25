import os


def list_cases(ct_folder, mask_folder):
    """
    根据 ct_folder 和 mask_folder 共同的子目录名，得到要处理的 case 列表。
    """
    ct_cases = {
        d for d in os.listdir(ct_folder)
        if os.path.isdir(os.path.join(ct_folder, d))
    }
    mask_cases = {
        d for d in os.listdir(mask_folder)
        if os.path.isdir(os.path.join(mask_folder, d))
    }
    return sorted(ct_cases & mask_cases)

def get_ct_path(ct_folder, case_id):
    return os.path.join(ct_folder, case_id, 'ct.nii.gz')
# -------- 各器官的 mask 寻址逻辑（保持原 main.py 行为不变） -------- #

def get_spleen_mask_path(mask_folder, case_id):
    primary = os.path.join(mask_folder, case_id, 'segmentations', 'spleen.nii.gz')
    alt = os.path.join(mask_folder, case_id, 'segmentations', '_spleen.nii.gz')
    if os.path.isfile(primary):
        return primary
    if os.path.isfile(alt):
        return alt
    return None


def get_liver_mask_path(mask_folder, case_id):
    primary = os.path.join(mask_folder, case_id, 'segmentations', 'liver.nii.gz')
    alt = os.path.join(mask_folder, case_id, 'segmentations', '_liver_.nii.gz')
    if os.path.isfile(primary):
        return primary
    if os.path.isfile(alt):
        return alt
    return None


def get_liver_related_paths(mask_folder, case_id):
    """
    返回:
      liver_segments_dir, liver_tumor, liver_cyst, liver_lesion
    """
    seg_dir = os.path.join(mask_folder, case_id, 'segmentations')
    liver_segments_dir = None
    liver_tumor = None
    liver_cyst = None
    liver_lesion = None

    if os.path.isdir(seg_dir):
        files = os.listdir(seg_dir)
        if any('liver_segment' in f for f in files):
            liver_segments_dir = seg_dir

        for prefix in ['liver', 'hepatic']:
            # 注意这里保持你当前 main.py 的命名：有空格 / 下划线混用
            t = os.path.join(seg_dir, f'{prefix} tumor.nii.gz')
            c = os.path.join(seg_dir, f'{prefix}_cyst.nii.gz')
            l = os.path.join(seg_dir, f'{prefix}_lesion.nii.gz')
            if liver_tumor is None and os.path.isfile(t):
                liver_tumor = t
            if liver_cyst is None and os.path.isfile(c):
                liver_cyst = c
            if liver_lesion is None and os.path.isfile(l):
                liver_lesion = l

    return liver_segments_dir, liver_tumor, liver_cyst, liver_lesion


def get_kidney_organ_paths(mask_folder, case_id):
    """
    返回: right_kidney_path, left_kidney_path
    """
    seg_dir = os.path.join(mask_folder, case_id, 'segmentations')
    right1 = os.path.join(seg_dir, 'kidney_right.nii.gz')
    right2 = os.path.join(seg_dir, '_kidney_right.nii.gz')
    right3 = os.path.join(seg_dir, 'right kidney.nii.gz')

    right = None
    for p in [right1, right2, right3]:
        if os.path.isfile(p):
            right = p
            break

    left = None
    if right is not None:
        if 'kidney_right' in right:
            candidate = right.replace('kidney_right', 'kidney_left')
        elif 'right kidney' in right:
            candidate = right.replace('right kidney', 'left kidney')
        else:
            candidate = None
        if candidate is not None and os.path.isfile(candidate):
            left = candidate

    return right, left


def get_kidney_lesion_paths(mask_folder, case_id):
    """
    返回: tumor, cyst, lesion
    """
    seg_dir = os.path.join(mask_folder, case_id, 'segmentations')
    tumor = cyst = lesion = None

    for prefix in ['kidney', 'renal']:
        t = os.path.join(seg_dir, f'{prefix} tumor.nii.gz')
        c = os.path.join(seg_dir, f'{prefix} cyst.nii.gz')
        l = os.path.join(seg_dir, f'{prefix} lesion.nii.gz')
        if tumor is None and os.path.isfile(t):
            tumor = t
        if cyst is None and os.path.isfile(c):
            cyst = c
        if lesion is None and os.path.isfile(l):
            lesion = l

    return tumor, cyst, lesion


def get_pancreas_organ_path(mask_folder, case_id):
    seg_dir = os.path.join(mask_folder, case_id, 'segmentations')
    primary = os.path.join(seg_dir, 'pancreas.nii.gz')
    alt = os.path.join(seg_dir, '_pancreas_.nii.gz')
    if os.path.isfile(primary):
        return primary
    if os.path.isfile(alt):
        return alt
    return None


def get_pancreas_segments_dir(mask_folder, case_id):
    seg_dir = os.path.join(mask_folder, case_id, 'segmentations')
    if not os.path.isdir(seg_dir):
        return None
    files = os.listdir(seg_dir)
    if any(f in files for f in ['pancreas_head.nii.gz', 'pancreas_body.nii.gz', 'pancreas_tail.nii.gz']):
        return seg_dir
    return None


def get_pancreas_lesion_paths(mask_folder, case_id):
    """
    返回: pdac, pnet, tumor, cyst, lesion
    """
    seg_dir = os.path.join(mask_folder, case_id, 'segmentations')
    pdac = pnet = tumor = cyst = lesion = None

    for prefix in ['pancreas', 'pancreatic']:
        pd = os.path.join(seg_dir, f'{prefix}_pdac.nii.gz')
        pn = os.path.join(seg_dir, f'{prefix}_pnet.nii.gz')
        t = os.path.join(seg_dir, f'{prefix} tumor.nii.gz')
        c = os.path.join(seg_dir, f'{prefix}_cyst.nii.gz')
        l = os.path.join(seg_dir, f'{prefix}_lesion.nii.gz')

        if pdac is None and os.path.isfile(pd):
            pdac = pd
        if pnet is None and os.path.isfile(pn):
            pnet = pn
        if tumor is None and os.path.isfile(t):
            tumor = t
        if cyst is None and os.path.isfile(c):
            cyst = c
        if lesion is None and os.path.isfile(l):
            lesion = l

    return pdac, pnet, tumor, cyst, lesion


def get_colon_paths(mask_folder, case_id):
    """
    返回: colon_organ, colon_lesion
    """
    seg_dir = os.path.join(mask_folder, case_id, 'segmentations')
    organ = os.path.join(seg_dir, 'colon.nii.gz')
    lesion = os.path.join(seg_dir, 'colon_lesion.nii.gz')
    if not os.path.isfile(organ):
        organ = None
    if not os.path.isfile(lesion):
        lesion = None
    return organ, lesion
