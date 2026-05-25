# main.py

import os
import csv
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
from contextlib import redirect_stdout

from utils.image_process import compute_ct_metaHU
from process_organs.spleen import process_spleen_case
from process_organs.liver import process_liver_case
from process_organs.kidney import process_kidney_case
from process_organs.pancreas import process_pancreas_case
from process_organs.colon import process_colon_case

from dataset.RadGenome import (
    list_cases,
    get_ct_path,
    get_spleen_mask_path,
    get_liver_mask_path,
    get_liver_related_paths,
    get_kidney_organ_paths,
    get_kidney_lesion_paths,
    get_pancreas_organ_path,
    get_pancreas_segments_dir,
    get_pancreas_lesion_paths,
    get_colon_paths,
)

def generate_reports(
    ct_folder: str,
    mask_folder: str,
    csv_file: str = "organ_reports.csv",
    restart_csv: bool = False,
):
    cases = list_cases(ct_folder, mask_folder)

    processed = set()
    if restart_csv or (not os.path.isfile(csv_file)):
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Case", "Report"])
    else:
        with open(csv_file, "r", encoding="utf-8") as f:
            lines = f.readlines()[1:]
        for line in lines:
            cid = line.split(",")[0]
            processed.add(cid)

    for case_id in cases:
        if case_id in processed:
            continue

        ct_path = get_ct_path(ct_folder, case_id)
        if not os.path.isfile(ct_path):
            continue

        organ_reports = []

        # ============================
        # 0) CT-level HU meta (NEW)
        # ============================
        meta_report = None
        try:
            meta_report = compute_ct_metaHU(ct_path)
            if isinstance(meta_report, str):
                meta_report = meta_report.strip()
        except Exception:
            meta_report = None

        # ============================
        # 1) Spleen
        # ============================
        spleen_mask = get_spleen_mask_path(mask_folder, case_id)
        if spleen_mask is not None:
            try:
                spleen_report = process_spleen_case(ct_path, spleen_mask)
                organ_reports.append(spleen_report.strip())
            except Exception:
                pass

        # ============================
        # 2) Liver
        # ============================
        liver_mask = get_liver_mask_path(mask_folder, case_id)
        if liver_mask is not None:
            liver_segments_dir, liver_tumor, liver_cyst, liver_lesion = \
                get_liver_related_paths(mask_folder, case_id)
            try:
                liver_report = process_liver_case(
                    ct_path=ct_path,
                    liver_mask_path=liver_mask,
                    liver_segments_dir=liver_segments_dir,
                    tumor_mask_path=liver_tumor,
                    cyst_mask_path=liver_cyst,
                    lesion_mask_path=liver_lesion,
                    phase=None,
                    spleen_hu=None,
                )
                organ_reports.append(liver_report.strip())
            except Exception:
                pass

        # ============================
        # 3) Pancreas
        # ============================
        pancreas_mask = get_pancreas_organ_path(mask_folder, case_id)
        if pancreas_mask is not None:
            pancreas_seg_dir = get_pancreas_segments_dir(mask_folder, case_id)
            pdac, pnet, pan_tumor, pan_cyst, pan_lesion = \
                get_pancreas_lesion_paths(mask_folder, case_id)
            try:
                pancreas_report = process_pancreas_case(
                    ct_path=ct_path,
                    pancreas_mask_path=pancreas_mask,
                    pancreas_segments_dir=pancreas_seg_dir,
                    pdac_mask_path=pdac,
                    pnet_mask_path=pnet,
                    tumor_mask_path=pan_tumor,
                    cyst_mask_path=pan_cyst,
                    lesion_mask_path=pan_lesion,
                    phase=None,
                    spleen_hu=None,
                )
                organ_reports.append(pancreas_report.strip())
            except Exception:
                pass

        # ============================
        # 4) Kidney
        # ============================
        right_kid, left_kid = get_kidney_organ_paths(mask_folder, case_id)
        if right_kid is not None and left_kid is not None:
            k_tumor, k_cyst, k_lesion = get_kidney_lesion_paths(mask_folder, case_id)
            try:
                kidney_report = process_kidney_case(
                    ct_path=ct_path,
                    kidney_right_mask_path=right_kid,
                    kidney_left_mask_path=left_kid,
                    kidney_segments_dir=None,
                    tumor_mask_path=k_tumor,
                    cyst_mask_path=k_cyst,
                    lesion_mask_path=k_lesion,
                )
                organ_reports.append(kidney_report.strip())
            except Exception:
                pass

        # ============================
        # 5) Colon
        # ============================
        colon_mask, colon_lesion = get_colon_paths(mask_folder, case_id)
        if colon_mask is not None:
            try:
                colon_report = process_colon_case(
                    ct_path=ct_path,
                    colon_mask_path=colon_mask,
                    colon_lesion_mask_path=colon_lesion,
                )
                organ_reports.append(colon_report.strip())
            except Exception:
                pass

        if not organ_reports and meta_report is None:
            continue

        # ============================
        # Final report assembly
        # ============================
        report_blocks = []

        if meta_report is not None:
            report_blocks.append(meta_report)

        if organ_reports:
            report_blocks.extend(organ_reports)

        full_report = "\n\n".join(report_blocks).strip()
        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([case_id, full_report])

def process_one_case(args):
    img_id, vqa_data = args

    subset = vqa_data[vqa_data["Image ID"] == img_id]
    dataset = subset["dataset"].dropna().unique()[0]
    ct_folder = f"/mnt/blobdata/ReXGroundingCT/data/ReXGroundingCT/img/{img_id}.nii.gz"
    # mask_folder = f"/mnt/blobdata/data/DeepTumorVQA/Subset-v3/segmentations/VISTA3D/{dataset}/{img_id}"
    # csv_file = f"/mnt/blobdata/data/DeepTumorVQA/Subset-v3/structured_report/VISTA3D/{img_id}_report.csv"
    mask_folder = f"/mnt/blobdata/ReXGroundingCT/data/ReXGroundingCT/label/{img_id}.nii.gz"
    csv_file = f"/mnt/blobdata/ReXGroundingCT/structured_report/{img_id}_report.csv"
    with open(os.devnull, "w") as f, redirect_stdout(f):
        generate_reports(
            ct_folder=ct_folder,
            mask_folder=mask_folder,
            csv_file=csv_file,
            restart_csv=True,
        )

    return img_id


def main():
    # vqa_path = "/mnt/blobdata/code/3DMedAgent/VQA/DeepTumorVQA_sampled-v3.csv"
    vqa_path = "/mnt/blobdata/ReXGroundingCT/data_json/DeepChestVQA.csv"
    vqa_data = pd.read_csv(vqa_path)

    image_ids = vqa_data["Image ID"].dropna().unique()

    num_workers = 12

    with mp.Pool(processes=num_workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(
                process_one_case,
                [(img_id, vqa_data) for img_id in image_ids],
            ),
            total=len(image_ids),
            desc="Analyzing cases",
            ncols=100,
        ):
            pass


if __name__ == "__main__":
    main()
