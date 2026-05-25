import os
import subprocess

def run_totalsegmentator_cli(input_path, out_case_dir, organ_list, device="gpu"):
    """
    Call TotalSegmentator CLI with --roi_subset
    """
    organ_map = {
        "liver": "liver",
        "spleen": "spleen",
        "pancreas": "pancreas",
        "colon": "colon",
        "left kidney": "kidney_left",
        "right kidney": "kidney_right",
        "left kidney cyst": "kidney_cyst_left",
        "right kidney cyst": "kidney_cyst_right",
    }

    # roi_args = [organ_map[o] for o in organ_list]

    cmd = [
        "TotalSegmentator",
        "-i", input_path,
        "-o", out_case_dir,
        "-ta", "liver_segments",
        "-d", device,
        # "--roi_subset", *roi_args,
    ]

    env = os.environ.copy()
    totalseg_home = env.get(
        "TOTALSEG_HOME_DIR",
        "/mnt/nas/ziyue/project/3DMedAgent/.cache/totalsegmentator",
    )
    os.makedirs(totalseg_home, exist_ok=True)
    env["TOTALSEG_HOME_DIR"] = totalseg_home

    result = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "TotalSegmentator failed "
            f"(returncode={result.returncode})\n"
            f"command: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    return result
