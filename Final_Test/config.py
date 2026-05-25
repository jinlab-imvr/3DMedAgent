from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


def _path_from_env(name: str, default: str | Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


PROJECT_ROOT = _path_from_env("THREEDMEDAGENT_PROJECT_ROOT", Path(__file__).resolve().parents[1])
DATA_ROOT = _path_from_env("DEEPTUMORVQA_DATA_ROOT", "data/DeepTumorVQA")
CTCLIP_PROJECT_ROOT = _path_from_env("CTCLIP_PROJECT_ROOT", "models/CT-CLIP")

for _path in (PROJECT_ROOT, PROJECT_ROOT / "CT_Clip"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

VQA_PATH = PROJECT_ROOT / "VQA" / "DeepTumorVQA_sampled-v3.csv"
RAW_CT_ROOT = DATA_ROOT / "data"
SUBSET_ROOT = DATA_ROOT / "Subset-v3"
REPORT_DIR = DATA_ROOT / "structured_report" / "VISTA3D"
MASK_DIR = SUBSET_ROOT / "segmentations" / "VISTA3D"
CLIP_GLOBAL_DIR = SUBSET_ROOT / "clip_global"
CLIP_EMBEDDING_DIR = SUBSET_ROOT / "clip_embedding"
CLIP_DETAIL_DIR = SUBSET_ROOT / "clip_detail"
CLIP_DETAIL_SLICE_DIR = SUBSET_ROOT / "clip_detail_slice"
RUNTIME_CACHE_DIR = SUBSET_ROOT / "runtime_tools" / "v1"
T1S_CACHE_DIR = SUBSET_ROOT / "t1s_loop" / "v1"
T1S_RENDER_DIR = SUBSET_ROOT / "t1s_loop" / "v1_renders"
PREDICTION_ROOT = DATA_ROOT / "prediction" / "Final_Test"
SAVE_DIR = PREDICTION_ROOT / "memory_v3"
T1S_SAVE_DIR = PREDICTION_ROOT / "memory_t1s"

CTCLIP_MODEL_PATH = CTCLIP_PROJECT_ROOT / "models" / "CT-CLIP_v2.pt"

MAX_CONCURRENCY = int(os.environ.get("FINAL_TEST_MAX_CONCURRENCY", "6"))
MAX_RETRIES = int(os.environ.get("FINAL_TEST_MAX_RETRIES", "3"))
BASE_BACKOFF = float(os.environ.get("FINAL_TEST_BASE_BACKOFF", "1.5"))
MODEL_NAME = os.environ.get("FINAL_TEST_OPENAI_MODEL", "gpt-5")

SYSTEM_PROMPT = "You are a careful medical imaging assistant. Always output STRICT JSON only."

TARGET_SUBTYPES = (
    "liver_lesion_existence",
    "largest lesion slice",
    "lesion counting",
    "tumor organ HU difference",
)


def _load_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    try:
        from key import OPENAI_API_KEY as local_key

        return str(local_key)
    except Exception:
        return ""


OPENAI_API_KEY = _load_api_key()


def _apply_local_overrides() -> None:
    try:
        from config_run import LOCAL_OVERRIDES
    except Exception:
        return
    if not isinstance(LOCAL_OVERRIDES, dict):
        return
    globals_dict = globals()
    for name, value in LOCAL_OVERRIDES.items():
        if name not in globals_dict:
            continue
        if isinstance(globals_dict[name], Path):
            globals_dict[name] = Path(value).expanduser()
        else:
            globals_dict[name] = value


_apply_local_overrides()

for _path in (PROJECT_ROOT, PROJECT_ROOT / "CT_Clip", CTCLIP_PROJECT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _export_environment_defaults() -> None:
    defaults = {
        "THREEDMEDAGENT_PROJECT_ROOT": PROJECT_ROOT,
        "DEEPTUMORVQA_DATA_ROOT": DATA_ROOT,
        "CTCLIP_PROJECT_ROOT": CTCLIP_PROJECT_ROOT,
        "CTCLIP_MODEL_PATH": CTCLIP_MODEL_PATH,
        "CTCLIP_EMBEDDING_DIR": CLIP_EMBEDDING_DIR,
        "CTCLIP_MASK_DIR": MASK_DIR,
        "CTCLIP_DATA_ROOT": RAW_CT_ROOT,
        "CTCLIP_REPORT_DIR": REPORT_DIR,
        "CTCLIP_DETAIL_DIR": CLIP_DETAIL_DIR,
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, str(value))


_export_environment_defaults()


def as_str(value: Any) -> str:
    return str(value)
