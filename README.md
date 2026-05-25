# 3DMedAgent

This repository contains the 3DMedAgent experimental code. The current DeepTumorVQA Final Test pipeline is implemented in `Final_Test/GPT_memory_t1s.py`.

## Final Test Pipeline

The main pipeline builds leak-safe text memory, optionally calls CT-CLIP runtime tools, and can optionally run a Think-with-1-Slice (T1S) visual loop.

## Preparation Steps

### 1. Prepare masks

Run `segmentation.py` from the repository root to generate organ and lesion masks. The expected output layout is:

```text
$DEEPTUMORVQA_DATA_ROOT/Subset-v3/segmentations/VISTA3D/<dataset>/<image_id>/
```

Adjust GPU, segmentation backend, and case range according to the current server setup.

### 2. Generate structured reports

Run `Structural_Report/main.py` with the raw CT and mask directories. It writes one structured report per case:

```text
$DEEPTUMORVQA_DATA_ROOT/structured_report/VISTA3D/<image_id>_report.csv
```

The report summarizes organ volume, HU, z range, lesion size, lesion HU, segment/location, and related text evidence. It is one of the primary sources for text memory.

### 3. Generate CT-CLIP global evidence

Run `CT_Clip/clip_classify.py` to produce global probabilities for 5 canonical organs and 3 lesion types:

```text
$DEEPTUMORVQA_DATA_ROOT/Subset-v3/clip_global/<image_id>.json
```

This evidence is mainly used for recognition and existence questions.

### 4. Generate CT-CLIP embeddings

Run `CT_Clip/clip_encode.py` to generate CT-CLIP patch embeddings for each volume:

```text
$DEEPTUMORVQA_DATA_ROOT/Subset-v3/clip_embedding/<image_id>.pt
```

`clip_detail` preprocessing and runtime tools reuse these embeddings.

### 5. Generate CT-CLIP detail evidence

Run `CT_Clip/precompute_organ_scores.py` with CT-CLIP embeddings and masks to generate organ-section lesion scores:

```text
$DEEPTUMORVQA_DATA_ROOT/Subset-v3/clip_detail/<image_id>.json
```

This evidence is used for lesion-level section evidence and candidate slice queues.

### 6. Generate CT-CLIP slice-detail evidence

Run `CT_Clip/precompute_clip_detail.py` to generate finer slice-level lesion scores:

```text
$DEEPTUMORVQA_DATA_ROOT/Subset-v3/clip_detail_slice/<image_id>.json
```

This evidence is used for candidate slices and largest-slice / option-slice questions.

### 7. Runtime cache and T1S cache

Runtime tool cache and T1S render/cache files do not need to be precomputed. `GPT_memory_t1s.py` writes them on demand:

```text
$DEEPTUMORVQA_DATA_ROOT/Subset-v3/runtime_tools/
$DEEPTUMORVQA_DATA_ROOT/Subset-v3/t1s_loop/
```

## Run the Main Pipeline

### Text memory + runtime tools, without T1S

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n ms-general python Final_Test/GPT_memory_t1s.py \
  --question-type "visual reasoning" \
  --max-per-subtype 0 \
  --normalizer rule \
  --memory-mode facts \
  --include-runtime-tools \
  --tool-selector gpt \
  --device cuda:0 \
  --save-dir "$DEEPTUMORVQA_DATA_ROOT/prediction/Final_Test/no_t1s_runtime" \
  --concurrency 6
```

### Text memory + runtime tools + T1S

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n ms-general python Final_Test/GPT_memory_t1s.py \
  --question-type "visual reasoning" \
  --max-per-subtype 0 \
  --normalizer rule \
  --memory-mode facts \
  --include-runtime-tools \
  --tool-selector gpt \
  --include-t1s \
  --t1s-max-iters 5 \
  --device cuda:0 \
  --save-dir "$DEEPTUMORVQA_DATA_ROOT/prediction/Final_Test/t1s_runtime" \
  --concurrency 4
```

## Common Options

- `--include-runtime-tools`: enable CT-CLIP runtime tools.
- `--include-t1s`: enable the T1S slice-level visual loop.
- `--t1s-max-iters`: maximum number of T1S iterations.
- `--normalizer rule|gpt`: choose the organ / lesion target normalizer.
- `--memory-mode facts|hybrid`: optionally generate compressed reasoning memory.
- `--skip-existing`: skip records that already exist in the output directory for the same `(case_idx, question_subtype)`.
