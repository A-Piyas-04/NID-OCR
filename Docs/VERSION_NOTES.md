# Environment & PaddleOCR Version Notes

> Verified on Kaggle via `notebooks/01-phase01-environment.ipynb` (Phase 01).  
> Recorded: 2026-09-18.

## Python

Version: 3.12.13

Platform: Linux (Kaggle) — `Linux-6.12.90+-x86_64-with-glibc2.35`

Accelerator: GPU available (2× Tesla T4). Driver reports CUDA 13.0.

## PaddlePaddle

Version: **3.3.0**

Compiled with CUDA (this Phase 01 session): `False` (CPU wheel landed after GPU index attempts; GPU wheel to be re-tried for training phases).

### Why 3.3.0 (not 3.2.0)

The earlier NID service report (`Docs/report/OCR_Model_Report.md`) documented **PaddlePaddle 3.2.0** on that deployment.

This fine-tuning project pins **PaddlePaddle 3.3.0** on Kaggle because:

- Kaggle is Python **3.12** with a modern NVIDIA driver (**CUDA 13.0**).
- `paddlepaddle==3.2.0` / older `cu118` / `cu121` indexes did not provide a matching wheel.
- Official install indexes for this environment use **3.3.0** (`cu130` → `cu126` → `cpu`).
- Matches the installed **PaddleOCR 3.3.0** / **PaddleX 3.3.3** family on the same session.

| Context | PaddlePaddle |
| --- | --- |
| Historical NID service (report) | 3.2.0 |
| **This repo / Kaggle fine-tuning (authoritative)** | **3.3.0** |

## PaddleOCR

Version: 3.3.0

Git commit: `dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf`

Describe / tag: `dab3fe3`

Requested ref: default shallow tip

Clone path: `/kaggle/working/PaddleOCR`

## PaddleX

Version: 3.3.3

## Target Models

Detection:
PP-OCRv5_server_det

Recognition:
PP-OCRv5_server_rec

Cached detection model path: Not cached yet (as of Phase 01)

Cached recognition model path: Not cached yet (as of Phase 01)

## Training Entry Points

Classic tools present under the cloned repo:

- `tools/train.py` — FOUND
- `tools/eval.py` — FOUND
- `tools/export_model.py` — FOUND

PaddleX-style `main.py`: not found in this Phase 01 search (prefer classic `tools/*.py` until re-checked).

## Detection Config

`configs/det/PP-OCRv5/PP-OCRv5_server_det.yml`

Absolute (Kaggle): `/kaggle/working/PaddleOCR/configs/det/PP-OCRv5/PP-OCRv5_server_det.yml`

## Recognition Config

`configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml`

Absolute (Kaggle): `/kaggle/working/PaddleOCR/configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml`

Key fields observed:

- `character_dict_path: ./ppocr/utils/dict/ppocrv5_dict.txt`
- `use_space_char: true`
- `pretrained_model` present (empty in stock rec config — supply at train time)

## Bangla + English Dictionary Behaviour

- Unified Bangla + English path in this recognizer (no second Bengali model).
- Config dict/charset field present: yes (`character_dict_path`)
- `use_space_char` present: yes
- UTF-8 Bangla round-trip on Kaggle: PASS
- Final NID charset built in Phase 05.

## Train Command

```bash
python tools/train.py -c configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml -o Global.pretrained_model=<REC_PRETRAINED_WEIGHTS>
```

```bash
python tools/train.py -c configs/det/PP-OCRv5/PP-OCRv5_server_det.yml -o Global.pretrained_model=<DET_PRETRAINED_WEIGHTS>
```

## Evaluation Command

```bash
python tools/eval.py -c configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml -o Global.pretrained_model=<REC_CHECKPOINT>
```

## Export Command

```bash
python tools/export_model.py -c configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml -o Global.pretrained_model=<REC_CHECKPOINT> Global.save_inference_dir=<REC_EXPORT_DIR>
```

## Install pins (Kaggle)

```text
paddleocr==3.3.0
paddlex==3.3.3
paddlepaddle-gpu==3.3.0   # prefer cu130, then cu126; else paddlepaddle==3.3.0 (CPU)
```

See `requirements.txt` and `notebooks/01-phase01-environment.ipynb`.

## Kaggle Paths

- `/kaggle/input/<private-dataset-name>/` — NID data (Phase 02+)
- `/kaggle/working/PaddleOCR/` — pinned source
- `/kaggle/working/output/` — checkpoints / exports (later)

## Phase 01 Checklist

- [x] Kaggle environment recorded
- [x] Packages installed and verified (PaddlePaddle **3.3.0**)
- [x] Models confirmed (server det + rec)
- [x] PaddleOCR commit recorded
- [x] Configs located
- [x] Train/eval/export entrypoints located
- [x] Bangla Unicode + dict mechanism checked
