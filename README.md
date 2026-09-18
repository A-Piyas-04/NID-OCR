# NID-OCR — PaddleOCR Fine-Tuning

Fine-tune PaddleOCR PP-OCRv5 (`PP-OCRv5_server_det` + `PP-OCRv5_server_rec`) for Bangladesh NID cards using **one unified Bangla + English** recognition path.

**All training, evaluation, export, and inference run on Kaggle.**
There is no local GPU / local training setup for this project.

## Goal

Improve NID OCR through model fine-tuning only:

- Bangla + English text recognition
- NID layout / multi-line text regions
- character / glyph spacing issues

Not in scope: preprocessing redesign, parser/regex logic, or a separate Bengali OCR model.

## Deliverables

1. Documented fine-tuning steps
2. Working scripts/notebooks tested on 2-4 NIDs (process validation)
3. Estimate of how much real NID data is needed
4. Reproducible repository

## Verified Kaggle stack (Phase 01)

| Package | Version |
| --- | --- |
| PaddleOCR | 3.3.0 |
| PaddlePaddle | **3.3.0** |
| PaddleX | 3.3.3 |

> The older NID service report used PaddlePaddle **3.2.0**. This fine-tuning repo uses **3.3.0** on Kaggle (Python 3.12 / modern CUDA driver). See `Docs/VERSION_NOTES.md` and `requirements.txt`.

## Folder structure

```text
NID-OCR/
|-- README.md
|-- requirements.txt
|-- .gitignore
|
|-- Docs/
|   |-- SCOPE.md                       # project scope
|   |-- Task1.txt                      # original task brief
|   |-- VERSION_NOTES.md               # verified Kaggle pins (PaddlePaddle 3.3.0)
|   |-- PaddleOCR_NID_Phase_Plan.md    # phase-by-phase plan
|   +-- report/                        # technical reports (Markdown)
|
|-- notebooks/                         # Kaggle-oriented notebooks
|   +-- README.md
|
|-- configs/
|   |-- detection/                     # PP-OCRv5_server_det configs
|   +-- recognition/                   # PP-OCRv5_server_rec configs
|
|-- scripts/                           # helpers (charset, validate, evaluate)
|   +-- README.md
|
|-- data/
|   +-- README.md                      # private Kaggle Dataset layout only
|                                      # (no real NID images in Git)
|
|-- experiments/
|   +-- README.md                      # lightweight anonymized summaries only
|
+-- models/
    +-- README.md                      # notes for Kaggle exports / checkpoints
                                       # (weights not committed)
```

## Where things live

| Location | Contents |
| --- | --- |
| **This GitHub repo** | docs, notebooks, scripts, configs |
| **Private Kaggle Dataset** | real NID images, polygons, crops, labels |
| **Kaggle Notebook** | install, train, evaluate, export, compare |
| `/kaggle/working/` | PaddleOCR clone, checkpoints, logs, exports |

## Docs

- [`Docs/SCOPE.md`](Docs/SCOPE.md) - in/out of scope
- [`Docs/plan/PaddleOCR_NID_Phase_Plan.md`](Docs/plan/PaddleOCR_NID_Phase_Plan.md) - full phase plan
- [`Docs/VERSION_NOTES.md`](Docs/VERSION_NOTES.md) - verified Kaggle versions (PaddlePaddle **3.3.0**, not the report 3.2.0)
- [`Docs/plan/Task1.txt`](Docs/plan/Task1.txt) - original task statement
- [`Docs/report/`](Docs/report/) - OCR model report and fine-tuning overview

## Quick start (Kaggle)

1. Clone / upload this repo content into a Kaggle Notebook (or sync notebooks from `notebooks/`).
2. Attach the **private** NID Kaggle Dataset.
3. Follow phases in [`Docs/plan/PaddleOCR_NID_Phase_Plan.md`](Docs/plan/PaddleOCR_NID_Phase_Plan.md), starting at Phase 01 (`notebooks/01-phase01-environment.ipynb`).
4. Keep real NID images out of Git.
