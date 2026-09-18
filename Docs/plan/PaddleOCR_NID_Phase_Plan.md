# Bangladesh NID OCR

## PaddleOCR Fine-Tuning — Phase Plan

*Beginner-friendly version — direct tasks, plain language*

> **This project runs entirely on Kaggle.**  
> Training, evaluation, export, and inference all happen in Kaggle Notebooks.  
> Do **not** set up a local GPU, local CUDA stack, or local training environment.  
> Real NID images live in a **private Kaggle Dataset**, not in Git.

---

## Ultimate goal

Fine-tune the existing PaddleOCR PP-OCRv5 detector and recognizer so **one** OCR pipeline reads Bangladesh NIDs better — including Bangla + English text — and becomes more familiar with NID layout, multi-line text regions, and character/glyph spacing issues.

## Scope (from Task1)

We are focusing on **PaddleOCR model fine-tuning only**.

We will fine-tune **both**:

- `PP-OCRv5_server_det` (detection)
- `PP-OCRv5_server_rec` (recognition)

Recognition stays **one unified Bangla + English path** inside this same PaddleOCR model. We will **not** add a second Bengali-only OCR model.

Fine-tuning should improve:

- Bangla text detection / recognition when accuracy is low
- NID format familiarization
- multi-line / dense text-region behaviour
- character / alphabet / glyph spacing issues

### Out of scope

We are **not** redesigning:

- image preprocessing
- parser / regex / field-extraction logic
- a separate “X model for Bengali”
- another OCR architecture

> Detector learns: *Where is the text?*  
> Recognizer learns: *What does the text say?*  
> Mapping strings to `father_name` / `mother_name` / `address` remains outside this task.

---

## Execution setup — Kaggle only

| Location | What goes there |
| --- | --- |
| **GitHub repository** | notebooks, scripts, configs, docs, README, lightweight result summaries |
| **Private Kaggle Dataset** | real NID images, detection polygons, recognition crops, labels |
| **Kaggle Notebook** | install / verify packages, train, evaluate, export, compare |
| **`/kaggle/working/`** | cloned PaddleOCR, checkpoints, logs, exported models, experiment outputs |
| **Git** | never store real NID images |

Typical Kaggle paths:

```text
/kaggle/input/<private-dataset-name>/     # read-only NID data
/kaggle/working/                          # writable experiment area
/kaggle/working/PaddleOCR/                # cloned PaddleOCR source
/kaggle/working/output/                   # checkpoints / exports / metrics
```

### Hard rules

1. All compute runs on **Kaggle** (CPU or Kaggle GPU accelerator — not a local machine).
2. Attach the private dataset to the notebook; do not download NID images onto a personal computer for training.
3. Persist useful outputs from `/kaggle/working/` as Kaggle Dataset versions / notebook outputs when needed.
4. Commit only code, configs, docs, and anonymized summaries back to the repo.

---

## Requested deliverables

- **D1** — Complete step-by-step fine-tuning documentation
- **D2** — Working fine-tuning scripts / notebooks tested on 2–4 NIDs
- **D3** — A justified estimate of how much real NID data is needed
- **D4** — A clean, reproducible repository

---

## Roadmap at a glance

*Read this section first. Every later section explains one phase in detail.*

| Phase | Main task | Where it runs |
| --- | --- | --- |
| 01 | Confirm PaddleOCR setup | Kaggle Notebook |
| 02 | Run current model (baseline) | Kaggle Notebook + private dataset |
| 03 | Prepare detection data | annotate → upload private Kaggle Dataset |
| 04 | Prepare recognition data | annotate → upload private Kaggle Dataset |
| 05 | Build Bangla + English dictionary | Kaggle Notebook |
| 06 | Fine-tune recognition | Kaggle Notebook (GPU recommended) |
| 07 | Fine-tune detection | Kaggle Notebook (GPU recommended) |
| 08 | Combine both models | Kaggle Notebook |
| 09 | Compare before vs after | Kaggle Notebook |
| 10 | Estimate real data need | docs + optional Kaggle learning-curve runs |
| 11 | Finalize docs + repo | GitHub + verified Kaggle notebooks |

**Deliverable finish points:** D2 finishes in Phase 8. D3 finishes in Phase 10. D1 and D4 are finalized in Phase 11.

---

## Phase 01 — Confirm the exact PaddleOCR setup on Kaggle

*First remove uncertainty: know exactly what software, model files and commands we will use inside a Kaggle Notebook.*

### What you actually do

1. **Create a Kaggle Notebook** and enable a GPU accelerator if available for later training phases.
2. **Record the Kaggle environment** — Python, accelerator type, CUDA info if present, PaddleOCR / PaddlePaddle / PaddleX versions.
3. **Confirm the two starting models** — detector = `PP-OCRv5_server_det`; recognizer = `PP-OCRv5_server_rec`.
4. **Clone / pin PaddleOCR under `/kaggle/working/`** — record the exact Git commit or release tag. Do not depend on an unspecified moving `main`.
5. **Find the real config files** for those exact models in that pinned version.
6. **Test the training commands** — confirm exact train / evaluate / export entrypoints in Kaggle. Do not copy an old local command blindly.
7. **Confirm custom Bangla + English characters** — check how this recognizer accepts a custom character dictionary while starting from pretrained weights, and that Bangla Unicode reads correctly in Kaggle.
8. **Save findings** into `docs/VERSION_NOTES.md` (and keep the notebook in the repo).

| | |
| --- | --- |
| **OUTPUT** | Exact versions, config paths, model paths, Git commit, and working Kaggle command templates. |
| **DONE WHEN** | We can explain exactly what Kaggle command will be run and why. |

**Deliverable status:** D1 starts. D4 starts.

Phase 01 does **not** mean annotate NIDs, build the final dictionary, or claim accuracy improvement.

---

## Phase 02 — Run the current model on 2–4 NIDs

*Create a clear “before fine-tuning” reference using the same cards we will test again later.*

### What you actually do

1. **Choose 2–4 private NID images** and put them in a **private Kaggle Dataset**.
2. **Attach that dataset** to a Kaggle Notebook.
3. **Run stock PaddleOCR** with `PP-OCRv5_server_det` + `PP-OCRv5_server_rec` — do not change weights yet.
4. **Save detector output** — boxes / polygons (and overlays if useful).
5. **Save recognizer output** — text string + confidence.
6. **Save one result file per card** under `/kaggle/working/`, e.g.:

```text
/kaggle/working/experiments/phase02_baseline/
  nid_01/result.json
  nid_01/overlay.png
  nid_02/result.json
  nid_02/overlay.png
```

| | |
| --- | --- |
| **OUTPUT** | Baseline OCR outputs for the 2–4 sample NIDs. |
| **DONE WHEN** | We have “before” boxes, text and confidence saved from Kaggle. |

**Deliverable status:** Supports D2 and D1.

---

## Phase 03 — Prepare detection training data

*Teach the detector where text is located on a Bangladesh NID.*

### What you actually do

1. **Use the full NID image** — detection training uses the complete card.
2. **Draw a box/polygon around each useful text line** — Bangla lines, English lines, DOB, NID number, address lines.
3. **Keep separate lines separate** — do not join neighbouring lines by accident.
4. **Create train and validation lists** in the dataset structure required by the verified PaddleOCR config.
5. **Upload images + annotations to the private Kaggle Dataset** — keep them out of Git.
6. **In a Kaggle Notebook, run a dataset check** — paths resolve under `/kaggle/input/...` and polygons load correctly.

| | |
| --- | --- |
| **OUTPUT** | Full NID images + text-region polygon labels in PaddleOCR detection format, available as a private Kaggle Dataset. |
| **DONE WHEN** | The detection dataset loads in Kaggle without format/path errors. |

**Deliverable status:** D2 progresses.

---

## Phase 04 — Prepare recognition training data

*Teach the recognizer what each text line actually says.*

### What you actually do

1. **Crop each labelled text line** — one image crop per line/region.
2. **Type the exact correct text** — Bangla, English, or numeric transcription.
3. **Keep Bangla + English together** — one recognizer, one unified training path.
4. **Create `train_label.txt` and `val_label.txt`** — each row maps crop path → correct text.
5. **Split by NID card** — all crops from the same physical card stay in the same split.
6. **Upload crops + labels to the private Kaggle Dataset** and verify they load in a Kaggle Notebook.

| | |
| --- | --- |
| **OUTPUT** | Single-line crops + exact text labels for recognition training, on Kaggle. |
| **DONE WHEN** | PaddleOCR can load the recognition dataset in Kaggle and every crop has a correct label. |

**Deliverable status:** D2 progresses.

---

## Phase 05 — Build one Bangla + English character dictionary

*Make sure the recognizer is allowed to output every character that appears on Bangladesh NIDs.*

### What you actually do

1. **In Kaggle, read all ground-truth labels** from Phase 04.
2. **Collect every unique character** — Bangla letters/signs, English letters, digits, needed punctuation.
3. **Preserve Bangla Unicode exactly** — do not simplify vowel signs or rewrite conjunct sequences.
4. **Generate the dictionary with a script** (e.g. `scripts/build_charset.py`) inside the notebook / repo workflow.
5. **Validate the dictionary** — fail if any train/validation character is missing.

| | |
| --- | --- |
| **OUTPUT** | One bilingual character dictionary generated from the dataset. |
| **DONE WHEN** | Every labelled character is covered by the recognizer vocabulary. |

**Deliverable status:** D1 gains the bilingual-recognition method.

---

## Phase 06 — Fine-tune the recognition model first

*Prove the easier half of the training pipeline first: text crop → training → exported recognizer — all on Kaggle.*

### What you actually do

1. **Load pretrained `PP-OCRv5_server_rec` weights** — adapt an existing model; do not train from zero.
2. **Point the config** to Kaggle paths for crops, labels, and the Bangla + English dictionary.
3. **Run a short Kaggle training test** — with only 2–4 NIDs, the goal is to prove the code works. Overfitting is expected.
4. **Confirm a checkpoint is created** under `/kaggle/working/output/`.
5. **Evaluate, export and reload** in the same (or a follow-up) Kaggle Notebook — prove the exported recognizer can read a crop.

| | |
| --- | --- |
| **OUTPUT** | Fine-tuned + exported recognition model and a working Kaggle recognition training notebook/script. |
| **DONE WHEN** | Training runs on Kaggle, a checkpoint is saved, export works, and the exported recognizer can read a crop. |

**Deliverable status:** D2 is half complete.

---

## Phase 07 — Fine-tune the detection model

*Now prove the full-card detector training path on Kaggle.*

### What you actually do

1. **Load pretrained `PP-OCRv5_server_det` weights**.
2. **Point the config** to full NID images and polygon labels from `/kaggle/input/...`.
3. **Run a short Kaggle training test** — 2–4 NIDs only validate mechanics.
4. **Confirm checkpoints** are created under `/kaggle/working/output/`.
5. **Evaluate, export and reload** — visualize returned text boxes on an NID inside Kaggle.

| | |
| --- | --- |
| **OUTPUT** | Fine-tuned + exported detector and a working Kaggle detection training notebook/script. |
| **DONE WHEN** | The exported detector loads in Kaggle and returns text regions on an NID image. |

**Deliverable status:** D2 is nearly complete.

---

## Phase 08 — Use both fine-tuned models together

*Run one PaddleOCR pipeline with the custom detector and custom Bangla + English recognizer — in Kaggle.*

### What you actually do

1. **Load the exported detector directory** from Phase 07.
2. **Load the exported recognizer directory** from Phase 06.
3. **Run the same 2–4 NIDs from Phase 02**.
4. **Save boxes, text and confidence** in the same format as the baseline.
5. **Confirm** detection feeds recognition correctly and both custom model directories are actually used.

| | |
| --- | --- |
| **OUTPUT** | One end-to-end PaddleOCR pipeline using both fine-tuned models, runnable on Kaggle. |
| **DONE WHEN** | The same NID image goes through custom detection + custom recognition successfully. |

**Deliverable status:** D2 finishes here.

---

## Phase 09 — Compare before vs after

*Check what changed, but do not claim real accuracy from only 2–4 NIDs.*

### What you actually do

1. **Compare detection** — missed, merged, split or clipped text boxes.
2. **Compare recognition** — Bangla, English and numeric transcription separately.
3. **Calculate simple model metrics** — recognition: CER/edit distance; detection: precision/recall and IoU matching when possible.
4. **Report Bangla separately** — overall OCR can look good while Bangla is still poor.
5. **Write the limitation clearly** — this is a process test and a debugging comparison, not a trustworthy benchmark.

| | |
| --- | --- |
| **OUTPUT** | A small before/after technical comparison plus working evaluation code/notebook. |
| **DONE WHEN** | We can show what changed without overstating the result. |

**Deliverable status:** D1 and D4 gain evaluation material.

---

## Phase 10 — Decide how much real data is needed

*Turn the small Kaggle demo into a realistic data-collection plan for serious fine-tuning.*

### What you actually do

1. **Treat 2–4 NIDs only as a smoke test** — they prove the pipeline, not accuracy.
2. **Use a practical first target** — roughly **300–500 diverse NID cards** and at least about **5,000 real recognition line crops** as an initial serious dataset.
3. **Collect diversity, not copies** — different NID generations, Bangla names, addresses, English text, digits, hard visual cases.
4. **Check Bangla character coverage** — common letters, signs and conjunct patterns.
5. **Use a learning curve on Kaggle** — train with 25%, 50%, 75%, 100% of the data. If validation error is still dropping strongly, collect more.

| | |
| --- | --- |
| **OUTPUT** | A written data-size recommendation plus a method for deciding whether more data is still needed. |
| **DONE WHEN** | The estimate is supported by coverage and learning-curve logic, not by guessing. |

**Deliverable status:** D3 finishes here.

---

## Phase 11 — Finalize the documentation and repository

*Turn the experiment into a clean project that another engineer can reproduce on Kaggle.*

### What you actually do

1. **Clean the repo structure** — notebooks, configs, scripts, docs, requirements.
2. **Keep real NID images outside Git** — private Kaggle Dataset only.
3. **Write Kaggle run instructions** — attach dataset → prepare data → train recognition → train detection → export → combined inference → evaluate.
4. **Document where outputs are saved** — `/kaggle/working/` checkpoints, exports, logs, evaluation files.
5. **Add beginner troubleshooting** — wrong `/kaggle/input` path, missing dictionary character, wrong config, checkpoint not loading, export path mismatch, accelerator not enabled, etc.
6. **Make the docs match the notebooks that actually worked on Kaggle** — remove unverified commands.

Repository shape:

```text
NID-OCR/
├── README.md
├── requirements.txt
├── .gitignore
│
├── Docs/
│   ├── SCOPE.md
│   ├── Task1.txt
│   ├── VERSION_NOTES.md
│   ├── PaddleOCR_NID_Phase_Plan.md
│   └── report/                        # written experiment notes
│
├── notebooks/                         # Kaggle-oriented notebooks
│   └── README.md
│
├── configs/
│   ├── detection/                     # PP-OCRv5_server_det configs
│   └── recognition/                   # PP-OCRv5_server_rec configs
│
├── scripts/                           # charset / validate / evaluate helpers
│   └── README.md
│
├── data/
│   └── README.md                      # private Kaggle Dataset layout only
│
├── experiments/
│   └── README.md                      # anonymized summaries only
│
└── models/
    └── README.md                      # export/checkpoint notes (weights not in Git)
```

| | |
| --- | --- |
| **OUTPUT** | A reproducible Kaggle-based fine-tuning repo and final step-by-step documentation. |
| **DONE WHEN** | A second engineer can clone the repo, attach the private Kaggle Dataset, and reproduce the workflow on Kaggle without a local GPU. |

**Deliverable status:** D1 and D4 finish here.

---

## Final completion map

| Deliverable | Finished at |
| --- | --- |
| D1 — Documentation | Phase 11, after all steps and commands have been verified on Kaggle. |
| D2 — 2–4 NID fine-tuning workflow | Phase 8, after recognition + detection training, export and combined inference work on Kaggle. |
| D3 — Data requirement | Phase 10, with an initial target plus learning-curve method. |
| D4 — Repository | Started in Phase 1 and finalized in Phase 11. |

> **Important:** For the 2–4 NID experiment, “success” does **not** mean high accuracy.  
> Success means the whole process works on Kaggle:  
> **data loads → training runs → checkpoint saves → model exports → exported model reloads → combined PaddleOCR inference runs.**
