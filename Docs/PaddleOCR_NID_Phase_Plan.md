# Bangladesh NID OCR

## PaddleOCR Fine-Tuning — Phase Plan

*Beginner-friendly version — direct tasks, plain language*

### Ultimate goal

Fine-tune the existing PaddleOCR PP-OCRv5 detector and recognizer so **one** OCR pipeline reads Bangladesh NIDs better, including Bangla + English text and NID-specific text regions.

### Scope

We are focusing on **model fine-tuning only**. We are not redesigning preprocessing, parser/regex logic, or switching to another OCR model.

### Requested deliverables

- **D1** — Complete step-by-step fine-tuning documentation
- **D2** — Working fine-tuning scripts tested on 2–4 NIDs
- **D3** — A justified estimate of how much real data is needed
- **D4** — A clean, reproducible repository

---

## Roadmap at a glance

*Read this section first. Every later section explains one phase in detail.*

| Phase | Main task |
| --- | --- |
| 01 | Confirm PaddleOCR setup |
| 02 | Run current model |
| 03 | Prepare detection data |
| 04 | Prepare recognition data |
| 05 | Build Bangla + English dictionary |
| 06 | Fine-tune recognition |
| 07 | Fine-tune detection |
| 08 | Combine both models |
| 09 | Compare before vs after |
| 10 | Estimate real data need |
| 11 | Finalize docs + repo |

**Deliverable finish points:** D2 finishes in Phase 8. D3 finishes in Phase 10. D1 and D4 are finalized in Phase 11.

---

## Phase 01 — Confirm the exact PaddleOCR setup

*First remove uncertainty: know exactly what software, model files and commands we will use.*

### What you actually do

1. **Check the installed versions** — write down the current PaddleOCR, PaddlePaddle and PaddleX versions.
2. **Confirm the two starting models** — detector = `PP-OCRv5_server_det`; recognizer = `PP-OCRv5_server_rec`.
3. **Find the real config files** — locate the training configuration files for those exact models in this version.
4. **Test the training commands** — confirm the exact commands for train, evaluate and export. Do not copy an old command blindly.
5. **Confirm custom Bangla + English characters** — check how this recognizer accepts a custom character dictionary while starting from pretrained weights.

| | |
| --- | --- |
| **OUTPUT** | Exact versions, config paths, model paths and working command templates. |
| **DONE WHEN** | We can explain exactly what command will be run and why. |

**Deliverable status:** D1 starts. D4 starts.

---

## Phase 02 — Run the current model on 2–4 NIDs

*Create a clear “before fine-tuning” reference using the same cards we will test again later.*

### What you actually do

1. **Choose 2–4 private NID images** — these are only for learning the process, not for measuring real accuracy.
2. **Run the current stock PaddleOCR** — do not change the model yet.
3. **Save detector output** — keep the boxes/polygons found on each card.
4. **Save recognizer output** — keep each text string and confidence score.
5. **Save a simple result file for every card** — later we will compare these exact outputs with the fine-tuned model.

| | |
| --- | --- |
| **OUTPUT** | Baseline OCR outputs for the 2–4 sample NIDs. |
| **DONE WHEN** | We have “before” boxes, text and confidence saved. |

**Deliverable status:** Supports D2 and D1.

---

## Phase 03 — Prepare detection training data

*Teach the detector where text is located on a Bangladesh NID.*

### What you actually do

1. **Use the full NID image** — detection training uses the complete card image.
2. **Draw a box/polygon around each useful text line** — include Bangla lines, English lines, DOB, NID number and address lines.
3. **Keep separate lines separate** — do not accidentally join two text lines into one annotation.
4. **Create train and validation lists** — use the same dataset structure we would use for a real project.
5. **Run a dataset check** — make sure PaddleOCR can actually read the image paths and polygon labels before training.

| | |
| --- | --- |
| **OUTPUT** | Full NID images + text-region polygon labels in PaddleOCR detection format. |
| **DONE WHEN** | The detection dataset loads without format/path errors. |

**Deliverable status:** D2 progresses.

---

## Phase 04 — Prepare recognition training data

*Teach the recognizer what each text line actually says.*

### What you actually do

1. **Crop each labelled text line** — make one image crop per line/region.
2. **Type the exact correct text** — for every crop, write the exact Bangla, English or numeric transcription.
3. **Keep Bangla + English together** — we are training one recognizer, not separate language models.
4. **Create `train_label.txt` and `val_label.txt`** — each row should map a crop path to its correct text.
5. **Split by NID card** — all crops from the same physical card must stay in the same split.

| | |
| --- | --- |
| **OUTPUT** | Single-line crops + exact text labels for recognition training. |
| **DONE WHEN** | PaddleOCR can load the recognition dataset and every crop has a correct label. |

**Deliverable status:** D2 progresses.

---

## Phase 05 — Build one Bangla + English character dictionary

*Make sure the recognizer is allowed to output every character that appears on Bangladesh NIDs.*

### What you actually do

1. **Read all ground-truth labels** — scan the text we typed in Phase 4.
2. **Collect every unique character** — Bangla letters/signs, English letters, digits and needed punctuation.
3. **Preserve Bangla Unicode exactly** — do not simplify or manually rewrite vowel signs / conjunct sequences.
4. **Generate the file with a script** — avoid manually maintaining a long character list.
5. **Validate the dictionary** — fail the check if any character in train/validation labels is missing.

| | |
| --- | --- |
| **OUTPUT** | One bilingual character dictionary generated from the dataset. |
| **DONE WHEN** | Every labelled character is covered by the recognizer vocabulary. |

**Deliverable status:** D1 gains the bilingual-recognition method.

---

## Phase 06 — Fine-tune the recognition model first

*Prove the easier half of the training pipeline first: text crop → training → exported recognizer.*

### What you actually do

1. **Load pretrained `PP-OCRv5_server_rec` weights** — we are adapting an existing model, not training from zero.
2. **Point the config to our crops, labels and character dictionary** — use the Phase 4–5 files.
3. **Run a short training test** — with only 2–4 NIDs, the goal is simply to prove the code works.
4. **Confirm a checkpoint is created** — training should update weights and save model checkpoints.
5. **Evaluate, export and reload** — export an inference model and run it on a few crops to prove it is usable.

| | |
| --- | --- |
| **OUTPUT** | Fine-tuned + exported recognition model and a working recognition training script. |
| **DONE WHEN** | Training runs, a checkpoint is saved, export works, and the exported recognizer can read a crop. |

**Deliverable status:** D2 is half complete.

---

## Phase 07 — Fine-tune the detection model

*Now prove the full-card detector training path.*

### What you actually do

1. **Load pretrained `PP-OCRv5_server_det` weights** — again, adapt the existing model instead of starting from zero.
2. **Point the config to full NID images and polygon labels** — use the Phase 3 dataset.
3. **Run a short training test** — 2–4 NIDs are enough only to check the workflow.
4. **Confirm checkpoints are created** — verify that detector training actually runs and saves weights.
5. **Evaluate, export and reload** — run the exported detector on an NID and visualize the returned text boxes.

| | |
| --- | --- |
| **OUTPUT** | Fine-tuned + exported detector and a working detection training script. |
| **DONE WHEN** | The exported detector loads and returns text regions on an NID image. |

**Deliverable status:** D2 is nearly complete.

---

## Phase 08 — Use both fine-tuned models together

*Run one PaddleOCR pipeline with the custom detector and custom Bangla + English recognizer.*

### What you actually do

1. **Load the exported detector directory** — use the model from Phase 7.
2. **Load the exported recognizer directory** — use the model from Phase 6.
3. **Run the same 2–4 NIDs from Phase 2** — keep the comparison fair.
4. **Save boxes, text and confidence again** — use the same result format as the baseline.
5. **Check the complete pipeline** — confirm detection feeds recognition correctly and final OCR output is produced.

| | |
| --- | --- |
| **OUTPUT** | One end-to-end PaddleOCR pipeline using both fine-tuned models. |
| **DONE WHEN** | The same NID image goes through custom detection + custom recognition successfully. |

**Deliverable status:** D2 finishes here.

---

## Phase 09 — Compare before vs after

*Check what changed, but do not claim real accuracy from only 2–4 NIDs.*

### What you actually do

1. **Compare detection** — look for missed, merged, split or clipped text boxes.
2. **Compare recognition** — look at Bangla, English and numeric transcription separately.
3. **Calculate simple model metrics** — recognition: CER/edit distance; detection: precision/recall and IoU matching when possible.
4. **Report Bangla separately** — overall OCR can look good while Bangla is still poor.
5. **Write the limitation clearly** — this is a process test and a debugging comparison, not a trustworthy benchmark.

| | |
| --- | --- |
| **OUTPUT** | A small before/after technical comparison plus working evaluation code. |
| **DONE WHEN** | We can show what changed without overstating the result. |

**Deliverable status:** D1 and D4 gain evaluation material.

---

## Phase 10 — Decide how much real data is needed

*Turn the small demo into a realistic data-collection plan for serious fine-tuning.*

### What you actually do

1. **Treat 2–4 NIDs only as a smoke test** — they prove the pipeline, not the accuracy.
2. **Use a practical first target** — plan roughly 300–500 diverse NID cards and at least about 5,000 real recognition line crops as an initial serious dataset.
3. **Collect diversity, not copies** — cover different NID generations, Bangla names, addresses, English text, digits and hard visual cases.
4. **Check Bangla character coverage** — make sure common letters, signs and conjunct patterns are represented.
5. **Use a learning curve** — train with 25%, 50%, 75%, 100% of the data. If validation error is still dropping strongly, collect more.

| | |
| --- | --- |
| **OUTPUT** | A written data-size recommendation plus a method for deciding whether more data is still needed. |
| **DONE WHEN** | The estimate is supported by coverage and learning-curve logic, not by guessing. |

**Deliverable status:** D3 finishes here.

---

## Phase 11 — Finalize the documentation and repository

*Turn the experiment into a clean project that another engineer can understand and run.*

### What you actually do

1. **Clean the repo structure** — keep configs, scripts, requirements, docs and result templates organized.
2. **Keep real NID images outside Git** — use a private dataset path; do not commit identity documents.
3. **Write simple run instructions** — prepare data → train recognition → train detection → export → combined inference → evaluate.
4. **Document where outputs are saved** — checkpoints, exported models, logs and evaluation files.
5. **Add beginner troubleshooting** — wrong path, missing dictionary character, wrong config, checkpoint not loading, export path mismatch, etc.
6. **Make the docs match the scripts that actually worked** — remove any old/illustrative command that was not verified.

| | |
| --- | --- |
| **OUTPUT** | A reproducible fine-tuning repo and final step-by-step documentation. |
| **DONE WHEN** | A second engineer can clone the repo, point it to a private dataset and reproduce the workflow. |

**Deliverable status:** D1 and D4 finish here.

---

## Final completion map

*Where each requested deliverable is finished*

| Deliverable | Finished at |
| --- | --- |
| D1 — Documentation | Phase 11, after all steps and commands have been verified in practice. |
| D2 — 2–4 NID fine-tuning workflow | Phase 8, after recognition + detection training, export and combined inference work. |
| D3 — Data requirement | Phase 10, with an initial target plus learning-curve method. |
| D4 — Repository | Started in Phase 1 and finalized in Phase 11. |

> **Important:** For the 2–4 NID experiment, “success” does **not** mean high accuracy. Success means the whole process works: data loads → training runs → checkpoint saves → model exports → exported model reloads → combined PaddleOCR inference runs.
