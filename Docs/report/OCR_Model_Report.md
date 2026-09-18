**TECHNICAL REPORT**
**NID-OCR**
The OCR Model, the Pipeline, its Limits,
and a Path to Fine-Tuning

| Field | Value |
| --- | --- |
| Subject | Bangladesh National ID card OCR — model & pipeline review |
| Repository | snd-ml/nid-ocr |
| Branch reviewed | develop (content-identical to bug/salekeen/nid-parser-fix) |
| Commit | 3a6e8aa |
| OCR engine | PaddleOCR 3.3.0 / PaddlePaddle 3.2.0 / PaddleX 3.3.3 |
| Model family | PP-OCRv5 (server detection + server recognition) |
| Execution target | CPU only (MKL-DNN, 8 threads) |
| Measured baseline | 96.8% mean field F1 over 65 labelled images (off-repo, 2026-02-05) |
| Date | 16 September 2026 |
| Author | Prepared for Salekeen |


**SCOPE**  This report describes what the code on the reviewed branch actually does, not what the in-repo docs claim. Where the two disagree, the discrepancy is called out explicitly — there are several, and they matter for anyone tuning this system.

# 1.  Executive Summary
The service is a FastAPI wrapper around an off-the-shelf PaddleOCR PP-OCRv5 pipeline, followed by a hand-written regex and heuristic layer that picks four fields out of the raw text: name, date of birth, NID number, and address. The models themselves have never been trained or adapted on Bangladesh NID data — every accuracy gain so far has come from post-processing the model's mistakes rather than from reducing them. That strategy has worked well, reaching 96.8% mean field F1, and it is now close to its ceiling.
**Four conclusions drive the rest of this document:**
1. **The model is generic, not domain-adapted.** It is the stock PP-OCRv5_server_det + PP-OCRv5_server_rec checkpoint pair downloaded from PaddleX. The only NID-specific knowledge in the system lives in Python regexes, not in weights.
1. **The parsing layer is compensating for recognition errors.** The DOB extractor carries a hard-coded table mapping Ocl→Oct, Jari→Jan, Feh→Feb and nine more. That table is a direct, measured inventory of where the recognition model fails on this card's font — and therefore the best possible argument for fine-tuning it.
1. **Bengali is configured but not readable.** Extractors match on ঠিকানা and পিতা, but the pipeline never passes a language setting to PaddleOCR and runs a Latin-oriented recognizer. Those branches are effectively dead code.
1. **A real baseline exists — but it lives outside the repository.** 65 labelled images, a scoring script and a scored comparison sit in ~/Downloads/server_images_nid/, uncommitted and unbacked-up. The current parser scores **96.8% mean field F1**, up from 87.7%. This is a genuine asset that the repository does not know about, and it is one `rm -rf` away from being lost.
**BOTTOM LINE**  The system is already accurate — 96.8% mean field F1 — so the question is not whether it works but where the remaining ~3% lives. Section 4 shows it is concentrated in the name field (93.4% F1), and that the residual errors are dominated by character confusion on the "MD." honorific plus one outright parser defect. Fine-tuning recognition is justified, but two of the six remaining name failures and both NID-number failures are fixable in Python today, for free. Do those first.

# 2.  The Model in Use
## 2.1  Identity and provenance
The engine is selected by name in configuration and instantiated once per process. The effective values on the reviewed branch, after the .env file overrides the Python defaults:
| Component | Value | Set in |
| --- | --- | --- |
| Pipeline version | PP-OCRv5 | config.py:38 |
| Text detection | PP-OCRv5_server_det | config.py:40, .env:27 |
| Text recognition | PP-OCRv5_server_rec | config.py:41, .env:28 |
| Doc orientation classifier | enabled (hard-coded True) | ocr_service.py:70 |
| Doc unwarping | disabled | ocr_service.py:71 |
| Text-line orientation | disabled | ocr_service.py:72 |
| Device | CPU — GPU flag is ignored | ocr_service.py:60-65 |
| CPU threads | 8 | config.py:48 |
| MKL-DNN | enabled | config.py:47 |
| HPI | disabled | config.py:51 |
| Recognition batch | 4 | config.py:49 |
| Detection box threshold | 0.3 | config.py:44 |
| Confidence cut-off | 0.3 | .env:30 |
| Max image dimension | 1080 px | .env:31 |


These are the stock public checkpoints. They are baked into the Docker image at build time so the container never needs network access at runtime:
# Dockerfile:30-33  (builder stage)
ENV PADDLEX_HOME=/home/appadmin/.paddlex
RUN python -c "from paddleocr import PaddleOCR; ocr = PaddleOCR(
        text_detection_model_name='PP-OCRv5_server_det',
        text_recognition_model_name='PP-OCRv5_server_rec',
        use_textline_orientation=False)"

# Dockerfile:94-95  (runtime stage) — pinned so PaddleX skips its network check
ENV OCR_DET_MODEL_DIR=/home/appadmin/.paddlex/official_models/PP-OCRv5_server_det
ENV OCR_REC_MODEL_DIR=/home/appadmin/.paddlex/official_models/PP-OCRv5_server_rec
*Models are image-resident, which is also the natural place to drop a fine-tuned checkpoint later.*
## 2.2  What the two models do
PaddleOCR splits the job in two, and the distinction matters when deciding what to fine-tune, so it is worth stating precisely.
| Stage | Job | Output | Fails as |
| --- | --- | --- | --- |
| Detection (server_det) | Find where text is. Predicts a per-pixel text/non-text map, then groups and expands it into quadrilateral boxes. | List of 4-point polygons | Missed lines; two words merged into one box; a box clipping the first or last character |
| Recognition (server_rec) | Read what each cropped box says. One forward pass per crop, decoded into a character sequence with a confidence score. | Text string + score per box | Character confusions — the O/0, l/1, t/l class of error visible in this repo's correction tables |


**ARCHITECTURE NOTE**  The detection head is a differentiable-binarization segmentation model and the recognizer is a sequence model over the crop; the specific backbones are upstream PaddleOCR implementation detail and are not pinned by this repository. Verify against the installed paddlex 3.3.3 before relying on architecture specifics for a training run.
The practical consequence: detection errors and recognition errors need different fixes and different training data. Section 7 treats them separately, and recommends starting with recognition.
## 2.3  "Server" versus "mobile" — an unresolved contradiction
The repository contains three mutually inconsistent statements about which model variant is in use:
| Source | Claims | Status |
| --- | --- | --- |
| config.py:40-41, .env:27-28, Dockerfile:94-95 | PP-OCRv5_server_det + PP-OCRv5_server_rec | Authoritative — this is what runs |
| docs/CPU_OPTIMIZATION_GUIDE.md | "Your project uses PP-OCRv5 Mobile Models" — mobile det + en_..._mobile_rec | Stale. Contradicted by the code |
| tests/quick_test_paddle_ocr.py | PP-OCRv5_mobile_det + en_PP-OCRv5_mobile_rec | Diverged. Tests a different model than production serves |


**ACTION**  The test script and the CPU guide should be brought in line with config.py, or the divergence documented deliberately. As it stands, a developer optimising against the test script is tuning a model the API does not use — and the benchmark numbers in the CPU guide (0.8–1.2 s/image at 480 px, mobile models) do not describe this deployment at all.
The switch was deliberate — commit 55584dd, "Switch to PP-OCRv5 server models for higher accuracy" — so the code is right and the docs simply were not updated with it. Server models are roughly an order of magnitude larger than mobile and materially slower on CPU, which is the tax being paid for accuracy here.

# 3.  The Processing Flow
One request through
POST /nid-ocr/extract, end to end. Front image is required; back image is optional.
HTTP POST  (multipart: nid_front, nid_back?)
      |
  [1] Middleware chain
      |     SecurityHeaders -> RequestLogging -> RateLimit -> PerformanceMonitor
      v
  [2] extract_nid_information()                      app/main.py:258
      |     read bytes, enforce MAX_FILE_SIZE
      v
  [3] OCRService.extract_text()                      ocr_service.py:204
      |     3a  extension allow-list + PIL verify()
      |     3b  SHA-256 cache probe          (disabled: ENABLE_CACHE=False)
      |     3c  decode -> convert to RGB
      |     3d  downscale longest side to 1080 px, BILINEAR
      |     3e  PIL -> numpy array
      v
  [4] PaddleOCR.predict(array)                       ocr_service.py:287
      |     doc-orientation classify -> detect boxes -> recognise crops
      |     returns rec_polys / rec_texts / rec_scores
      v
  [5] Confidence filter, score >= threshold          ocr_service.py:323-333
      |     -> list[OCRResult(text, confidence, bounding_box)]
      v
  [6] NIDParser.parse_nid_front() / _back()          nid_parser.py:30 / 76
      |     clean_text() on every line, then four extractors
      |     Name | DOB | NIDNumber   (front)     Address   (back)
      v
  [7] NIDExtractionResponse  (fields + raw_text + timing)
*The back image, when supplied, repeats steps 3–5 and then runs only the address extractor.*
## 3.1  Stage detail
### Steps 1–2 — intake
- Rate limiting is in-process and per-window: 100 requests / 60 s (config.py:71-73). With WORKERS=4 each worker keeps its own counter, so the real ceiling is roughly 4× the configured value.
- Size limit is enforced *after* the full upload is read into memory (main.py:288-293). A malicious 1 GB upload is buffered before it is rejected.
- Accepted formats: jpg, jpeg, png, bmp. Checked by file extension, then confirmed by PIL.Image.verify().
### Steps 3c–3d — preprocessing
This is thinner than it may appear. The entire image preparation is a colour-space conversion and an optional downscale:
# ocr_service.py:249-260
if image.mode != 'RGB':
    image = image.convert('RGB')

if settings.OCR_MAX_IMAGE_DIMENSION:          # 1080 from .env
    largest_side = max(width, height)
    if largest_side > max_dim:
        scale = max_dim / largest_side
        image = image.resize(new_size, Image.BILINEAR)
**NO ENHANCEMENT STAGE EXISTS**  There is no deskew, no contrast normalisation, no CLAHE, no sharpening, no glare or shadow handling, and no crop-to-card step. Every pixel defect in the uploaded photo reaches the model as-is. For phone-captured ID cards — the expected input — this is the single largest cheap win available, and it is strictly cheaper than fine-tuning.
### Step 5 — the confidence filter
# ocr_service.py:325
threshold = 0.3 if settings.DEBUG else settings.OCR_CONFIDENCE_THRESHOLD
The configured threshold is silently overridden whenever DEBUG is on. Because docker-compose.yml sets DEBUG=True, the local Docker environment ignores OCR_CONFIDENCE_THRESHOLD entirely. It happens to be harmless today — .env also sets 0.3, so the two agree — but any future tuning of this value will appear to have no effect under compose, which is a costly hour to lose.
### Step 6 — field extraction
Four independent extractors run over the cleaned text lines. None of them use the bounding-box geometry that step 5 carefully preserved; all of them work purely on the flat, reading-order list of strings.
| Extractor | Strategy | Notable mechanics |
| --- | --- | --- |
| Name `name.py | Fuzzy-match a line against the word "name" (Levenshtein ≤ 2), then collect the following lines | Two collection modes: split on ":" if present, else collect only ALL-CAPS lines — exploiting that BD NID names are uppercase while misread Bengali is mixed case. Strips trailing garbage tokens such as "CS", "GT3T" |
| Date of birth `dob.py | Five cascading strategies: keyword-anchored, sliding-window keyword, bare pattern, window-combined pattern, whole-text pattern | Two regexes — one for detached dates, one for letter-attached ("Bith13Sep1971"). A 12-entry month-correction table. Year sanity-checked to 1900…current |
| NID number `nid_number.py | Keyword-anchored digit extraction, searching the same line, then 3 lines forward, then 2 back | Pre-merges every adjacent line pair to survive split labels ("ID" + "NO:123"). Accepts only lengths 10, 13 or 17 |
| Address `address.py | Enter an "address section" on keyword, accumulate up to 6 lines, exit on any other field's keyword | Filters out dates and lines that are >70% digits. Falls back to spotting village/post/thana/district if no "Address:" label was found |


**DESIGN OBSERVATION**  Discarding the box coordinates is a significant loss. On a card with fixed layout, the y-position of a line is strong evidence for which field it belongs to — far more robust than fuzzy-matching a label the model may have misread. Re-introducing geometry into the parser is an independent improvement that needs no model change at all.

# 4.  The Measured Baseline
**LOCATED OUTSIDE THE REPOSITORY**  A labelled evaluation set, a test runner and a scored accuracy report exist at ~/Downloads/server_images_nid/ — 65 front images with field-level ground truth, plus run_comprehensive_test.sh and accuracy_report_final.txt dated 2026-02-05. None of it is committed to git, on any branch. This is the most valuable artefact in the project and it is the least protected.
## 4.1  Where the system stands
The report compares the then-production parser against the current bug-fix parser over all 65 images. These are the numbers to beat:
| Field | Production F1 | Current F1 | Change | Residual errors |
| --- | --- | --- | --- | --- |
| Name | 90.8% | 93.4% | +2.7 | 3 null, 5 wrong |
| Date of birth | 98.4% | 99.2% | +0.8 | 0 null, 1 wrong |
| NID number | 73.8% | 97.6% | +23.9 | 1 null, 2 wrong |
| Mean | 87.7% | 96.8% | +9.1 | — |


- **The NID number field carried the improvement** — F1 rose 24 points, almost entirely by cutting false positives from 25 to 2. The keyword-anchored extraction strategy did its job.
- **Name is now the weakest field** at 93.4%, and it is the only field where the fix *lost* recall — 98.2% down to 95.0%. The stricter heuristics traded three recoveries for five precision wins. That trade was worth it, but it is where the remaining headroom is.
- **Address is not measured at all.** The ground truth covers name, date of birth and NID number only — all front-side fields. The back-side address extractor has no ground truth, no score, and therefore no known accuracy.
## 4.2  What the residual failures actually are
This is the most decision-relevant table in the report. Every remaining failure, classified by probable cause:
| Image | Produced | Expected | Probable cause |
| --- | --- | --- | --- |
| nid_8 | MOZAKARIA HOSSAIN | MD. ZAKARIA HOSSAIN | Recognition: D→O confusion, plus lost word boundary |
| nid_44 | MO.SAZIBUR RAHMAN | MD. SAZIBUR RAHMAN | Recognition: D→O confusion |
| nid_49 | D.RAJIB HOSSAIN | MD. RAJIB HOSSAIN | Detection: leading M clipped from the box |
| nid_63 | MD SAIDURE RAHMANSAGOR | MD SAIDUR RAHMAN SAGOR | Recognition: spurious E; detection merged two words |
| nid_55 | MD. SHOHRAB | MD. SHOHRAB HOSSIN SOWRAB | Parser: name collection terminated early |
| nid_10, nid_15, nid_53 | null | (name present) | Parser: keyword or all-caps gate rejected the line |
| nid_8 | 0 Jan 1999 | 01 Jan 1999 | Leading digit lost — detection clipping or regex boundary |
| nid_9, nid_16 | 1994291563700 | 19942915637000 | Parser defect — confirmed, see below |
| nid_37 | null | 5103969043 | Unclassified |


**ATTRIBUTION CAVEAT**  These causes are inferred from the shape of each error, not proven — the stored results keep only the final field values, not the raw_text the model produced. Retaining raw_text in the harness output would make attribution exact instead of inferred, and costs nothing. That change should be the first thing done to the harness.
## 4.3  A confirmed parser defect worth fixing today
The two nid_9 / nid_16 failures are not model errors. The extractor only accepts NID numbers of length 10, 13 or 17:
# nid_number.py:14
VALID_LENGTHS = {10, 13, 17}
But the ground truth itself contains 14-digit numbers:
| NID length in ground truth | Count | Accepted by extractor? |
| --- | --- | --- |
| 10 digits | 54 | Yes |
| 13 digits | 5 | Yes |
| 14 digits | 2 | No — rejected |
| 17 digits | 4 | Yes |


Both 14-digit cases are the same number, 19942915637000, and in both the extractor returned a 13-digit truncation. The length allow-list did not cause a miss — it caused a **plausible-looking wrong answer**, which is considerably worse. A caller has no way to detect it.
**RECOMMENDATION**  Widen the allow-list to the range 10–17 and rank candidates rather than accept the first match, or drop the length gate entirely in favour of keyword anchoring plus a returned confidence. This is a few lines of Python that recovers 2 of the 3 remaining NID-number errors — no model work involved.
## 4.4  Reading the profile
Of the twelve residual failures, roughly four are recognition or detection errors at the character level, four are parser gating problems, two are the confirmed length-allow-list defect, and two are unclassified. Two conclusions follow:
- **Parser work is not exhausted.** Half the remaining errors are in Python, not in the weights. They are cheaper to fix and carry no training or data-governance cost.
- **The model errors are real and systematic.** The D→O confusion on the "MD." honorific appears independently in three images. That is not noise — it is a reproducible weakness on the single most common token on a Bangladesh NID, and it is precisely what recognition fine-tuning fixes.

# 5.  Limitations
Grouped by cause, since the fix differs. Each item cites the code that demonstrates it.
## 5.1  Model and domain fit
| # | Limitation | Evidence | Impact |
| --- | --- | --- | --- |
| M1 | The model has never seen a Bangladesh NID. Generic checkpoints on a specific card stock, font, security overlay and holographic background. | No training artefacts in repo history | High |
| M2 | Recognition demonstrably misreads the card's font. The month-correction table is a catalogue of real failures: Ocl, Jari, Feh, Nar, Aor, Mav, Juri, Ju1, Auq, Seo, Nou, Oec. | dob.py:36-49 | High |
| M3 | Bengali is unreadable in practice. Extractors match ঠিকানা / পিতা / মাতা / স্বামী, but no language is ever passed to PaddleOCR and the recognizer is Latin-oriented. Bengali name and address — the authoritative fields on a smart NID — are out of reach. | address.py:13, name.py:16 | High |
| M4 | OCR_LANG is dead configuration. Defined, logged, never passed to the engine. | config.py:37; ocr_service.py:117 only | Medium |
| M5 | No orientation robustness beyond the document classifier. Unwarping and text-line orientation are both off, so a rotated, skewed or perspective-distorted capture is fed in raw. | ocr_service.py:71-72 | Medium |
| M6 | Downscaling to 1080 px discards resolution on the smallest, most error-prone glyphs — exactly the digits of the NID number. | .env:31 | Medium |


## 5.2  Parsing layer
| # | Limitation | Evidence | Impact |
| --- | --- | --- | --- |
| P1 | Bounding boxes are captured and then thrown away. The parser is blind to layout. | nid_parser.py:51, 97 | High |
| P2 | Error correction is a hard-coded allow-list. Any OCR corruption not already enumerated falls straight through. This does not generalise and grows without bound. | dob.py:36-49; name.py:180-201 | High |
| P3 | No confidence is propagated to the caller per field. A field guessed from a 0.31-score line is indistinguishable from one read at 0.99. | schemas.py:17-28 | Medium |
| P4 | No checksum or format validation on the NID number beyond length ∈ {10, 13, 17}. A digit misread inside a correct-length number is undetectable. | nid_number.py:14 | Medium |
| P5 | The all-caps heuristic for names is brittle. A correctly-read mixed-case name, or a name whose first line the detector clipped, terminates collection early. | name.py:155-159 | Medium |
| P6 | DOB strategies 1–2 build large candidate sets with nested windowed loops and return the first regex hit — not the best or most likely one. | dob.py:67-107 | Low |


## 5.3  Engineering and operations
| # | Limitation | Evidence | Impact |
| --- | --- | --- | --- |
| E1 | Synchronous OCR inside an async endpoint. `extract_text() is CPU-bound and blocks the event loop for the whole inference, so one in-flight request stalls everything else that worker is serving — including /health. | main.py:258, 300 | High |
| E2 | DEBUG silently overrides the confidence threshold, and compose sets DEBUG=True. | ocr_service.py:325 | Medium |
| E3 | Cache is FIFO, not LRU despite the comment — insertion order is never refreshed on access. `CACHE_TTL_SECONDS is reported by /metrics but never enforced. Caching is off in .env regardless. | ocr_service.py:168-174, 407 | Low |
| E4 | Singleton construction is not guarded. Two threads entering `__new__ concurrently can both initialise the engine. | ocr_service.py:33-42 | Low |
| E5 | Upload is fully buffered before the size check. | main.py:288-293 | Medium |
| E6 | Docker healthcheck targets `/health, but the route is /nid-ocr/health. The container healthcheck can never pass as written. | Dockerfile:101 vs main.py:210 | Medium |
| E7 | OCR_USE_GPU is accepted and then warned away. No GPU path exists. | ocr_service.py:60-65 | Low |


## 5.4  The evaluation gap
Not the gap it first appears to be. A working harness and a scored baseline do exist (Section 4) — they are simply not part of the repository. The real limitations are these:
| # | Limitation | Impact |
| --- | --- | --- |
| V1 | The eval set, runner and reports live only in `~/Downloads/server_images_nid/. Uncommitted, un-backed-up, on one machine. Losing that directory loses the only measurement of the system that exists. | High |
| V2 | No regression test runs in CI. `.gitlab-ci.yml does not invoke the harness, so accuracy can silently regress between commits. | High |
| V3 | The harness stores only final field values, not the raw_text behind them. Failures cannot be attributed to detection, recognition or parsing without re-running by hand. | Medium |
| V4 | Address is unmeasured. Ground truth covers name, DOB and NID number only — all front-side. The back-side extractor's accuracy is entirely unknown. | High |
| V5 | 65 images is a thin set for 3-significant-figure claims, and its capture-condition distribution is undocumented. A 2.7-point F1 change over 65 samples is a handful of images moving. | Medium |
| V6 | Ground truth is field-level only. No line-level transcriptions with boxes exist — which is exactly the format recognition fine-tuning needs. | Medium |


The parsing layer was clearly tuned by hand against these specific images — the correction tables prove it — and to the team's credit that tuning was scored rather than guessed at. The work is sound; it is the custody of it that is fragile.

# 6.  Do These First — Cheaper Than Fine-Tuning
Fine-tuning is expensive in effort, data and review. Several of the limitations above are significantly cheaper to fix and may close enough of the gap to change what — or whether — you need to train. Do them first, and measure after each.
| Order | Action | Why it pays | Effort |
| --- | --- | --- | --- |
| 1 | Commit and harden the existing harness (§7.1) | It works, but lives on one laptop and is not in CI. Everything below is judged by it. | M |
| 2 | Widen `VALID_LENGTHS to accept 14-digit NID numbers (§4.3) | Confirmed defect producing silently truncated, wrong-but-plausible numbers. Recovers 2 of 3 residual NID errors for a few lines of Python. | S |
| 3 | Handle `MO./`MD`/D. honorific variants in the name normaliser | Three of the five wrong names are D→O confusion on "MD." that the existing PREFIXES list does not cover. Cheaper than fine-tuning and fixes the same errors. | S |
| 4 | Add a preprocessing stage: deskew, CLAHE or contrast normalisation, card detection and crop, adaptive sharpening | Attacks M5/M6 directly. Standard document-OCR practice and currently entirely absent. Often worth several accuracy points on phone captures. | M |
| 5 | Stop downscaling to 1080 px, or raise it — and measure the accuracy/latency trade | The NID digits are the smallest glyphs on the card and the highest-value field. Currently the value is set without any measurement behind it. | S |
| 6 | Use the bounding boxes in the parser (P1) | Layout is fixed and highly informative. Likely the largest single accuracy gain available without touching the model. | M |
| 7 | Return per-field confidence (P3) | Lets the caller route low-confidence extractions to manual review — turning a silent wrong answer into a flagged one. | S |
| 8 | Move inference off the event loop (E1) | Correctness and throughput under concurrency. `run_in_threadpool or a process pool. | S |
| 9 | Fix the Docker healthcheck path (E6) and the DEBUG threshold override (E2) | Both are small, unambiguous bugs with operational consequences. | S |
| 10 | Reconcile the docs and test script with the server-model reality (§2.3) | Prevents further tuning against the wrong model. | S |


**SEQUENCING**  Items 1-3 are quick and should land together: the harness in CI, plus the two confirmed parser defects fixed. Re-score. Then add preprocessing (item 4) and re-score again. If the remaining errors are still dominated by character confusion on the card's font - as Section 4.2 suggests they will be - fine-tuning is clearly justified, and you have the measured before-and-after to prove its value.

# 7.  How to Approach Fine-Tuning
A concrete, staged plan. The ordering is deliberate: measurement, then data, then the smallest model that addresses the observed errors, then deployment. Skipping to the training step is the usual way these projects fail.
## 7.1  Stage 0 — Harden the measuring stick
You are not starting from zero here — Section 4's harness already works. The task is to make it a committed, trustworthy asset rather than a directory on one laptop.
**Deliverable: a versioned harness in CI, and a richer ground truth.**
1. **Move the harness into the repository.** Commit the runner and the scoring logic under tests/ or eval/. The images and ground truth cannot be committed — they are real identity documents — so point the harness at a configurable dataset path and store the data in access-controlled storage with a documented retention policy. Fixes V1.
1. **Wire it into CI as a regression gate.** Even a nightly run against a stored dataset beats nothing. Fail the build if mean field F1 drops below the committed baseline. Fixes V2.
1. **Retain raw_text in the harness output.** One-line change, and it converts Section 4.2's inferred attributions into exact ones. Do this before any training work, because it is what tells you whether training is the right lever. Fixes V3.
1. **Add address ground truth.** The back-side extractor is completely unmeasured, and address is the messiest field — free-form, multi-line, partly Bengali. It is plausibly the worst-performing field in the system and nobody would know. Fixes V4.
1. **Grow and stratify the set.** Target 300–500 fronts and a matching set of backs, sampled deliberately across phone and scanner, glare and shadow, old laminated and new smart cards, skewed and straight. Record the distribution. Fixes V5.
1. **Add line-level labels with boxes.** Field-level ground truth measures the product; line-level transcriptions with polygons are what diagnose the model — and they are the training data for Stage 2. Fixes V6.
1. **Add the missing metrics.** The existing report gives per-field precision/recall/F1, which is the right shape. Add normalised edit distance per field — it distinguishes "one character wrong" from "completely wrong", which exact-match F1 cannot — plus character error rate on the transcriptions and detection precision/recall at IoU 0.5.
**PRIVACY**  This dataset is a corpus of real national identity documents. It needs the same handling as the production data it came from: access-controlled storage, no commits to git, no third-party annotation tooling without review, a documented retention policy, and a lawful basis for using customer submissions as training data. Settle this before collection starts, not after.
## 7.2  Stage 1 — Decide what to train
Section 4.2 already provides a first-pass attribution. For the profile it shows:
| If failures are mostly… | Then | Rationale |
| --- | --- | --- |
| Character confusions on correctly-located text (O/0, l/1, t/l, month names) | Fine-tune recognition only — `PP-OCRv5_server_rec | Cheapest, fastest, highest expected return. Crops are small, so training is quick and a few thousand labelled lines go a long way. This is where the repo's correction tables point. |
| Missed, merged or clipped text lines | Fine-tune detection — `PP-OCRv5_server_det | More labelling effort (polygons, not strings) and slower to train. Try preprocessing and `det_db_box_thresh / `det_db_unclip_ratio tuning first — they are free. |
| Correct text, wrong field assignment | Do not train. Fix the parser. | This is P1/P2/P5 and four of the twelve residual failures. A better model will not help; using layout geometry will. |
| Bengali script entirely | A separate track: a Bengali-capable recognizer | This is a different model and a different dataset, not a tweak. Scope it on its own once the Latin path is solid. |


**RECOMMENDATION**  Start with recognition only. It is the smallest change with the clearest evidence behind it, and it lets you validate the whole train-export-deploy loop on the easy case before attempting detection.
## 7.3  Stage 2 — Prepare training data
For recognition fine-tuning, PaddleOCR expects cropped single-line images plus a tab-separated label file and a character dictionary.
dataset/
  train/  img_00001.jpg  img_00002.jpg  ...      # single text-line crops
  val/    ...
  train_label.txt
  val_label.txt
  bd_nid_dict.txt

# train_label.txt  — <relative path>\t<ground-truth string>
train/img_00001.jpg\tMD. RASHED KHAN
train/img_00002.jpg\t13 Apr 1952
train/img_00003.jpg\t2716031329835
*Crops are generated from the Stage 0 line-level labels by cutting each ground-truth polygon out of the source image.*
- **Restrict the dictionary.** BD NID front fields use a narrow character set: A–Z, digits, period, space, and the twelve month abbreviations. A tight bd_nid_dict.txt makes many of the observed confusions structurally impossible — the model cannot emit a character that is not in its vocabulary.
- **Augment realistically.** Mild rotation, blur, JPEG compression, brightness and contrast jitter, and synthetic glare. Match the augmentation to your actual capture conditions; do not add distortions you never see.
- **Consider synthetic pre-training.** If the card's font can be identified or approximated, rendering large volumes of synthetic labelled lines is a cheap way to bulk out a small real dataset. Always validate on real images only.
- **Split by card, not by crop.** Multiple crops from the same physical card must not straddle the train/val boundary, or your validation score will be optimistic.
## 7.4  Stage 3 — Run the fine-tune
PaddleOCR 3.x routes training through PaddleX. The shape of the workflow is: take the config for the module you are training, point it at your dataset, and start from the pretrained weights rather than from scratch.
# 1. Verify the training entrypoint and config path for your installed version
pip show paddlex        # 3.3.3 in this project
python -c "import paddlex, os; print(os.path.dirname(paddlex.__file__))"

# 2. Validate the dataset against the module's expected format
python main.py -c paddlex/configs/modules/text_recognition/PP-OCRv5_server_rec.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=/data/bd_nid_rec

# 3. Fine-tune from the pretrained checkpoint (low LR, short schedule)
python main.py -c paddlex/configs/modules/text_recognition/PP-OCRv5_server_rec.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=/data/bd_nid_rec \
    -o Train.pretrain_weight_path=<path-to-PP-OCRv5_server_rec-weights> \
    -o Train.epochs_iters=<short> \
    -o Train.learning_rate=<1e-4 or lower>

# 4. Evaluate, then export to an inference directory
python main.py -c <same config> -o Global.mode=evaluate
python main.py -c <same config> -o Global.mode=export
*Illustrative. Confirm exact keys and the entrypoint against the installed paddlex 3.3.3 — the CLI surface has moved between PaddleOCR 2.x and 3.x.*
**VERIFY BEFORE YOU BUDGET**  The command shapes above reflect the PaddleX module-training workflow, but option names and the entrypoint script vary by version. Spend the first hour confirming them against the installed package and the matching upstream docs rather than trusting this snippet — and pin whatever you find, because it will change again.
**Practical guidance for the run itself:**
- **Always start from pretrained weights.** Training PP-OCRv5 from scratch on a few thousand NID lines would be far worse than the stock model. You are adapting, not building.
- **Use a low learning rate and a short schedule.** Fine-tuning on a narrow domain overfits quickly. Watch validation CER per epoch and stop when it turns.
- **Keep a small slice of generic text in the training mix** if you want the model to stay usable on anything other than NID cards. Pure-NID training will specialise it hard.
- **Train on GPU, deploy on CPU.** The deployment target here is CPU-only, but there is no reason to train that way. Export and then measure CPU latency separately — adaptation should not change inference cost much, since the architecture is unchanged.
## 7.5  Stage 4 — Deploy the checkpoint
This repository is already structured to accept a custom model directory, which makes deployment genuinely straightforward — this is the part that needs no new engineering.
# The existing override path, ocr_service.py:97-108
if settings.OCR_REC_MODEL_DIR:
    ocr_params['text_recognition_model_dir']  = settings.OCR_REC_MODEL_DIR
    ocr_params['text_recognition_model_name'] = settings.OCR_REC_MODEL

# So shipping a fine-tuned recognizer is a Dockerfile change plus one env var:
COPY --chown=appadmin:appadmin ./models/bd_nid_rec_v1 \
     /home/appadmin/.paddlex/official_models/bd_nid_rec_v1
ENV OCR_REC_MODEL_DIR=/home/appadmin/.paddlex/official_models/bd_nid_rec_v1
*Version the checkpoint directory name. You will want to A/B against the stock model and roll back cleanly.*
- **Keep the model name consistent** with what the directory actually contains. The comment at ocr_service.py:82-83 warns that a mismatched name causes PaddleOCR to fall back to server defaults — a silent, easy-to-miss failure.
- **Re-run the Stage 0 harness against the new checkpoint** before shipping, and record the score alongside the model version.
- **Expect the parser to need retuning.** If the fine-tuned recognizer stops producing "Ocl" and "Feh", the correction tables become dead weight — and any parser heuristic that quietly depended on a specific misread may break. Retest the parsing layer, do not assume it is independent.
- **Retire compensation code deliberately.** Removing correction entries is how you convert a model improvement into a maintainability improvement. Do it with the harness green, not on faith.
## 7.6  Suggested milestones
| Milestone | Deliverable | Exit criterion |
| --- | --- | --- |
| M0  Secure the baseline | Harness committed, dataset in controlled storage, CI regression gate, raw_text retained, address ground truth added | 96.8% reproduced from a clean checkout by someone other than its author |
| M1  Cheap wins | Preprocessing stage, resolution decision, geometry-aware parser, per-field confidence | Measured improvement over M0, with the failure profile re-attributed |
| M2  Training data | Recognition crops, labels, restricted dictionary, card-level train/val split, augmentation pipeline | Dataset validation passes; split verified leak-free |
| M3  Fine-tuned recognizer | Adapted `PP-OCRv5_server_rec, exported for inference, versioned | Validation CER beats stock by a margin that survives the held-out set |
| M4  Deployed | Checkpoint baked into the image, env var wired, A/B against stock, rollback path tested | Field accuracy improvement confirmed in production conditions; CPU latency within budget |
| M5  Consolidate | Retire obsolete correction tables, regression tests, document the model version and its score | Parsing layer simpler than before, harness still green |


# Appendix A  ·  File Reference
| File | Lines | Role |
| --- | --- | --- |
| app/services/ocr_service.py | 418 | Engine init, preprocessing, predict, confidence filter, cache |
| app/services/nid_parser.py | 123 | Orchestrates the four extractors; drops box geometry |
| app/services/extractors/name.py | 232 | Fuzzy "name" match, all-caps collection, garbage-suffix stripping |
| app/services/extractors/dob.py | 183 | Five date strategies, two regexes, month-correction table |
| app/services/extractors/nid_number.py | 116 | Keyword-anchored digits, adjacent-line merge, length check |
| app/services/extractors/address.py | 125 | Section accumulation with field-keyword exits |
| app/services/extractors/base.py | 40 | clean_text, `has_keyword |
| app/config.py | 93 | Pydantic settings; model names and OCR tuning knobs |
| app/main.py | 442 | FastAPI app, endpoints, exception handlers |
| app/middleware.py | 276 | Logging, rate limit, security headers, perf monitor |
| app/schemas.py | 123 | Request/response models |
| Dockerfile | 103 | Two-stage build; bakes PP-OCRv5 server models into the image |
| docs/CPU_OPTIMIZATION_GUIDE.md | — | Stale: describes mobile models (see §2.3) |
| tests/quick_test_paddle_ocr.py | 51 | Diverged: tests mobile models |


# Appendix B  ·  Discrepancies Found
Collected for convenience. None of these are opinions about design; each is a place where two parts of the repository disagree with each other, or where configuration has no effect.
| Discrepancy | Location |
| --- | --- |
| Docs and test script say mobile models; code and Dockerfile use server models | CPU_OPTIMIZATION_GUIDE.md, quick_test_paddle_ocr.py vs config.py:40-41 |
| OCR_LANG is defined and logged but never passed to PaddleOCR | config.py:37, ocr_service.py:117 |
| OCR_USE_GPU is accepted, then warned away; no GPU path exists | ocr_service.py:60-65 |
| CACHE_TTL_SECONDS is exposed via /metrics but never enforced | ocr_service.py:407 |
| Cache described as LRU in the comment; implemented as FIFO | ocr_service.py:168-174 |
| DEBUG silently overrides `OCR_CONFIDENCE_THRESHOLD; compose sets DEBUG=True | ocr_service.py:325, docker-compose.yml |
| Docker healthcheck polls `/health; the route is `/nid-ocr/health | Dockerfile:101 vs main.py:210 |
| .env sets max dimension 1080; `config.py default is None; CPU guide recommends 480 | three-way disagreement |
| Test script disables doc-orientation classify; the service hard-codes it on | quick_test_paddle_ocr.py vs ocr_service.py:70 |
