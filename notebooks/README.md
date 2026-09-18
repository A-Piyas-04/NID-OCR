# Notebooks

Kaggle-oriented notebooks for the Bangladesh NID PaddleOCR fine-tuning workflow.

Upload / sync these notebooks to Kaggle. Attach the private NID dataset.
Do not train on a local GPU.

## Planned notebooks

| File | Phase | Purpose |
| --- | --- | --- |
| `01_phase01_environment.ipynb` | 01 | Verify Kaggle env, pin PaddleOCR, record configs |
| `02_baseline.ipynb` | 02 | Run stock PP-OCRv5 on 2–4 NIDs |
| `05_build_charset.ipynb` | 05 | Build / validate Bangla + English dictionary |
| `06_train_recognition.ipynb` | 06 | Fine-tune `PP-OCRv5_server_rec` |
| `07_train_detection.ipynb` | 07 | Fine-tune `PP-OCRv5_server_det` |
| `08_combined_inference.ipynb` | 08 | Run both fine-tuned models together |
| `09_evaluation.ipynb` | 09 | Before vs after comparison |

Notebooks are added as phases are executed.
