# Saftey Vision

Saftey Vision uses ImageNet-pretrained MobileNetV2 to classify detected face/head crops and OpenCV for image, video, and webcam processing. Mask and helmet tasks use separate models and datasets.

## Project files

- `src/baseline_classifier.py` — MobileNetV2 training, evaluation, and single-crop prediction
- `src/opencv_inference.py` — YOLO helmet detections, OpenCV mask proposals, compliance counts, and video processing
- `src/camera_demo.py` — OpenCV image, video, and live-webcam inference
- `src/app.py` — Streamlit image/video/webcam-snapshot UI
- `src/event_store.py` — SQLite violation records, local evidence photos, and retention cleanup
- `src/prepare_voc_dataset.py` — Pascal VOC annotation conversion and source-image splits
- `src/prepare_yolo_dataset.py` — Pascal VOC bounding-box conversion for detector training
- `src/kaggle_dataset.py` — KaggleHub dataset downloader
- `src/train.py` and `src/inference.py` — optional YOLO detection workflow
- `data/` — downloaded data and converted train/validation/test crops
- `models/` — trained checkpoints and evaluation metadata

## Install

```powershell
python -m pip install -r requirements.txt
```

## Data

The project uses these public-domain Kaggle datasets (both dataset pages list CC0/Public Domain):

- [Face Mask Detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection): 853 images, 3 classes, Pascal VOC XML boxes
- [Helmet Detection](https://www.kaggle.com/datasets/andrewmvd/helmet-detection): 764 images, 2 classes, Pascal VOC XML boxes

Downloads have been placed in `data/raw/`. To fetch them again:

```powershell
python src\kaggle_dataset.py --slug andrewmvd/face-mask-detection
python src\kaggle_dataset.py --slug andrewmvd/helmet-detection
```

Convert annotations to classifier crops with image-level 70/20/10 train/validation/test splits:

```powershell
python src\prepare_voc_dataset.py --task mask --source data\raw\face-mask-detection --output data\classification\mask
python src\prepare_voc_dataset.py --task helmet --source data\raw\helmet-detection --output data\classification\helmet
```

The train, validation, and test splits are grouped by original source image, preventing crops from a single image appearing in multiple splits.

## Train and evaluate

```powershell
python src\baseline_classifier.py --task mask --data-root data\classification --epochs 15 --batch-size 32
python src\baseline_classifier.py --task helmet --data-root data\classification --epochs 15 --batch-size 32
```

Training starts with the pretrained MobileNetV2 classifier head frozen, then fine-tunes the backbone. Train crops receive flips, small rotations, brightness/contrast jitter, and occasional blur. The script selects by validation loss and reports test accuracy, per-class precision/recall/F1, and confusion matrix. The test set is evaluated only after training.

To compare against randomly initialized MobileNetV2, add `--scratch`; scratch runs save separately under `models/<task>_mobilenetv2_scratch/`.

Current bundled checkpoints were short three-epoch starter runs:

| Task | Test accuracy | Violation-class recall |
|---|---:|---:|
| Mask | 90.2% | `without_mask`: 94.7%; `mask_worn_incorrectly`: 77.8% |
| Helmet | 91.3% | `no_helmet`: 94.9% |

The included checkpoint files were produced before the latest augmentation changes, in three-epoch starter runs without augmentation. These are crop-classification scores on small public datasets—not object-detection mAP or evidence of deployment readiness. Retrain for longer, inspect errors, and evaluate on consented camera-specific data before relying on results.

## Run image, video, or live webcam inference

Live webcam (press `q` to exit):

```powershell
python -m src.camera_demo --task mask --source 0
```

For video or image files, pass the file path; annotated output is written beside the source by default:

```powershell
python -m src.camera_demo --task helmet --source data\sample.mp4
python -m src.camera_demo --task mask --source data\sample.jpg --output output\annotated.jpg
```

For each no-mask/no-helmet detection, the system saves one padded **face crop** (mask task) or **head crop** (helmet task), not the full frame. Lower-confidence red violation candidates (at least 25%) are also saved and marked as review candidates in the source field; only detections at the fixed 95% cutoff count as confirmed. Each crop has its own database row with UTC timestamp, task, class, confidence, source, evidence type, and relative photo path in `data/events.sqlite3`. Photos are stored under `evidence/` and automatically removed from disk with their database records after 30 days. During video/webcam processing, overlapping detections are tracked and one evidence crop is saved per violation appearance; a new photo can be captured after that person is absent for 8 seconds. Duplicate overlapping proposals are suppressed. Older full-frame evidence from earlier versions is automatically removed during the database migration. Use `--no-save-evidence` in the CLI to disable both persistence actions.

The confidence threshold is fixed at **95%** in both the dashboard and CLI; it is not user-adjustable.

Helmet localization prefers the latest trained YOLO detector under `models/helmet_detector*/weights/best.pt`. To create or retrain it from the bundled Pascal VOC helmet annotations:

```powershell
python -m src.prepare_yolo_dataset --task helmet --source data\raw\helmet-detection --output data\helmet_detector
python src\train.py --task helmet --data data\helmet_detector --epochs 15 --imgsz 416 --batch-size 8
```

The converter groups train/validation/test splits by source image. Boxes are red for predicted violations and green for predicted compliant detections. Detections at 95% are confirmed; lower-confidence boxes are review suggestions and do not count as confirmed violations. Red review crops are saved as review candidates so they can be checked later. If no candidate is found, the result is inconclusive rather than confirmation of compliance. Without trained YOLO weights the app retains its OpenCV/MobileNetV2 fallback.

The current local detector was fine-tuned for five additional CPU epochs. On its 77-image held-out test split it reached 64.5% mAP@0.5 at the detector's low evaluation cutoff; the fixed 95% cutoff produced no confirmed boxes on that split. Review boxes therefore remain important, and this starter model is not suitable for unattended enforcement.

The Streamlit sidebar also has a **Capture violation evidence** toggle (on by default). The evidence vault displays saved photos and allows individual records/photos to be deleted. The CLI captures evidence by default; pass `--no-save-evidence` to disable it.

## Streamlit UI

The repository includes `packages.txt` with Linux runtime libraries needed by OpenCV on Streamlit Community Cloud. OpenCV requires NumPy below 2.3 for the supported 4.12 wheels.

```powershell
streamlit run src/app.py --server.address 127.0.0.1
```

The UI supports image upload, annotated video upload/download, and webcam snapshots. Use the CLI webcam command for continuous live processing.

To deploy, connect this GitHub repository to Streamlit Community Cloud, select branch `main` and app file `src/app.py`. Each push to `main` triggers a redeploy.

## Local privacy and evidence handling

- Processing, SQLite, and evidence photos stay on the local machine; the app does not require a hosted backend service.
- Photos show only a padded face/head crop with a small class/confidence header. No face recognition or identity labels are produced.
- Retention cleanup runs whenever the dashboard executes or the CLI initializes the local database; individual items can also be deleted in the Evidence Vault.
- Use only with appropriate notice/consent and access controls. Keep the local `evidence/` folder and `data/events.sqlite3` private.

## Detection limitations

- Mask candidates come from OpenCV's frontal-face Haar cascade.
- If detector weights are not available, helmet checks fall back to generic OpenCV face/person proposals and crop classification. Small, side-facing, occluded, or distant people can be missed.
- The fixed 95% cutoff is intentionally conservative and may leave candidates unconfirmed. Manually review red/green boxes marked REVIEW and validate on representative camera images.
- Classifier accuracy is not mAP. The optional YOLO workflow is separate and requires YOLO-format bounding-box labels.
- No face recognition is performed. Media is processed locally; only explicit outputs and opt-in violation logs are written to disk.
