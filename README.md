# Archery Target and Arrow Detection with YOLO11s

[简体中文](README.zh-CN.md) · [Model card](MODEL_CARD.md) · [Dataset card](DATASET_CARD.md) · [Results](RESULTS.md) · [Reproducibility](REPRODUCIBILITY.md)

An evidence-oriented, reproducible release of a 150-epoch YOLO11s detector for archery targets and score regions. The repository contains the extracted Target and Arrow Detection v6 dataset, training and evaluation code, Streamlit demo, final checkpoints, all 150 epoch records, figures, and held-out test reports.

**Authors:** Zhengyang Wang and Jiacheng Yao, Beijing Normal–Hong Kong Baptist University (BNBU).

## Scope and claim boundaries

- This is an independent YOLO11s reproduction on Target v6; it is **not the original Roboflow checkpoint**.
- Validation is used for model selection. Test results are reported separately and are not substituted by validation results.
- Web-graph readings are exploratory comparisons only, not formal reproduction evidence.
- A reported score point is the predicted **bounding-box center**, not a physically annotated arrow-impact point.
- The reset in the `time` column at Epoch 81 records a resumed training process; model and optimizer state were restored rather than reinitialized.

## Main results

| Evaluation protocol | Split / selection | Precision | Recall | F1 | mAP50 | mAP50–95 |
|---|---|---:|---:|---:|---:|---:|
| Best validation checkpoint | validation, Epoch 142 | 73.94% | 78.58% | — | 80.12% | 43.65% |
| Standard AP evaluation | independent test, confidence floor 0.001, NMS IoU 0.70 | 78.11% | 67.27% | — | 75.90% | 41.31% |
| Locked operating point | independent test, `conf=0.25`, `iou=0.50` | 78.16% | 67.23% | 72.29% | 67.50%* | 37.74%* |

\* At the locked operating point, AP was recomputed after confidence filtering and is therefore not directly comparable to standard AP. See [RESULTS.md](RESULTS.md).

## Visual evidence

![Training dynamics from the canonical 150-row log](docs/assets/visuals/training_dynamics.png)

The raw log shows continued optimization across all 150 epochs, a resume boundary at Epoch 81, and the best validation mAP50–95 at Epoch 142. Model selection, standard AP test and the locked deployment operating point remain separate protocols.

![Real local inference using the published best.pt](docs/assets/visuals/local_model_inference_streamlit.png)

This qualitative local run by Zhengyang Wang uses `confidence=0.25` and `NMS IoU=0.50`. It demonstrates full-target detection (`target 0.94`) and also makes the dense-overlap/readability limitation visible. It does not replace the 65-image independent test, and predicted box centers are not physical impact points. See the [Detailed visual analysis](VISUAL_ANALYSIS.md) for the model, dataset, training, evaluation and diagnostic figures.

## Dataset

Target and Arrow Detection v6 contains 1,645 images at \(640\times640\) with paired YOLO labels:

| Split | Images | Labels |
|---|---:|---:|
| Train | 1,482 | 1,482 |
| Validation | 98 | 98 |
| Test | 65 | 65 |

The 12 class indices are `0, 1, 10, 2, 3, 4, 5, 6, 7, 8, 9, target`. The source is [Roboflow Universe, Target and Arrow Detection v6](https://universe.roboflow.com/archery-zrbei/target-and-arrow-detection/dataset/6), attributed to its original provider and retained under its MIT dataset metadata. The extracted data are tracked with Git LFS; the duplicate 187 MiB source ZIP is omitted and recorded by SHA-256 in the dataset card.

## Architecture and training

YOLO11s uses a convolutional backbone to extract features, a multi-scale neck to combine fine and semantic information, and a decoupled detection head to predict classes and boxes at multiple resolutions. Training used 150 epochs, batch size 8, image size 640, seed 42, AdamW selected by Ultralytics `optimizer=auto`, and the effective initial learning rate 0.000625. The requested `lr0=0.01` is a configured value, not the optimizer's effective initial learning rate under `auto`.

The run resumed between Epochs 80 and 81. The published `best.pt` was selected at Epoch 142 by validation mAP50–95. Intermediate epoch checkpoints are intentionally excluded; `best.pt`, `last.pt`, configurations, optimizer summary, plots and the full 150-row `results.csv` are retained.

## Installation

Python 3.12 is required. Git LFS must be installed before cloning.

```bash
git lfs install
git clone https://github.com/Yang1107-wzy/archery-yolo11s-detection.git
cd archery-yolo11s-detection
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[demo,dev]'
make verify
```

Dependency pins describe the verified macOS environment. CUDA users may need a platform-specific PyTorch installation before installing the project.

## Usage

Training is guarded against accidental full reruns:

```bash
python -m archery_ml.cli train --config configs/train/yolo11s_target_v6_150e.yaml --smoke
ARCHERY_RUN_FULL=1 python -m archery_ml.cli train --config configs/train/yolo11s_target_v6_150e.yaml
```

Inference writes JSON predictions and annotated images:

```bash
python -m archery_ml.cli infer \
  --model models/yolo11s-target-v6/best.pt \
  --source data/target-arrow-detection-v6/test/images \
  --output outputs/example \
  --device cpu
```

Validation threshold selection and one-time locked test evaluation:

```bash
python -m archery_ml.cli evaluate-validation --model models/yolo11s-target-v6/best.pt --data data/target-arrow-detection-v6/data.yaml --output outputs/validation.json --device cpu
python -m archery_ml.cli evaluate-test --model models/yolo11s-target-v6/best.pt --data data/target-arrow-detection-v6/data.yaml --validation outputs/validation.json --output outputs/test.json --device cpu
```

Launch the local demo with `make demo`. Convenience targets include `make test`, `make verify`, and `make infer-example`.

## Repository map

| Path | Purpose |
|---|---|
| `data/target-arrow-detection-v6/` | Extracted dataset and source metadata |
| `models/yolo11s-target-v6/` | `best.pt`, `last.pt`, hashes and model documentation |
| `training/150epoch-seed42/` | Full epoch log, configuration, restart history and figures |
| `evaluation/` | Standard AP, locked-point and error-analysis evidence |
| `src/`, `scripts/`, `app/` | Training, inference, evaluation and Streamlit application |
| `provenance/`, `release/` | Sanitized origin records and cryptographic manifest |

## Limitations

The public splits may contain visually related or offline-augmented variants across splits, so the test results should not be interpreted as subject-, venue-, or capture-session-independent generalization. The dataset is small and heterogeneous, class frequency is imbalanced, and the detector has not been calibrated for safety-critical or competition judging. A physical impact-point system requires dedicated point annotations and geometric calibration.

## License and citation

Project code and fine-tuned weights are released under the GNU Affero General Public License v3.0. Dataset files retain the source dataset's MIT terms and attribution. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [CITATION.cff](CITATION.cff). No ORCID, email, or DOI is asserted.
