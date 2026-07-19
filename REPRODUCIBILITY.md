# Reproducibility Statement

## Evidence retained

The repository preserves the extracted dataset, portable configuration, 150 contiguous epoch records, restart history, optimizer summary, figures, `best.pt`, `last.pt`, source inventory, path-rewrite record and file manifest. It excludes the duplicate source ZIP and 150 intermediate checkpoints.

## Training state

The run used YOLO11s, 150 epochs, batch 8, image size 640 and seed 42. `optimizer=auto` selected AdamW with effective initial learning rate 0.000625; configured `lr0=0.01`, `lrf=0.01`, beta values 0.9/0.999 and weight decay 0.0005 are recorded separately. Training resumed between Epochs 80 and 81 from restored model and optimizer state. The published configuration records effective `patience=0`; the initial request used patience 30 before the no-early-stop full-run decision.

## Verification

Run `make test` for unit and contract tests, `make verify` for data, privacy, checkpoint and manifest checks, and `make infer-example` for CPU checkpoint loading plus one-image JSON/visual output. GitHub Actions intentionally uses `lfs: false` and runs lightweight tests only, avoiding automatic model/data bandwidth consumption.

Exact platform pins are in `requirements-*.lock`. Hardware-dependent floating-point variation may prevent bit-identical retraining, so reproducibility is defined as data/config/code/provenance completeness plus metric-compatible reruns, not guaranteed bitwise-identical weights.

