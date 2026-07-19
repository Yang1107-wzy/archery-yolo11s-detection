from __future__ import annotations

import csv
import json
from pathlib import Path

from tools.sync_release import sync_release


def _write_results(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "metrics/mAP50(B)", "metrics/mAP50-95(B)"],
        )
        writer.writeheader()
        for epoch in range(1, 151):
            writer.writerow(
                {
                    "epoch": epoch,
                    "metrics/mAP50(B)": epoch / 200,
                    "metrics/mAP50-95(B)": 0.9 if epoch == 142 else epoch / 1000,
                }
            )


def _source_fixture(root: Path) -> None:
    dataset = root / "data/raw/target_arrow_v6"
    for split in ("train", "valid", "test"):
        (dataset / split / "images").mkdir(parents=True)
        (dataset / split / "labels").mkdir(parents=True)
        (dataset / split / "images" / f"{split}.jpg").write_bytes(b"jpg")
        (dataset / split / "labels" / f"{split}.txt").write_text(
            "0 0.5 0.5 0.1 0.1\n", encoding="utf-8"
        )
    (dataset / "README.dataset.txt").write_text("License: MIT\n", encoding="utf-8")
    (dataset / "README.roboflow.txt").write_text("Roboflow export\n", encoding="utf-8")
    (dataset / "data.yaml").write_text("train: ../train/images\n", encoding="utf-8")

    model = root / "artifacts/models/replica_target_v6/default_aug_seed42"
    model.mkdir(parents=True)
    (model / "best.pt").write_bytes(b"best")
    (model / "last.pt").write_bytes(b"last")

    run = root / "artifacts/training_runs/replica_20260719_002149_184e8ce2_full"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "epoch0.pt").write_bytes(b"must-not-publish")
    (run / "args.yaml").write_text(f"data: {root}/data/raw/target_arrow_v6/data.yaml\n", encoding="utf-8")
    (run / "results.png").write_bytes(b"plot")
    _write_results(run / "results.csv")

    manifest = root / "artifacts/run_manifests"
    manifest.mkdir(parents=True)
    (manifest / "replica_20260719_002149_184e8ce2_full.json").write_text(
        json.dumps(
            {
                "checkpoint": str(model / "best.pt"),
                "last_checkpoint": str(model / "last.pt"),
                "dataset": str(root / "data/raw/target_arrow_v6/data.yaml"),
                "config_path": str(root / "configs/train/config.yaml"),
            }
        ),
        encoding="utf-8",
    )
    config = root / "configs/train"
    config.mkdir(parents=True)
    (config / "replica_target_v6_default_aug.yaml").write_text("epochs: 150\n", encoding="utf-8")


def test_sync_release_copies_whitelist_and_omits_intermediate_checkpoints(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "release"
    _source_fixture(source)

    summary = sync_release(source, destination, include_code=False)

    assert summary["training"]["best_epoch"] == 142
    assert (destination / "models/yolo11s-target-v6/best.pt").read_bytes() == b"best"
    assert (destination / "models/yolo11s-target-v6/last.pt").read_bytes() == b"last"
    assert not list(destination.rglob("epoch0.pt"))
    assert (destination / "training/150epoch-seed42/results.csv").is_file()
    assert (destination / "data/target-arrow-detection-v6/train/images/train.jpg").is_file()


def test_sync_release_makes_yaml_and_manifest_paths_portable(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "release"
    _source_fixture(source)

    sync_release(source, destination, include_code=False)

    data_yaml = (destination / "data/target-arrow-detection-v6/data.yaml").read_text(encoding="utf-8")
    run_manifest = (destination / "training/150epoch-seed42/run_manifest.json").read_text(encoding="utf-8")
    args_yaml = (destination / "training/150epoch-seed42/args.yaml").read_text(encoding="utf-8")
    assert "train: train/images" in data_yaml
    assert "path:" not in data_yaml
    assert str(source) not in run_manifest
    assert str(source) not in args_yaml
    assert "models/yolo11s-target-v6/best.pt" in run_manifest
    assert (destination / "provenance/path_rewrite.json").is_file()
