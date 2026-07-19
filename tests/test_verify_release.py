from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import yaml
from PIL import Image

from tools.verify_release import ReleaseContract, verify_release


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fixture(root: Path) -> ReleaseContract:
    dataset = root / "data/target-arrow-detection-v6"
    for split in ("train", "valid", "test"):
        (dataset / split / "images").mkdir(parents=True)
        (dataset / split / "labels").mkdir(parents=True)
        Image.new("RGB", (16, 16)).save(dataset / split / "images" / f"{split}.jpg")
        (dataset / split / "labels" / f"{split}.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    (dataset / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "train": "train/images",
                "val": "valid/images",
                "test": "test/images",
                "names": ["0", "target"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (dataset / "source_metadata").mkdir()
    (dataset / "source_metadata/README.dataset.txt").write_text("License: MIT\n", encoding="utf-8")

    training = root / "training/150epoch-seed42"
    training.mkdir(parents=True)
    with (training / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "metrics/mAP50(B)", "metrics/mAP50-95(B)"],
        )
        writer.writeheader()
        for epoch in range(1, 151):
            writer.writerow(
                {
                    "epoch": epoch,
                    "metrics/mAP50(B)": 0.8,
                    "metrics/mAP50-95(B)": 0.9 if epoch == 142 else 0.4,
                }
            )
    model = root / "models/yolo11s-target-v6"
    model.mkdir(parents=True)
    (model / "best.pt").write_bytes(b"best")
    (model / "last.pt").write_bytes(b"last")
    return ReleaseContract(
        split_counts={"train": 1, "valid": 1, "test": 1},
        class_names=("0", "target"),
        best_sha256=_sha(b"best"),
        last_sha256=_sha(b"last"),
    )


def test_verify_release_accepts_complete_portable_fixture(tmp_path: Path) -> None:
    contract = _fixture(tmp_path)

    report = verify_release(tmp_path, contract=contract)

    assert report["status"] == "passed"
    assert report["dataset"]["images"] == 3
    assert report["training"]["best_epoch"] == 142
    assert report["privacy"]["exif_images"] == 0


def test_verify_release_rejects_intermediate_checkpoint_and_absolute_path(tmp_path: Path) -> None:
    contract = _fixture(tmp_path)
    (tmp_path / "training/150epoch-seed42/epoch7.pt").write_bytes(b"checkpoint")
    (tmp_path / "notes.md").write_text("/Users/alvin/private", encoding="utf-8")  # pragma: allowlist publication-scan

    report = verify_release(tmp_path, contract=contract)

    assert report["status"] == "failed"
    codes = {issue["code"] for issue in report["issues"]}
    assert {"intermediate_checkpoint", "forbidden_text"} <= codes
