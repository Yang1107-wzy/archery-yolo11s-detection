from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from archery_ml.data.audit_yolo import audit_yolo_dataset
from archery_ml.data.dataset_fingerprint import fingerprint_yolo_dataset


def _write_dataset(root: Path) -> Path:
    for split in ("train", "valid", "test"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(root / split / "images" / f"{split}.jpg")
        (root / split / "labels" / f"{split}.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "train/images",
                "val": "valid/images",
                "test": "test/images",
                "names": ["target", "10"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return data_yaml


def test_yolo_audit_and_fingerprint_are_deterministic(tmp_path: Path) -> None:
    data_yaml = _write_dataset(tmp_path)

    audit = audit_yolo_dataset(data_yaml)
    first = fingerprint_yolo_dataset(data_yaml)
    second = fingerprint_yolo_dataset(data_yaml)

    assert audit["status"] == "passed"
    assert audit["splits"]["train"]["images"] == 1
    assert audit["class_instances"] == {"target": 3, "10": 0}
    assert audit["bbox_statistics"]["overall"]["count"] == 3
    assert audit["bbox_statistics"]["overall"]["area"]["median"] == 0.0625
    assert audit["duplicates"]["exact_cross_split_group_count"] == 1
    assert audit["duplicates"]["near_cross_split_pair_count"] >= 3
    assert first["overall_fingerprint"] == second["overall_fingerprint"]
    assert len(first["overall_fingerprint"]) == 64


def test_yolo_audit_reports_invalid_label_without_silently_accepting_it(tmp_path: Path) -> None:
    data_yaml = _write_dataset(tmp_path)
    (tmp_path / "train" / "labels" / "train.txt").write_text("9 1.2 0.5 -0.1 0.2\n", encoding="utf-8")

    audit = audit_yolo_dataset(data_yaml)

    assert audit["status"] == "failed"
    codes = {item["code"] for item in audit["issues"]}
    assert {"class_id_out_of_range", "coordinate_out_of_range", "non_positive_bbox"} <= codes
    json.dumps(audit)
