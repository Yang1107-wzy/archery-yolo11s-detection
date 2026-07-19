from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from archery_ml.viz.repository_visuals import (
    collect_dataset_statistics,
    generate_all,
    load_evaluation_protocols,
    load_training_series,
)


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_dataset_statistics() -> None:
    stats = collect_dataset_statistics(ROOT / "data/target-arrow-detection-v6")

    assert stats["class_names"] == [
        "0", "1", "10", "2", "3", "4", "5", "6", "7", "8", "9", "target"
    ]
    assert {name: split["images"] for name, split in stats["splits"].items()} == {
        "train": 1482,
        "validation": 98,
        "test": 65,
    }
    assert stats["total_images"] == 1645
    assert stats["total_labels"] == 1645
    assert sum(stats["class_instances"].values()) == stats["total_instances"]


def test_canonical_training_series() -> None:
    frame = load_training_series(ROOT / "training/150epoch-seed42/results.csv")

    assert frame["epoch"].tolist() == list(range(1, 151))
    best = frame.loc[frame["metrics/mAP50-95(B)"].idxmax()]
    assert int(best["epoch"]) == 142
    assert float(best["metrics/mAP50-95(B)"]) == pytest.approx(0.43646)


def test_canonical_evaluation_protocols() -> None:
    protocols = load_evaluation_protocols(ROOT / "evaluation/results_summary.json")

    assert protocols["validation_best"]["epoch"] == 142
    assert protocols["test_standard_ap"]["map50"] == pytest.approx(
        0.7590066637476979, abs=1e-15
    )
    assert protocols["test_locked_operating_point"]["confidence"] == 0.25
    assert protocols["test_locked_operating_point"]["nms_iou"] == 0.5
    json.dumps(protocols, allow_nan=False)


def test_generate_all_visuals_and_source_tables(tmp_path: Path) -> None:
    output_dir = tmp_path / "visuals"
    source_dir = tmp_path / "sources"
    outputs = generate_all(ROOT, output_dir=output_dir, source_dir=source_dir)

    expected_pngs = {
        "model_pipeline.png",
        "dataset_overview.png",
        "dataset_samples.png",
        "training_dynamics.png",
        "evaluation_comparison.png",
    }
    assert {path.name for path in outputs if path.suffix == ".png"} == expected_pngs
    for name in expected_pngs:
        with Image.open(output_dir / name) as image:
            assert image.width >= 1200
            assert image.height >= 700

    expected_tables = {
        "dataset_statistics.csv",
        "training_metrics.csv",
        "evaluation_protocols.csv",
    }
    assert {path.name for path in source_dir.glob("*.csv")} == expected_tables
    manifest = json.loads((source_dir / "visual_manifest.json").read_text(encoding="utf-8"))
    assert manifest["canonical_values"]["standard_test_map50"] == pytest.approx(
        0.7590066637476979, abs=1e-15
    )
    assert manifest["canonical_values"]["best_validation_epoch"] == 142
    assert manifest["dataset"]["total_images"] == 1645


def test_local_inference_asset_is_sanitized() -> None:
    asset = ROOT / "docs/assets/visuals/local_model_inference_streamlit.png"
    assert asset.exists()
    with Image.open(asset) as image:
        assert image.format == "PNG"
        assert image.size == (1732, 980)
        assert image.getexif() == {}
        assert "exif" not in image.info
