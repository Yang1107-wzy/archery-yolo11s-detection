from __future__ import annotations

from pathlib import Path

import pytest

from archery_ml.evaluation.threshold_sweep import (
    score_macro_metrics,
    select_operating_point,
    validate_public_test_request,
)


def test_select_operating_point_maximizes_f1_then_map50() -> None:
    rows = [
        {"confidence": 0.10, "iou": 0.50, "precision": 0.5, "recall": 0.5, "f1": 0.5, "map50": 0.7},
        {"confidence": 0.20, "iou": 0.60, "precision": 0.6, "recall": 0.6, "f1": 0.6, "map50": 0.6},
        {"confidence": 0.30, "iou": 0.70, "precision": 0.6, "recall": 0.6, "f1": 0.6, "map50": 0.8},
    ]

    selected = select_operating_point(rows)

    assert selected["confidence"] == 0.30
    assert selected["iou"] == 0.70


def test_score_macro_metrics_excludes_target_class() -> None:
    names = {0: "0", 1: "1", 2: "target"}

    metrics = score_macro_metrics(names, [0.5, 0.75, 1.0], [0.5, 0.25, 1.0])

    assert metrics["score_macro_f1"] == 0.4375
    assert metrics["target_f1"] == 1.0


def test_public_test_requires_same_selected_model_and_fresh_output(tmp_path: Path) -> None:
    model = tmp_path / "best.pt"
    model.write_bytes(b"checkpoint")
    sweep = {"model": str(model.resolve()), "selected": {"confidence": 0.2, "iou": 0.5}}
    output = tmp_path / "metrics.json"

    selected = validate_public_test_request(model, sweep, output)

    assert selected == {"confidence": 0.2, "iou": 0.5}
    with pytest.raises(ValueError, match="model mismatch"):
        validate_public_test_request(tmp_path / "other.pt", sweep, output)
    output.write_text('{"public_test": {"map50": 0.7}}', encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        validate_public_test_request(model, sweep, output)
