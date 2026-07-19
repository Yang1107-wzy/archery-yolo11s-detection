from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from archery_ml.demo.yolo_replica_demo import (
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU,
    EXPECTED_CLASS_NAMES,
    EXPECTED_CHECKPOINT_SHA256,
    InferenceResult,
    build_batch_zip,
    decode_uploaded_image,
    duplicate_filenames,
    load_metrics_dashboard,
    parse_yolo_result,
    predict_uploaded_image,
    result_csv_bytes,
    result_json_bytes,
    validate_checkpoint,
)


class FakeTensor:
    def __init__(self, values) -> None:
        self._values = np.asarray(values)

    def cpu(self):
        return self

    def tolist(self):
        return self._values.tolist()


def fake_yolo_result() -> SimpleNamespace:
    boxes = SimpleNamespace(
        xyxy=FakeTensor([[10.0, 20.0, 30.0, 50.0], [1.0, 2.0, 90.0, 95.0]]),
        conf=FakeTensor([0.81234, 0.95555]),
        cls=FakeTensor([2, 11]),
    )
    return SimpleNamespace(
        orig_shape=(100, 120),
        names={index: name for index, name in enumerate(EXPECTED_CLASS_NAMES)},
        boxes=boxes,
        speed={"preprocess": 1.0, "inference": 2.5, "postprocess": 0.5},
        plot=lambda: np.full((100, 120, 3), 127, dtype=np.uint8),
    )


def test_validate_checkpoint_accepts_matching_hash(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"formal-checkpoint")
    expected = hashlib.sha256(b"formal-checkpoint").hexdigest()

    assert validate_checkpoint(checkpoint, expected_sha256=expected) == expected


def test_validate_checkpoint_rejects_missing_or_wrong_hash(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        validate_checkpoint(tmp_path / "missing.pt")

    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"wrong")
    with pytest.raises(RuntimeError, match="SHA-256"):
        validate_checkpoint(checkpoint, expected_sha256="0" * 64)


def test_decode_uploaded_image_supports_png_and_rejects_empty() -> None:
    image = np.full((16, 24, 3), 80, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok

    decoded = decode_uploaded_image(encoded.tobytes(), "sample.png")

    assert decoded.shape == (16, 24, 3)
    with pytest.raises(RuntimeError, match="空文件"):
        decode_uploaded_image(b"", "empty.png")


def test_parse_yolo_result_preserves_checkpoint_class_order_and_box_geometry() -> None:
    parsed = parse_yolo_result(
        fake_yolo_result(),
        image_name="sample.jpg",
        checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
        run_id="replica-test",
        dataset_fingerprint="dataset-test",
        confidence=DEFAULT_CONFIDENCE,
        iou=DEFAULT_IOU,
        device="mps",
        latency_ms=8.0,
    )

    assert isinstance(parsed, InferenceResult)
    assert parsed.width == 120
    assert parsed.height == 100
    assert parsed.detection_count == 2
    assert parsed.score_detection_count == 1
    assert parsed.target_detected is True
    assert parsed.detections[0].label == "10"
    assert parsed.detections[0].center_x == 20.0
    assert parsed.detections[0].center_y == 35.0
    assert parsed.detections[0].width == 20.0
    assert parsed.detections[0].height == 30.0
    assert parsed.mean_confidence == pytest.approx((0.81234 + 0.95555) / 2)


def test_parse_yolo_result_rejects_unexpected_class_mapping() -> None:
    result = fake_yolo_result()
    result.names[2] = "wrong"

    with pytest.raises(RuntimeError, match="类别映射"):
        parse_yolo_result(
            result,
            image_name="sample.jpg",
            checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
            run_id=None,
            dataset_fingerprint=None,
            confidence=0.25,
            iou=0.5,
            device="cpu",
            latency_ms=1.0,
        )


def test_parse_yolo_result_allows_no_detections() -> None:
    result = fake_yolo_result()
    result.boxes = None

    parsed = parse_yolo_result(
        result,
        image_name="empty.jpg",
        checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
        run_id=None,
        dataset_fingerprint=None,
        confidence=0.25,
        iou=0.5,
        device="cpu",
        latency_ms=1.0,
    )

    assert parsed.detection_count == 0
    assert parsed.mean_confidence == 0.0
    assert parsed.target_detected is False


def test_json_csv_and_batch_zip_exports_are_consistent() -> None:
    parsed = parse_yolo_result(
        fake_yolo_result(),
        image_name="same.jpg",
        checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
        run_id="run",
        dataset_fingerprint="dataset",
        confidence=0.25,
        iou=0.5,
        device="cpu",
        latency_ms=1.0,
    )

    json_payload = json.loads(result_json_bytes(parsed))
    assert json_payload["model"]["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256
    assert json_payload["notice"].startswith("当前图片无 GT")
    assert b"class_id,label,confidence" in result_csv_bytes(parsed)

    archive = build_batch_zip([parsed])
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        assert sorted(bundle.namelist()) == [
            "same/annotated.png",
            "same/detections.csv",
            "same/result.json",
            "summary.csv",
        ]


def test_batch_zip_deduplicates_same_stem_without_overwriting() -> None:
    first = parse_yolo_result(
        fake_yolo_result(),
        image_name="same.jpg",
        checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
        run_id=None,
        dataset_fingerprint=None,
        confidence=0.25,
        iou=0.5,
        device="cpu",
        latency_ms=1.0,
    )
    second = parse_yolo_result(
        fake_yolo_result(),
        image_name="same.png",
        checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
        run_id=None,
        dataset_fingerprint=None,
        confidence=0.25,
        iou=0.5,
        device="cpu",
        latency_ms=1.0,
    )

    with zipfile.ZipFile(io.BytesIO(build_batch_zip([first, second]))) as bundle:
        names = bundle.namelist()
    assert "same/result.json" in names
    assert "same_2/result.json" in names


def test_duplicate_filenames_are_reported_case_insensitively() -> None:
    assert duplicate_filenames(["A.JPG", "a.jpg", "b.png", "b.png"]) == ["a.jpg", "b.png"]


def test_result_serializes_mps_fallback_reason() -> None:
    parsed = parse_yolo_result(
        fake_yolo_result(),
        image_name="fallback.jpg",
        checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
        run_id=None,
        dataset_fingerprint=None,
        confidence=0.25,
        iou=0.5,
        device="cpu",
        latency_ms=1.0,
        device_fallback_reason="MPS NMS unavailable",
    )

    assert parsed.device_fallback_reason == "MPS NMS unavailable"
    assert parsed.to_dict()["inference"]["device_fallback_reason"] == "MPS NMS unavailable"


def test_predict_forwards_thresholds_and_falls_back_from_mps(monkeypatch, tmp_path: Path) -> None:
    image = np.full((16, 24, 3), 80, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"test")

    class FakeModel:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def predict(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs["device"] == "mps":
                raise RuntimeError("MPS NMS unavailable")
            return [fake_yolo_result()]

    model = FakeModel()
    monkeypatch.setattr("archery_ml.demo.yolo_replica_demo.validate_checkpoint", lambda path: "a" * 64)
    monkeypatch.setattr("archery_ml.demo.yolo_replica_demo._read_manifest", lambda path: {})
    monkeypatch.setattr("archery_ml.demo.yolo_replica_demo.load_formal_model", lambda path: model)
    monkeypatch.setattr("archery_ml.demo.yolo_replica_demo.resolve_device", lambda requested: "mps")

    result = predict_uploaded_image(
        encoded.tobytes(),
        "sample.png",
        confidence=0.35,
        iou=0.60,
        checkpoint=checkpoint,
    )

    assert [call["device"] for call in model.calls] == ["mps", "cpu"]
    assert all(call["conf"] == 0.35 and call["iou"] == 0.60 for call in model.calls)
    assert result.device == "cpu"
    assert result.device_fallback_reason == "MPS NMS unavailable"


def test_load_metrics_dashboard_uses_traceable_public_test_sources(tmp_path: Path) -> None:
    reports = tmp_path / "evaluation/source_reports"
    model_dir = tmp_path / "training/150epoch-seed42"
    reports.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    (reports / "04_replication_metrics.json").write_text(
        json.dumps(
            {
                "selected": {"confidence": 0.25, "iou": 0.5},
                "public_test": {"precision": 0.78, "recall": 0.67, "f1": 0.72},
                "per_class": [{"class_id": 11, "name": "target", "f1": 0.98}],
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "results.csv").write_text(
        "epoch,metrics/precision(B),metrics/recall(B)\n1,0.5,0.6\n",
        encoding="utf-8",
    )

    dashboard = load_metrics_dashboard(tmp_path)

    assert dashboard["public_test"]["f1"] == 0.72
    assert dashboard["per_class"][0]["name"] == "target"
    assert dashboard["training_rows"][0]["epoch"] == 1
