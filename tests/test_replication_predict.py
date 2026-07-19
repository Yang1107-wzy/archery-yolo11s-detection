from __future__ import annotations

from archery_ml.models.predict_score import prediction_payload


def test_prediction_payload_marks_bbox_center_as_non_physical_point() -> None:
    payload = prediction_payload(
        image="example.jpg",
        width=640,
        height=480,
        names={0: "target", 1: "9"},
        boxes=[
            {"xywh": [320.0, 240.0, 500.0, 400.0], "confidence": 0.9, "class_id": 0},
            {"xywh": [350.0, 260.0, 20.0, 20.0], "confidence": 0.8, "class_id": 1},
        ],
        checkpoint="best.pt",
        run_id="smoke-001",
        dataset_fingerprint="abc",
    )

    assert payload["target"]["class"] == "target"
    assert payload["predictions"][0]["class"] == "9"
    assert payload["predictions"][0]["point_source"] == "bbox_center"
    assert payload["predictions"][0]["detection_id"] == "det-0001"
