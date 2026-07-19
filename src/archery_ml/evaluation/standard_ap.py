from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def run_standard_ap_test(
    model_path: str | Path,
    data_yaml: str | Path,
    output_path: str | Path,
    *,
    device: str = "mps",
    imgsz: int = 640,
    batch: int = 8,
    model: Any | None = None,
) -> dict[str, Any]:
    """Evaluate the public test split with a low confidence floor for standard AP integration."""
    model_file = Path(model_path)
    data_file = Path(data_yaml)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if model is None:
        from ultralytics import YOLO

        model = YOLO(str(model_file))
    result = model.val(
        data=str(data_file),
        split="test",
        imgsz=int(imgsz),
        batch=int(batch),
        workers=0,
        device=device,
        conf=0.001,
        iou=0.7,
        plots=False,
        verbose=False,
        project=str(target.parent / "standard_ap_runs"),
        name="test_low_conf",
        exist_ok=True,
    )
    values = result.results_dict or {}
    metrics = {
        "precision": float(values.get("metrics/precision(B)", 0.0)),
        "recall": float(values.get("metrics/recall(B)", 0.0)),
        "map50": float(values.get("metrics/mAP50(B)", 0.0)),
        "map50_95": float(values.get("metrics/mAP50-95(B)", 0.0)),
    }
    payload = {
        "protocol": "standard_ap_low_confidence_sweep",
        "model": str(model_file),
        "data_yaml": str(data_file),
        "split": "test",
        "imgsz": int(imgsz),
        "batch": int(batch),
        "confidence_floor": 0.001,
        "nms_iou": 0.7,
        "device": str(device),
        "metrics": metrics,
        "claim_boundary": (
            "Public test evaluated with a low confidence floor for AP integration. "
            "These AP values are not the locked deployment-threshold operating point."
        ),
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
