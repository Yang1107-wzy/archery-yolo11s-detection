from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


CONFIDENCE_VALUES = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
IOU_VALUES = (0.30, 0.40, 0.50, 0.60, 0.70)


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def select_operating_point(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("threshold sweep produced no rows")
    return max(rows, key=lambda row: (row["f1"], row["map50"], -row["confidence"], row["iou"]))


def score_macro_metrics(
    names: dict[int, str] | list[str],
    precision: Iterable[float],
    recall: Iterable[float],
) -> dict[str, float]:
    name_map = dict(enumerate(names)) if isinstance(names, list) else {int(key): str(value) for key, value in names.items()}
    p_values = [float(value) for value in precision]
    r_values = [float(value) for value in recall]
    score_f1: list[float] = []
    target_f1 = 0.0
    for class_id, name in name_map.items():
        if class_id >= len(p_values) or class_id >= len(r_values):
            continue
        value = _f1(p_values[class_id], r_values[class_id])
        if name == "target":
            target_f1 = value
        else:
            score_f1.append(value)
    return {
        "score_macro_f1": float(np.mean(score_f1)) if score_f1 else 0.0,
        "target_f1": target_f1,
    }


def _aggregate(metrics: Any) -> dict[str, float]:
    values = metrics.results_dict or {}
    precision = float(values.get("metrics/precision(B)", 0.0))
    recall = float(values.get("metrics/recall(B)", 0.0))
    return {
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "map50": float(values.get("metrics/mAP50(B)", 0.0)),
        "map50_95": float(values.get("metrics/mAP50-95(B)", 0.0)),
    }


def _per_class(metrics: Any, names: dict[int, str]) -> list[dict[str, Any]]:
    box = metrics.box
    class_ids = [int(value) for value in np.asarray(box.ap_class_index).tolist()]
    precision = np.asarray(box.p, dtype=float).tolist()
    recall = np.asarray(box.r, dtype=float).tolist()
    ap50 = np.asarray(box.ap50, dtype=float).tolist()
    ap50_95 = np.asarray(box.ap, dtype=float).tolist()
    return [
        {
            "class_id": class_id,
            "name": names[class_id],
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": _f1(float(precision[index]), float(recall[index])),
            "ap50": float(ap50[index]),
            "ap50_95": float(ap50_95[index]),
        }
        for index, class_id in enumerate(class_ids)
    ]


def run_validation_sweep(
    model_path: str | Path,
    data_yaml: str | Path,
    output_path: str | Path,
    *,
    device: str = "mps",
    imgsz: int = 640,
    batch: int = 8,
) -> dict[str, Any]:
    from ultralytics import YOLO

    model_file = Path(model_path).expanduser().resolve()
    data_file = Path(data_yaml).expanduser().resolve()
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    runs_root = target.parent / "threshold_sweep_runs"
    model = YOLO(str(model_file))
    rows: list[dict[str, float]] = []
    for confidence in CONFIDENCE_VALUES:
        for iou in IOU_VALUES:
            result = model.val(
                data=str(data_file),
                split="val",
                imgsz=imgsz,
                batch=batch,
                workers=0,
                device=device,
                conf=confidence,
                iou=iou,
                plots=False,
                verbose=False,
                project=str(runs_root),
                name=f"val_c{confidence:.2f}_i{iou:.2f}",
                exist_ok=True,
            )
            row = {"confidence": confidence, "iou": iou, **_aggregate(result)}
            rows.append(row)
    selected = select_operating_point(rows)
    payload = {
        "model": str(model_file),
        "data_yaml": str(data_file),
        "validation_sweep": rows,
        "selected": selected,
        "claim_boundary": "Thresholds and candidate checkpoints are selected only on validation; public test has not been evaluated.",
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def validate_public_test_request(
    model_path: str | Path,
    validation_payload: dict[str, Any],
    output_path: str | Path,
) -> dict[str, float]:
    model_file = Path(model_path).expanduser().resolve()
    selected_model = Path(str(validation_payload.get("model", ""))).expanduser().resolve()
    if model_file != selected_model:
        raise ValueError(f"validation-selected model mismatch: {model_file} != {selected_model}")
    selected = validation_payload.get("selected")
    if not isinstance(selected, dict) or "confidence" not in selected or "iou" not in selected:
        raise ValueError("validation payload does not contain locked confidence/IoU")
    target = Path(output_path).expanduser().resolve()
    if target.is_file():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if "public_test" in existing:
            raise FileExistsError(f"public-test result already exists: {target}")
    return {"confidence": float(selected["confidence"]), "iou": float(selected["iou"])}


def run_public_test_from_validation(
    model_path: str | Path,
    data_yaml: str | Path,
    validation_path: str | Path,
    output_path: str | Path,
    *,
    device: str = "mps",
    imgsz: int = 640,
    batch: int = 8,
) -> dict[str, Any]:
    from ultralytics import YOLO

    model_file = Path(model_path).expanduser().resolve()
    data_file = Path(data_yaml).expanduser().resolve()
    validation_file = Path(validation_path).expanduser().resolve()
    target = Path(output_path).expanduser().resolve()
    validation_payload = json.loads(validation_file.read_text(encoding="utf-8"))
    selected = validate_public_test_request(model_file, validation_payload, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_file))
    test_result = model.val(
        data=str(data_file),
        split="test",
        imgsz=imgsz,
        batch=batch,
        workers=0,
        device=device,
        conf=selected["confidence"],
        iou=selected["iou"],
        plots=True,
        verbose=True,
        project=str(target.parent),
        name="replica_target_v6_public_test",
        exist_ok=True,
    )
    names = {int(key): str(value) for key, value in model.names.items()}
    per_class = _per_class(test_result, names)
    macro = score_macro_metrics(
        names,
        [next((row["precision"] for row in per_class if row["class_id"] == class_id), 0.0) for class_id in names],
        [next((row["recall"] for row in per_class if row["class_id"] == class_id), 0.0) for class_id in names],
    )
    payload = {
        "model": str(model_file),
        "data_yaml": str(data_file),
        "validation_sweep": validation_payload["validation_sweep"],
        "selected": selected,
        "public_test": {**_aggregate(test_result), **macro},
        "per_class": per_class,
        "claim_boundary": "Thresholds selected only on validation; public test evaluated once. Official splits contain related images across splits.",
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def run_validation_sweep_and_test(
    model_path: str | Path,
    data_yaml: str | Path,
    output_path: str | Path,
    *,
    device: str = "mps",
    imgsz: int = 640,
    batch: int = 8,
) -> dict[str, Any]:
    target = Path(output_path).expanduser().resolve()
    validation_path = target.with_name(f"{target.stem}_validation.json")
    run_validation_sweep(model_path, data_yaml, validation_path, device=device, imgsz=imgsz, batch=batch)
    return run_public_test_from_validation(
        model_path,
        data_yaml,
        validation_path,
        target,
        device=device,
        imgsz=imgsz,
        batch=batch,
    )
