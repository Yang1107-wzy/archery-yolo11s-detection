from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2


def prediction_payload(
    *,
    image: str,
    width: int,
    height: int,
    names: dict[int, str],
    boxes: list[dict[str, Any]],
    checkpoint: str,
    run_id: str | None,
    dataset_fingerprint: str | None,
) -> dict[str, Any]:
    target_candidates = [item for item in boxes if str(names[int(item["class_id"])]).lower() == "target"]
    target = max(target_candidates, key=lambda item: float(item["confidence"]), default=None)
    target_payload = None
    if target:
        tx, ty, tw, th = (float(value) for value in target["xywh"])
        target_payload = {
            "x": tx,
            "y": ty,
            "width": tw,
            "height": th,
            "confidence": float(target["confidence"]),
            "class": "target",
            "class_id": int(target["class_id"]),
        }
    predictions: list[dict[str, Any]] = []
    raw_predictions: list[dict[str, Any]] = []
    score_index = 0
    for item in boxes:
        class_id = int(item["class_id"])
        class_name = str(names[class_id])
        x, y, box_width, box_height = (float(value) for value in item["xywh"])
        row = {
            "x": x,
            "y": y,
            "width": box_width,
            "height": box_height,
            "confidence": float(item["confidence"]),
            "class": class_name,
            "class_id": class_id,
        }
        raw_predictions.append(row)
        if class_name.lower() == "target":
            continue
        score_index += 1
        inside_target = None
        normalized_radius = None
        if target_payload:
            dx = x - float(target_payload["x"])
            dy = y - float(target_payload["y"])
            radius = max(float(target_payload["width"]), float(target_payload["height"])) / 2.0
            normalized_radius = ((dx * dx + dy * dy) ** 0.5) / max(radius, 1e-9)
            inside_target = normalized_radius <= 1.0
        predictions.append(
            {
                **row,
                "detection_id": f"det-{score_index:04d}",
                "point_source": "bbox_center",
                "target_relative_x": None if not target_payload else x - float(target_payload["x"]),
                "target_relative_y": None if not target_payload else y - float(target_payload["y"]),
                "normalized_radius": normalized_radius,
                "inside_target": inside_target,
            }
        )
    return {
        "image": image,
        "coordinate_space": "original",
        "image_width": width,
        "image_height": height,
        "model": {
            "type": "local_score_replica",
            "checkpoint": checkpoint,
            "run_id": run_id,
            "dataset_fingerprint": dataset_fingerprint,
        },
        "target": target_payload,
        "raw_predictions": raw_predictions,
        "predictions": predictions,
        "claim_boundary": "Score points are bbox centers, not physical impact-point ground truth.",
    }


def _manifest_for_checkpoint(root: Path, checkpoint: Path) -> dict[str, Any]:
    manifests = sorted((root / "artifacts" / "run_manifests").glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    for path in manifests:
        payload = json.loads(path.read_text(encoding="utf-8"))
        recorded = root / str(payload.get("checkpoint", ""))
        if recorded.resolve() == checkpoint.resolve():
            return payload
    return {}


def predict_scores(
    checkpoint: str | Path,
    source: str | Path,
    output_dir: str | Path,
    *,
    confidence: float = 0.25,
    iou: float = 0.7,
    device: str | int = "cpu",
) -> list[dict[str, Any]]:
    from ultralytics import YOLO

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[3]
    manifest = _manifest_for_checkpoint(root, checkpoint_path)
    model = YOLO(str(checkpoint_path))
    results = model.predict(source=str(Path(source).expanduser().resolve()), conf=confidence, iou=iou, device=device, verbose=False)
    payloads: list[dict[str, Any]] = []
    for result in results:
        height, width = result.orig_shape
        boxes: list[dict[str, Any]] = []
        if result.boxes is not None:
            for xywh, confidence_value, class_id in zip(result.boxes.xywh.cpu(), result.boxes.conf.cpu(), result.boxes.cls.cpu()):
                boxes.append(
                    {
                        "xywh": [float(value) for value in xywh.tolist()],
                        "confidence": float(confidence_value),
                        "class_id": int(class_id),
                    }
                )
        payload = prediction_payload(
            image=str(result.path),
            width=width,
            height=height,
            names={int(key): str(value) for key, value in result.names.items()},
            boxes=boxes,
            checkpoint=str(checkpoint_path),
            run_id=manifest.get("run_id"),
            dataset_fingerprint=manifest.get("dataset_fingerprint"),
        )
        stem = Path(result.path).stem
        (output / f"{stem}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        cv2.imwrite(str(output / f"{stem}_annotated.jpg"), result.plot())
        payloads.append(payload)
    (output / "summary.json").write_text(json.dumps(payloads, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payloads
