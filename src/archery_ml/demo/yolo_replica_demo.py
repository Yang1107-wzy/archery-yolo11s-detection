from __future__ import annotations

import csv
import hashlib
import io
import json
import time
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2  # type: ignore
import numpy as np  # type: ignore


ARCHERY_ML_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = (
    ARCHERY_ML_ROOT
    / "models"
    / "yolo11s-target-v6"
    / "best.pt"
)
EXPECTED_CHECKPOINT_SHA256 = "699235268b229cbb5e401d4fb0559d788630de04267605d2927aad60aa262b20"
EXPECTED_CLASS_NAMES = ("0", "1", "10", "2", "3", "4", "5", "6", "7", "8", "9", "target")
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU = 0.50
DEFAULT_RUN_ID = "replica_20260719_002149_184e8ce2_full"
NO_GT_NOTICE = "当前图片无 GT，本页不计算单图 Precision/Recall/mAP；请根据预测框肉眼判断。"
POINT_NOTICE = "0–10 类检测框中心是模型检测位置，不是经过人工 GT 验证的物理撞击点。"


@dataclass(frozen=True)
class DetectionRecord:
    detection_id: str
    class_id: int
    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    center_x: float
    center_y: float
    width: float
    height: float


@dataclass
class InferenceResult:
    image_name: str
    width: int
    height: int
    detections: list[DetectionRecord]
    annotated_image: np.ndarray
    checkpoint_sha256: str
    run_id: str | None
    dataset_fingerprint: str | None
    confidence_threshold: float
    iou_threshold: float
    device: str
    latency_ms: float
    speed_ms: dict[str, float]
    device_fallback_reason: str | None = None

    @property
    def detection_count(self) -> int:
        return len(self.detections)

    @property
    def score_detection_count(self) -> int:
        return sum(item.label != "target" for item in self.detections)

    @property
    def target_detected(self) -> bool:
        return any(item.label == "target" for item in self.detections)

    @property
    def mean_confidence(self) -> float:
        if not self.detections:
            return 0.0
        return float(sum(item.confidence for item in self.detections) / len(self.detections))

    @property
    def max_confidence(self) -> float:
        return max((item.confidence for item in self.detections), default=0.0)

    @property
    def class_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(item.label for item in self.detections).items()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "image": {
                "name": self.image_name,
                "width": self.width,
                "height": self.height,
                "coordinate_space": "original_image_px",
            },
            "model": {
                "name": "YOLO11s Target v6 local replica",
                "checkpoint": "models/yolo11s-target-v6/best.pt",
                "checkpoint_sha256": self.checkpoint_sha256,
                "run_id": self.run_id,
                "dataset_fingerprint": self.dataset_fingerprint,
                "training_contract": "150 epochs, batch 8, seed 42, patience 0; deployed best epoch 142",
            },
            "inference": {
                "confidence_threshold": self.confidence_threshold,
                "iou_threshold": self.iou_threshold,
                "device": self.device,
                "device_fallback_reason": self.device_fallback_reason,
                "latency_ms": self.latency_ms,
                "speed_ms": self.speed_ms,
            },
            "summary": {
                "detection_count": self.detection_count,
                "score_detection_count": self.score_detection_count,
                "target_detected": self.target_detected,
                "mean_confidence": self.mean_confidence,
                "max_confidence": self.max_confidence,
                "class_counts": self.class_counts,
            },
            "detections": [asdict(item) for item in self.detections],
            "notice": NO_GT_NOTICE,
            "point_claim_boundary": POINT_NOTICE,
        }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_checkpoint(
    checkpoint: str | Path,
    *,
    expected_sha256: str = EXPECTED_CHECKPOINT_SHA256,
) -> str:
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"正式 checkpoint 不存在: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(
            "正式 checkpoint SHA-256 校验失败；"
            f"期望 {expected_sha256}，实际 {actual}。不会自动改用其他权重。"
        )
    return actual


def decode_uploaded_image(data: bytes, filename: str) -> np.ndarray:
    if not data:
        raise RuntimeError("上传的是空文件，无法执行推理。")
    encoded = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法解码图片: {filename}")
    return image


def _normalized_names(names: Any) -> tuple[str, ...]:
    if isinstance(names, dict):
        return tuple(str(names[index]) for index in sorted(int(key) for key in names))
    return tuple(str(value) for value in names)


def _validate_class_mapping(names: Any) -> None:
    actual = _normalized_names(names)
    if actual != EXPECTED_CLASS_NAMES:
        raise RuntimeError(f"模型类别映射不符合正式 data.yaml：期望 {EXPECTED_CLASS_NAMES}，实际 {actual}")


def parse_yolo_result(
    result: Any,
    *,
    image_name: str,
    checkpoint_sha256: str,
    run_id: str | None,
    dataset_fingerprint: str | None,
    confidence: float,
    iou: float,
    device: str,
    latency_ms: float,
    device_fallback_reason: str | None = None,
) -> InferenceResult:
    _validate_class_mapping(result.names)
    height, width = (int(value) for value in result.orig_shape)
    detections: list[DetectionRecord] = []
    boxes = result.boxes
    if boxes is not None:
        for index, (xyxy, confidence_value, class_value) in enumerate(
            zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist()),
            start=1,
        ):
            class_id = int(class_value)
            x1, y1, x2, y2 = (float(value) for value in xyxy)
            detections.append(
                DetectionRecord(
                    detection_id=f"det-{index:04d}",
                    class_id=class_id,
                    label=EXPECTED_CLASS_NAMES[class_id],
                    confidence=float(confidence_value),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    center_x=(x1 + x2) / 2.0,
                    center_y=(y1 + y2) / 2.0,
                    width=x2 - x1,
                    height=y2 - y1,
                )
            )
    speed = {str(key): float(value) for key, value in (getattr(result, "speed", {}) or {}).items()}
    return InferenceResult(
        image_name=image_name,
        width=width,
        height=height,
        detections=detections,
        annotated_image=result.plot(),
        checkpoint_sha256=checkpoint_sha256,
        run_id=run_id,
        dataset_fingerprint=dataset_fingerprint,
        confidence_threshold=float(confidence),
        iou_threshold=float(iou),
        device=str(device),
        latency_ms=float(latency_ms),
        speed_ms=speed,
        device_fallback_reason=device_fallback_reason,
    )


def _read_manifest(checkpoint: Path) -> dict[str, Any]:
    path = ARCHERY_ML_ROOT / "training" / "150epoch-seed42" / "run_manifest.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = ARCHERY_ML_ROOT / str(payload.get("checkpoint", ""))
    return payload if recorded.resolve() == checkpoint.resolve() else {}


@lru_cache(maxsize=1)
def load_formal_model(checkpoint: str = str(DEFAULT_CHECKPOINT)):
    from ultralytics import YOLO

    checkpoint_path = Path(checkpoint).resolve()
    validate_checkpoint(checkpoint_path)
    return YOLO(str(checkpoint_path))


def resolve_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def predict_uploaded_image(
    data: bytes,
    filename: str,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    iou: float = DEFAULT_IOU,
    device: str = "auto",
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
) -> InferenceResult:
    image = decode_uploaded_image(data, filename)
    checkpoint_path = Path(checkpoint).resolve()
    checkpoint_hash = validate_checkpoint(checkpoint_path)
    manifest = _read_manifest(checkpoint_path)
    model = load_formal_model(str(checkpoint_path))
    selected_device = resolve_device(device)
    fallback_reason = None
    start = time.perf_counter()
    try:
        yolo_result = model.predict(
            source=image,
            conf=float(confidence),
            iou=float(iou),
            device=selected_device,
            verbose=False,
        )[0]
    except Exception as exc:
        if selected_device != "mps":
            raise
        fallback_reason = str(exc) or exc.__class__.__name__
        selected_device = "cpu"
        yolo_result = model.predict(
            source=image,
            conf=float(confidence),
            iou=float(iou),
            device=selected_device,
            verbose=False,
        )[0]
    latency_ms = (time.perf_counter() - start) * 1000.0
    return parse_yolo_result(
        yolo_result,
        image_name=filename,
        checkpoint_sha256=checkpoint_hash,
        run_id=manifest.get("run_id", DEFAULT_RUN_ID),
        dataset_fingerprint=manifest.get("dataset_fingerprint"),
        confidence=confidence,
        iou=iou,
        device=selected_device,
        latency_ms=latency_ms,
        device_fallback_reason=fallback_reason,
    )


def result_json_bytes(result: InferenceResult) -> bytes:
    return (json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def result_csv_bytes(result: InferenceResult) -> bytes:
    buffer = io.StringIO()
    fields = list(asdict(DetectionRecord("", 0, "", 0, 0, 0, 0, 0, 0, 0, 0, 0)).keys())
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(asdict(item) for item in result.detections)
    return buffer.getvalue().encode("utf-8-sig")


def annotated_png_bytes(result: InferenceResult) -> bytes:
    ok, encoded = cv2.imencode(".png", result.annotated_image)
    if not ok:
        raise RuntimeError(f"无法编码预测标注图: {result.image_name}")
    return encoded.tobytes()


def _unique_stems(results: list[InferenceResult]) -> list[str]:
    seen: Counter[str] = Counter()
    stems: list[str] = []
    for result in results:
        base = Path(result.image_name).stem or "image"
        seen[base] += 1
        stems.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
    return stems


def duplicate_filenames(filenames: list[str]) -> list[str]:
    normalized = [str(name).strip().lower() for name in filenames]
    counts = Counter(normalized)
    return sorted(name for name, count in counts.items() if count > 1)


def _batch_summary_csv(results: list[InferenceResult], stems: list[str]) -> bytes:
    buffer = io.StringIO()
    fields = [
        "output_id",
        "image_name",
        "width",
        "height",
        "detection_count",
        "score_detection_count",
        "target_detected",
        "mean_confidence",
        "max_confidence",
        "latency_ms",
        "device",
        "confidence_threshold",
        "iou_threshold",
        "class_counts",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for stem, result in zip(stems, results):
        writer.writerow(
            {
                "output_id": stem,
                "image_name": result.image_name,
                "width": result.width,
                "height": result.height,
                "detection_count": result.detection_count,
                "score_detection_count": result.score_detection_count,
                "target_detected": result.target_detected,
                "mean_confidence": result.mean_confidence,
                "max_confidence": result.max_confidence,
                "latency_ms": result.latency_ms,
                "device": result.device,
                "confidence_threshold": result.confidence_threshold,
                "iou_threshold": result.iou_threshold,
                "class_counts": json.dumps(result.class_counts, ensure_ascii=False, sort_keys=True),
            }
        )
    return buffer.getvalue().encode("utf-8-sig")


def build_batch_zip(results: list[InferenceResult]) -> bytes:
    buffer = io.BytesIO()
    stems = _unique_stems(results)
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("summary.csv", _batch_summary_csv(results, stems))
        for stem, result in zip(stems, results):
            archive.writestr(f"{stem}/result.json", result_json_bytes(result))
            archive.writestr(f"{stem}/detections.csv", result_csv_bytes(result))
            archive.writestr(f"{stem}/annotated.png", annotated_png_bytes(result))
    return buffer.getvalue()


def save_result_artifacts(result: InferenceResult, output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_bytes(result_json_bytes(result))
    (output / "detections.csv").write_bytes(result_csv_bytes(result))
    (output / "annotated.png").write_bytes(annotated_png_bytes(result))
    return output


def _coerce_csv_value(value: str) -> Any:
    if value == "":
        return value
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() else number


def load_metrics_dashboard(root: str | Path = ARCHERY_ML_ROOT) -> dict[str, Any]:
    project = Path(root)
    metrics_path = project / "evaluation" / "source_reports" / "04_replication_metrics.json"
    results_path = project / "training" / "150epoch-seed42" / "results.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"全局评测报告不存在: {metrics_path}")
    if not results_path.is_file():
        raise FileNotFoundError(f"训练曲线数据不存在: {results_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    manifest_path = project / "training" / "150epoch-seed42" / "run_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    with results_path.open("r", encoding="utf-8-sig", newline="") as handle:
        training_rows = [
            {key.strip(): _coerce_csv_value(value.strip()) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    return {
        "selected": metrics.get("selected", {}),
        "best_validation": manifest.get("best_validation_metrics", {}),
        "public_test": metrics.get("public_test", {}),
        "per_class": metrics.get("per_class", []),
        "claim_boundary": metrics.get("claim_boundary", ""),
        "training_rows": training_rows,
        "metrics_path": str(metrics_path),
        "results_path": str(results_path),
        "manifest_path": str(manifest_path),
        "confusion_matrix_path": str(project / "training/150epoch-seed42/confusion_matrix_normalized.png"),
        "training_graph_path": str(project / "training/150epoch-seed42/roboflow_style_metrics_150epochs.png"),
    }
