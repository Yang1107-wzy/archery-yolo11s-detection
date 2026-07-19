from __future__ import annotations

import hashlib
import csv
import json
import math
import platform
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw

from archery_ml.data.dataset_fingerprint import fingerprint_yolo_dataset, sha256_file
from archery_ml.registry import append_run_record


def choose_device() -> str | int:
    import torch

    if torch.cuda.is_available():
        return 0
    if platform.machine() == "arm64" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_train_config(path: str | Path, *, smoke: bool = False, sanity: bool = False) -> dict[str, Any]:
    if smoke and sanity:
        raise ValueError("smoke and sanity modes are mutually exclusive")
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    resolved = dict(raw)
    resolved["config_path"] = str(config_path)
    resolved["config_hash"] = sha256_file(config_path)
    resolved["requested_full_model"] = str(raw.get("model", "yolo11s.pt"))
    if smoke:
        resolved["model"] = str(raw.get("smoke_model", "yolo11s.yaml"))
        resolved["epochs"] = 2
        resolved["batch"] = int(raw.get("smoke_batch", min(int(raw.get("batch", 8)), 8)))
        resolved["workers"] = 0
        resolved["mode"] = "smoke"
    elif sanity:
        resolved["mode"] = "real_sanity"
    else:
        resolved["mode"] = "full"
    resolved["device"] = choose_device() if raw.get("device", "auto") == "auto" else raw["device"]
    return resolved


def _draw_smoke_image(path: Path, class_id: int, seed: int) -> tuple[float, float, float, float]:
    rng = random.Random(seed)
    size = 320
    image = Image.new("RGB", (size, size), (238, 229, 203))
    draw = ImageDraw.Draw(image)
    center_x = size // 2 + rng.randint(-12, 12)
    center_y = size // 2 + rng.randint(-12, 12)
    radius = 135
    palette = [(245, 245, 245), (30, 30, 30), (70, 140, 205), (210, 50, 50), (245, 215, 55)]
    for index, ring_radius in enumerate(range(radius, 0, -27)):
        color = palette[index % len(palette)]
        draw.ellipse(
            (center_x - ring_radius, center_y - ring_radius, center_x + ring_radius, center_y + ring_radius),
            fill=color,
            outline=(20, 20, 20),
            width=2,
        )
    angle = rng.uniform(0, math.tau)
    distance = (class_id % 11) / 11.0 * radius * 0.9
    impact_x = center_x + math.cos(angle) * distance
    impact_y = center_y + math.sin(angle) * distance
    draw.ellipse((impact_x - 5, impact_y - 5, impact_x + 5, impact_y + 5), fill=(10, 10, 10))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=92)
    return impact_x / size, impact_y / size, 12 / size, 12 / size


def build_smoke_dataset(
    output_dir: str | Path,
    names: list[str],
    *,
    seed: int = 42,
) -> dict[str, Any]:
    if len(names) != 12:
        raise ValueError("smoke fixture requires exactly 12 classes")
    root = Path(output_dir).expanduser().resolve()
    counts = {"train": 32, "valid": 16, "test": 8}
    seen: set[int] = {0}
    global_index = 0
    for split, count in counts.items():
        for local_index in range(count):
            score_class = 1 + (global_index % 11)
            seen.add(score_class)
            stem = f"{split}_{local_index:03d}"
            image_path = root / split / "images" / f"{stem}.jpg"
            x, y, width, height = _draw_smoke_image(image_path, score_class - 1, seed + global_index)
            label_path = root / split / "labels" / f"{stem}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text(
                f"0 0.5 0.5 0.90 0.90\n{score_class} {x:.8f} {y:.8f} {width:.8f} {height:.8f}\n",
                encoding="utf-8",
            )
            global_index += 1
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "train/images",
                "val": "valid/images",
                "test": "test/images",
                "names": names,
                "fixture": True,
                "claim_boundary": "Synthetic plumbing-only fixture; not Target v6 and not a performance benchmark.",
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return {"root": str(root), "data_yaml": str(data_yaml), "counts": counts, "class_ids": sorted(seen)}


def canonical_model_dir(
    project_root: str | Path,
    config: dict[str, Any],
    *,
    smoke: bool,
    sanity: bool,
    run_id: str,
) -> Path:
    root = Path(project_root).expanduser().resolve()
    if smoke:
        return root / "artifacts" / "models" / "smoke_fixture" / run_id
    if sanity:
        return root / "artifacts" / "models" / "replica_target_v6" / "sanity" / run_id
    return root / "artifacts" / "models" / "replica_target_v6" / str(config.get("artifact_subdir", run_id))


def copy_training_artifacts(run_dir: str | Path, destination: str | Path) -> list[str]:
    source = Path(run_dir).expanduser().resolve()
    target = Path(destination).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    sources = {
        "best.pt": source / "weights" / "best.pt",
        "last.pt": source / "weights" / "last.pt",
        "args.yaml": source / "args.yaml",
        "results.csv": source / "results.csv",
        "results.png": source / "results.png",
        "confusion_matrix.png": source / "confusion_matrix.png",
        "confusion_matrix_normalized.png": source / "confusion_matrix_normalized.png",
        "BoxF1_curve.png": source / "BoxF1_curve.png",
        "BoxPR_curve.png": source / "BoxPR_curve.png",
        "BoxP_curve.png": source / "BoxP_curve.png",
        "BoxR_curve.png": source / "BoxR_curve.png",
    }
    copied: list[str] = []
    for artifact_name, artifact_source in sources.items():
        if artifact_source.is_file():
            shutil.copy2(artifact_source, target / artifact_name)
            copied.append(artifact_name)
    return copied


def summarize_training_results(results_csv: str | Path) -> dict[str, Any]:
    path = Path(results_csv).expanduser().resolve()
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"training results contain no epochs: {path}")
    metric_keys = (
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    )
    best = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    elapsed_values = [float(row["time"]) for row in rows]
    reported_training_seconds = elapsed_values[0]
    for previous, current in zip(elapsed_values, elapsed_values[1:]):
        reported_training_seconds += current if current < previous else current - previous
    return {
        "epochs_completed": len(rows),
        "best_epoch": int(best["epoch"]),
        "best_validation_metrics": {key: float(best[key]) for key in metric_keys},
        "reported_training_seconds": reported_training_seconds,
    }


def train_yolo(
    config_path: str | Path,
    *,
    smoke: bool,
    project_root: str | Path,
    sanity: bool = False,
) -> dict[str, Any]:
    from ultralytics import YOLO, __version__ as ultralytics_version
    import torch

    root = Path(project_root).expanduser().resolve()
    config = load_train_config(config_path, smoke=smoke, sanity=sanity)
    if not smoke and str(config.get("model")) != "yolo11s.pt":
        raise ValueError("full replica training must use yolo11s.pt")
    if smoke:
        fixture = build_smoke_dataset(
            root / "data" / "processed" / "smoke_target_v6_fixture",
            ["target", *[str(value) for value in range(11)]],
            seed=int(config.get("seed", 42)),
        )
        data_yaml = Path(fixture["data_yaml"])
    else:
        data_yaml = Path(str(config["data"])).expanduser().resolve()
    fingerprint = fingerprint_yolo_dataset(data_yaml)
    started = datetime.now().astimezone()
    suffix = "_smoke" if smoke else ("_sanity" if sanity else "_full")
    run_id = started.strftime("replica_%Y%m%d_%H%M%S_") + config["config_hash"][:8] + suffix
    training_root = root / "artifacts" / "training_runs"
    train_args = {
        "data": str(data_yaml),
        "imgsz": int(config.get("imgsz", 640)),
        "epochs": int(config["epochs"]),
        "batch": int(config.get("batch", 8)),
        "workers": int(config.get("workers", 0)),
        "patience": int(config.get("patience", 30)),
        "seed": int(config.get("seed", 42)),
        "deterministic": bool(config.get("deterministic", True)),
        "device": config["device"],
        "project": str(training_root),
        "name": run_id,
        "exist_ok": False,
        "save": True,
        "plots": True,
        "val": True,
        "verbose": True,
    }
    if "optimizer" in config:
        train_args["optimizer"] = config["optimizer"]
    for key in (
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "flipud",
        "fliplr",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "mosaic",
        "mixup",
        "copy_paste",
        "close_mosaic",
        "save_period",
        "pretrained",
    ):
        if key in config:
            train_args[key] = config[key]
    model = YOLO(str(config["model"]))
    results = model.train(**train_args)
    run_dir = Path(results.save_dir)
    best_source = run_dir / "weights" / "best.pt"
    last_source = run_dir / "weights" / "last.pt"
    canonical_dir = canonical_model_dir(root, config, smoke=smoke, sanity=sanity, run_id=run_id)
    copy_training_artifacts(run_dir, canonical_dir)
    metrics = {key: float(value) for key, value in (results.results_dict or {}).items() if np.isscalar(value)}
    completed = datetime.now().astimezone()
    training_summary = summarize_training_results(run_dir / "results.csv")
    manifest = {
        "run_id": run_id,
        "mode": config["mode"],
        "is_smoke": smoke,
        "architecture": "YOLO11s",
        "initialization": "random yolo11s.yaml offline fallback" if smoke else "pretrained yolo11s.pt",
        "requested_full_model": config["requested_full_model"],
        "actual_model": config["model"],
        "config_path": str(config["config_path"]),
        "config_hash": config["config_hash"],
        "dataset": str(data_yaml),
        "dataset_fingerprint": fingerprint["overall_fingerprint"],
        "checkpoint": str((canonical_dir / "best.pt").relative_to(root)),
        "last_checkpoint": str((canonical_dir / "last.pt").relative_to(root)),
        "seed": int(config.get("seed", 42)),
        "device": str(config["device"]),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "ultralytics": ultralytics_version,
        "started_at": started.isoformat(timespec="seconds"),
        "completed_at": completed.isoformat(timespec="seconds"),
        "duration_seconds": (completed - started).total_seconds(),
        "status": "completed",
        "resume_state": "not_resumed",
        **training_summary,
        "resolved_args": train_args,
        "metrics": metrics,
        "claim_boundary": (
            "Smoke fixture verifies plumbing only; metrics are not Dataset v6, held-out, or comparable to Roboflow."
            if smoke
            else (
                "Five-epoch sanity run on the real Target v6 train/validation splits; not a full-training or held-out-test result."
                if sanity
                else "Full training on Target v6; validation metrics are not public-test metrics until a separate test evaluation is run."
            )
        ),
    }
    manifest_dir = root / "artifacts" / "run_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{run_id}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    shutil.copy2(manifest_path, canonical_dir / "run_manifest.json")
    (canonical_dir / "model_card.md").write_text(
        "\n".join(
            (
                "# YOLO11s Target v6 replica",
                "",
                f"- Run ID: `{run_id}`",
                f"- Mode: `{manifest['mode']}`",
                f"- Dataset fingerprint: `{manifest['dataset_fingerprint']}`",
                f"- Config hash: `{manifest['config_hash']}`",
                f"- Seed: `{manifest['seed']}`",
                f"- Metrics: `{json.dumps(metrics, sort_keys=True)}`",
                "",
                "## Claim boundary",
                "",
                str(manifest["claim_boundary"]),
                "The original Roboflow author's private checkpoint was not obtained; this is a local retraining artifact.",
                "",
            )
        ),
        encoding="utf-8",
    )
    append_run_record(
        root / "memory" / "RUN_REGISTRY.jsonl",
        {
            "run_id": run_id,
            "config_hash": config["config_hash"],
            "dataset_fingerprint": fingerprint["overall_fingerprint"],
            "checkpoint": manifest["checkpoint"],
            "metrics": metrics,
            "best_validation_metrics": training_summary["best_validation_metrics"],
            "epochs": training_summary["epochs_completed"],
            "best_epoch": training_summary["best_epoch"],
            "device": manifest["device"],
            "duration_seconds": manifest["duration_seconds"],
            "status": manifest["status"],
            "resume_state": manifest["resume_state"],
            "seed": manifest["seed"],
            "mode": manifest["mode"],
            "manifest": str(manifest_path.relative_to(root)),
        },
    )
    return manifest
