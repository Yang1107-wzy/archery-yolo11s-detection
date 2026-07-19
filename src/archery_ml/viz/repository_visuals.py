"""Deterministic, source-backed figures for the public repository."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import FancyBboxPatch
from PIL import Image


SPLIT_DIRS = {"train": "train", "validation": "valid", "test": "test"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
NAVY = "#17324D"
BLUE = "#2F6B9A"
LIGHT_BLUE = "#8DB8D8"
GOLD = "#D4A72C"
CHARCOAL = "#263238"
LIGHT_GRAY = "#E8EEF2"


def collect_dataset_statistics(dataset_root: Path) -> dict[str, Any]:
    """Validate the published YOLO dataset and return exact counts."""
    dataset_root = Path(dataset_root)
    metadata = yaml.safe_load((dataset_root / "data.yaml").read_text(encoding="utf-8"))
    class_names = [str(name) for name in metadata["names"]]
    per_class: Counter[str] = Counter({name: 0 for name in class_names})
    split_stats: dict[str, dict[str, int]] = {}

    for public_name, directory in SPLIT_DIRS.items():
        images_dir = dataset_root / directory / "images"
        labels_dir = dataset_root / directory / "labels"
        images = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        labels = sorted(labels_dir.glob("*.txt"))
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}
        if image_stems != label_stems:
            raise ValueError(f"Image/label mismatch in {public_name}")

        instances = 0
        for label_path in labels:
            for line_number, raw_line in enumerate(
                label_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                line = raw_line.strip()
                if not line:
                    continue
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f"Invalid YOLO row: {label_path}:{line_number}")
                class_id = int(fields[0])
                if class_id < 0 or class_id >= len(class_names):
                    raise ValueError(f"Invalid class id: {label_path}:{line_number}")
                coordinates = [float(value) for value in fields[1:]]
                if not all(0.0 <= value <= 1.0 for value in coordinates):
                    raise ValueError(f"Out-of-range box: {label_path}:{line_number}")
                per_class[class_names[class_id]] += 1
                instances += 1

        split_stats[public_name] = {
            "images": len(images),
            "labels": len(labels),
            "instances": instances,
        }

    return {
        "class_names": class_names,
        "splits": split_stats,
        "class_instances": dict(per_class),
        "total_images": sum(item["images"] for item in split_stats.values()),
        "total_labels": sum(item["labels"] for item in split_stats.values()),
        "total_instances": sum(per_class.values()),
    }


def load_training_series(csv_path: Path) -> pd.DataFrame:
    """Load and validate the canonical 150-epoch training series."""
    frame = pd.read_csv(csv_path)
    frame.columns = [str(column).strip() for column in frame.columns]
    if "epoch" not in frame:
        raise ValueError("Training CSV has no epoch column")
    frame["epoch"] = frame["epoch"].astype(int)
    expected = list(range(1, len(frame) + 1))
    if frame["epoch"].tolist() != expected:
        raise ValueError("Training epochs are not contiguous and one-indexed")
    return frame


def load_evaluation_protocols(summary_path: Path) -> dict[str, Any]:
    """Load the protocol-separated evaluation summary without rewriting values."""
    payload = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    required = {"validation_best", "test_standard_ap", "test_locked_operating_point"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Evaluation summary missing: {sorted(missing)}")
    return payload


def _finish_figure(fig: plt.Figure, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def _style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color=LIGHT_GRAY, linewidth=0.8)
    axis.set_axisbelow(True)


def render_model_pipeline(output_path: Path) -> Path:
    fig, axis = plt.subplots(figsize=(12, 7))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    fig.suptitle("YOLO11s Target Detection Pipeline", fontsize=22, fontweight="bold", color=NAVY)
    stages = [
        ("Input", "640 × 640 RGB\nletterbox resize"),
        ("Backbone", "Conv · C3k2 · SPPF · C2PSA\nvisual feature extraction"),
        ("Neck", "upsample · concatenate · C3k2\nmulti-scale feature fusion"),
        ("Detect head", "P3/8 · P4/16 · P5/32\n12 classes + bounding boxes"),
        ("NMS", "confidence + IoU filtering\nfinal detections"),
    ]
    xs = np.linspace(0.105, 0.895, len(stages))
    box_width = 0.17
    box_height = 0.21
    for index, ((title, detail), x) in enumerate(zip(stages, xs, strict=True)):
        box = FancyBboxPatch(
            (x - box_width / 2, 0.57 - box_height / 2), box_width, box_height,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=BLUE if index < 4 else GOLD, edgecolor="white", linewidth=2,
        )
        axis.add_patch(box)
        axis.text(x, 0.615, title, ha="center", va="center", fontsize=11.5, color="white")
        axis.text(x, 0.535, detail, ha="center", va="center", fontsize=8.2,
                  color="white", linespacing=1.25)
        if index < len(stages) - 1:
            axis.annotate("", xy=(xs[index + 1] - box_width / 2 - 0.004, 0.57),
                          xytext=(x + box_width / 2 + 0.004, 0.57),
                          arrowprops={"arrowstyle": "->", "color": CHARCOAL, "lw": 2})
    axis.text(0.5, 0.19, "Published checkpoint: best.pt  ·  12-class Target v6 model",
              ha="center", fontsize=13, fontweight="bold", color=NAVY)
    axis.text(0.5, 0.12, "Output boxes support score-region/target detection; box centers are not physical impact-point labels.",
              ha="center", fontsize=10.5, color=CHARCOAL)
    return _finish_figure(fig, output_path)


def render_dataset_overview(stats: dict[str, Any], output_path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), gridspec_kw={"width_ratios": [0.8, 1.5]})
    fig.suptitle("Target v6 Dataset Evidence", fontsize=22, fontweight="bold", color=NAVY)
    split_names = list(stats["splits"])
    image_counts = [stats["splits"][name]["images"] for name in split_names]
    bars = axes[0].bar(split_names, image_counts, color=[BLUE, LIGHT_BLUE, GOLD], width=0.62)
    axes[0].bar_label(bars, padding=4, fontsize=11, fontweight="bold")
    axes[0].set_title("Image split (total 1,645)")
    axes[0].set_ylabel("Images")
    axes[0].set_ylim(0, max(image_counts) * 1.15)
    _style_axis(axes[0])

    names = stats["class_names"]
    values = [stats["class_instances"][name] for name in names]
    bars = axes[1].bar(names, values, color=[GOLD if name == "target" else BLUE for name in names])
    axes[1].bar_label(bars, padding=3, fontsize=8)
    axes[1].set_title(f"Bounding-box instances by class (total {stats['total_instances']:,})")
    axes[1].set_xlabel("Class name (dataset order)")
    axes[1].set_ylabel("Instances")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].set_ylim(0, max(values) * 1.14)
    _style_axis(axes[1])
    fig.text(0.5, 0.015, "Counts are recomputed from all tracked YOLO label files; empty labels remain represented.",
             ha="center", fontsize=10, color=CHARCOAL)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    return _finish_figure(fig, output_path)


def _select_sample(dataset_root: Path, split_dir: str) -> Path:
    candidates = sorted(
        path for path in (dataset_root / split_dir / "images").iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not candidates:
        raise ValueError(f"No sample image found for {split_dir}")
    return candidates[len(candidates) // 2]


def render_sample_montage(dataset_root: Path, output_path: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13, 7))
    fig.suptitle("Deterministic Dataset Samples", fontsize=22, fontweight="bold", color=NAVY)
    for axis, (label, directory) in zip(axes, SPLIT_DIRS.items(), strict=True):
        image_path = _select_sample(dataset_root, directory)
        with Image.open(image_path) as source:
            axis.imshow(source.convert("RGB"))
        axis.set_title(f"{label.title()} split\n{image_path.name[:34]}", fontsize=11)
        axis.axis("off")
    fig.text(0.5, 0.04, "One fixed lexicographic midpoint image per split; shown for data character, not metric estimation.",
             ha="center", fontsize=10.5, color=CHARCOAL)
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    return _finish_figure(fig, output_path)


def render_training_dynamics(frame: pd.DataFrame, output_path: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    fig.suptitle("YOLO11s Training Dynamics — 150 Epochs, Batch 8, Seed 42",
                 fontsize=20, fontweight="bold", color=NAVY)
    panels = [
        (axes[0, 0], ["train/box_loss", "train/cls_loss", "train/dfl_loss"], "Training losses", False),
        (axes[0, 1], ["val/box_loss", "val/cls_loss", "val/dfl_loss"], "Validation losses", False),
        (axes[1, 0], ["metrics/precision(B)", "metrics/recall(B)"], "Validation precision and recall", True),
        (axes[1, 1], ["metrics/mAP50(B)", "metrics/mAP50-95(B)"], "Validation average precision", True),
    ]
    epochs = frame["epoch"]
    colors = [BLUE, GOLD, "#667A8A"]
    for axis, columns, title, metric_axis in panels:
        for index, column in enumerate(columns):
            raw = frame[column]
            color = colors[index]
            axis.plot(epochs, raw, color=color, alpha=0.25, linewidth=1, label=f"{column} raw")
            axis.plot(epochs, raw.rolling(7, center=True, min_periods=1).mean(), color=color,
                      linewidth=2.2, label=f"{column} 7-epoch mean")
        axis.axvline(81, color=CHARCOAL, linestyle="--", linewidth=1.2, alpha=0.8)
        axis.axvline(142, color="#B23A48", linestyle=":", linewidth=1.8)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        if metric_axis:
            axis.set_ylim(0, 1)
        _style_axis(axis)
        axis.legend(fontsize=7, ncol=2, loc="best")
    axes[0, 1].text(82.5, axes[0, 1].get_ylim()[1] * 0.93, "resume boundary", fontsize=8, color=CHARCOAL)
    axes[1, 1].text(141, 0.08, "best val\nEpoch 142", ha="right", fontsize=8, color="#B23A48")
    fig.text(0.5, 0.015, "Raw epoch values remain visible. The time counter reset at Epoch 81 after resume; weights were not reinitialized.",
             ha="center", fontsize=10, color=CHARCOAL)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    return _finish_figure(fig, output_path)


def render_evaluation_comparison(protocols: dict[str, Any], output_path: Path) -> Path:
    validation = protocols["validation_best"]
    standard = protocols["test_standard_ap"]
    locked = protocols["test_locked_operating_point"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 7))
    fig.suptitle("Evaluation Protocols — Kept Separate", fontsize=22, fontweight="bold", color=NAVY)

    labels = ["Best validation\n(Epoch 142)", "Standard AP test", "Locked test\n(conf=.25, IoU=.50)"]
    x = np.arange(3)
    width = 0.34
    p_bars = axes[0].bar(x - width / 2, [validation["precision"], standard["precision"], locked["precision"]],
                         width, label="Precision", color=BLUE)
    r_bars = axes[0].bar(x + width / 2, [validation["recall"], standard["recall"], locked["recall"]],
                         width, label="Recall", color=GOLD)
    axes[0].bar_label(p_bars, fmt="%.3f", fontsize=8)
    axes[0].bar_label(r_bars, fmt="%.3f", fontsize=8)
    axes[0].set_xticks(x, labels, fontsize=8)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Operating metrics")
    axes[0].legend()
    _style_axis(axes[0])

    ap_labels = ["Best validation", "Standard AP test"]
    ap50 = [validation["map50"], standard["map50"]]
    ap5095 = [validation["map50_95"], standard["map50_95"]]
    x2 = np.arange(2)
    bars1 = axes[1].bar(x2 - width / 2, ap50, width, label="mAP50", color=BLUE)
    bars2 = axes[1].bar(x2 + width / 2, ap5095, width, label="mAP50-95", color=GOLD)
    axes[1].bar_label(bars1, fmt="%.3f", fontsize=9)
    axes[1].bar_label(bars2, fmt="%.3f", fontsize=9)
    axes[1].set_xticks(x2, ap_labels, fontsize=9)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Comparable AP integration")
    axes[1].legend()
    _style_axis(axes[1])

    locked_labels = ["Precision", "Recall", "F1", "Filtered\nmAP50", "Filtered\nmAP50-95"]
    locked_values = [locked["precision"], locked["recall"], locked["f1"],
                     locked["map50_after_confidence_filtering"], locked["map50_95_after_confidence_filtering"]]
    bars = axes[2].bar(locked_labels, locked_values, color=[BLUE, GOLD, "#4B8F8C", LIGHT_BLUE, "#A7B7C4"])
    axes[2].bar_label(bars, fmt="%.3f", fontsize=8)
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Locked deployment threshold")
    axes[2].tick_params(axis="x", labelsize=8, rotation=15)
    _style_axis(axes[2])
    fig.text(0.5, 0.018, "Validation selects the checkpoint; standard test estimates AP; locked-threshold test describes deployment behavior. Filtered AP is not standard AP.",
             ha="center", fontsize=10, color=CHARCOAL)
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    return _finish_figure(fig, output_path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_local_inference_asset(source_path: Path, destination_path: Path) -> Path:
    """Re-encode a qualitative inference screenshot without metadata."""
    source_path = Path(source_path)
    destination_path = Path(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as source:
        pixels = source.copy()
        pixels.save(destination_path, format="PNG", optimize=True)
    return destination_path


def generate_all(
    root: Path,
    *,
    output_dir: Path | None = None,
    source_dir: Path | None = None,
) -> list[Path]:
    """Generate all quantitative repository visuals and auditable source tables."""
    root = Path(root).resolve()
    output_dir = Path(output_dir) if output_dir else root / "docs/assets/visuals"
    source_dir = Path(source_dir) if source_dir else root / "evaluation/visualization_sources"
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = root / "data/target-arrow-detection-v6"
    stats = collect_dataset_statistics(dataset_root)
    training = load_training_series(root / "training/150epoch-seed42/results.csv")
    protocols = load_evaluation_protocols(root / "evaluation/results_summary.json")

    outputs = [
        render_model_pipeline(output_dir / "model_pipeline.png"),
        render_dataset_overview(stats, output_dir / "dataset_overview.png"),
        render_sample_montage(dataset_root, output_dir / "dataset_samples.png"),
        render_training_dynamics(training, output_dir / "training_dynamics.png"),
        render_evaluation_comparison(protocols, output_dir / "evaluation_comparison.png"),
    ]

    dataset_rows = []
    for split, values in stats["splits"].items():
        dataset_rows.append({"record_type": "split", "name": split, **values})
    for class_name, instances in stats["class_instances"].items():
        dataset_rows.append({"record_type": "class", "name": class_name, "instances": instances})
    pd.DataFrame(dataset_rows).to_csv(source_dir / "dataset_statistics.csv", index=False)
    training.to_csv(source_dir / "training_metrics.csv", index=False)
    protocol_rows = [
        {"protocol": "best_validation", **protocols["validation_best"]},
        {"protocol": "standard_ap_test", **protocols["test_standard_ap"]},
        {"protocol": "locked_operating_point_test", **protocols["test_locked_operating_point"]},
    ]
    pd.DataFrame(protocol_rows).to_csv(source_dir / "evaluation_protocols.csv", index=False)

    manifest = {
        "schema_version": 1,
        "generator": "scripts/generate_repository_visuals.py",
        "dataset": stats,
        "canonical_values": {
            "best_validation_epoch": protocols["validation_best"]["epoch"],
            "best_validation_map50_95": protocols["validation_best"]["map50_95"],
            "standard_test_map50": protocols["test_standard_ap"]["map50"],
            "standard_test_map50_95": protocols["test_standard_ap"]["map50_95"],
            "locked_confidence": protocols["test_locked_operating_point"]["confidence"],
            "locked_nms_iou": protocols["test_locked_operating_point"]["nms_iou"],
        },
        "figures": {path.name: {"sha256": _sha256(path)} for path in outputs},
        "claim_boundary": protocols["claim_boundary"],
    }
    (source_dir / "visual_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return outputs
