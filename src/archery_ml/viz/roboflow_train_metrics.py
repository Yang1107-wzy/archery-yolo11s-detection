"""Render Roboflow-style YOLO training graphs from an Ultralytics results.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


METRIC_COLUMNS = [
    "time",
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "metrics/precision(B)",
    "metrics/mAP50-95(B)",
    "val/box_loss",
    "val/cls_loss",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
]


def load_metrics(csv_path: Path, *, expected_epochs: int | None = None) -> pd.DataFrame:
    """Load and validate the exact metrics needed by the 2x5 comparison graph."""
    csv_path = Path(csv_path)
    data = pd.read_csv(csv_path)
    data.columns = [str(column).strip() for column in data.columns]

    required = ["epoch", *METRIC_COLUMNS]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    metrics = data.loc[:, required].copy()
    for column in required:
        metrics[column] = pd.to_numeric(metrics[column], errors="raise")

    if metrics.empty:
        raise ValueError(f"No epoch rows found in {csv_path}")
    if metrics.isna().any().any():
        raise ValueError(f"NaN values found in required metrics from {csv_path}")

    epochs = metrics["epoch"].astype(int).to_numpy()
    if not np.array_equal(epochs, np.arange(1, len(metrics) + 1)):
        raise ValueError("Epochs must be complete, ordered, and consecutive from 1")
    metrics["epoch"] = epochs

    if expected_epochs is not None and len(metrics) != expected_epochs:
        raise ValueError(f"Expected {expected_epochs} epochs, found {len(metrics)}")
    return metrics


def render_metrics(metrics: pd.DataFrame, *, png_path: Path, pdf_path: Path) -> None:
    """Render raw values and Gaussian-smoothed curves in the reference panel order."""
    png_path = Path(png_path)
    pdf_path = Path(pdf_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = metrics["epoch"].to_numpy(dtype=float)
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.linewidth": 1.0,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    ):
        figure, axes = plt.subplots(2, 5, figsize=(18, 10), constrained_layout=True)
        for index, (axis, column) in enumerate(zip(axes.ravel(), METRIC_COLUMNS, strict=True)):
            values = metrics[column].to_numpy(dtype=float)
            axis.plot(
                epochs,
                values,
                color="#1f77b4",
                marker="o",
                markersize=3.0,
                linewidth=1.5,
                label="results",
            )
            axis.plot(
                epochs,
                gaussian_filter1d(values, sigma=3),
                color="#ff7f0e",
                linestyle=":",
                linewidth=2.0,
                label="smooth",
            )
            axis.set_title(column)
            axis.set_xlim(0, max(epochs) + 2)
            axis.tick_params(direction="out", length=4, width=0.9)
            if index == 1:
                axis.legend(loc="best", frameon=True, fontsize=10)

        figure.savefig(png_path, dpi=200, facecolor="white")
        figure.savefig(pdf_path, facecolor="white")
        plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_csv", type=Path)
    parser.add_argument("--expected-epochs", type=int, default=None)
    parser.add_argument("--png", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = load_metrics(args.results_csv, expected_epochs=args.expected_epochs)
    render_metrics(metrics, png_path=args.png, pdf_path=args.pdf)
    print(f"Rendered {len(metrics)} epochs to {args.png} and {args.pdf}")


if __name__ == "__main__":
    main()
