from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from archery_ml.viz.roboflow_train_metrics import METRIC_COLUMNS, load_metrics, render_metrics


def test_render_metrics_uses_all_rows_and_exports_png_pdf(tmp_path: Path) -> None:
    source = tmp_path / "results.csv"
    rows = []
    for epoch in range(1, 4):
        rows.append(
            {
                "epoch": epoch,
                "time": epoch * 10.0,
                "train/box_loss": 2.0 / epoch,
                "train/cls_loss": 1.5 / epoch,
                "train/dfl_loss": 1.2 / epoch,
                "metrics/precision(B)": 0.2 * epoch,
                "metrics/recall(B)": 0.18 * epoch,
                "metrics/mAP50(B)": 0.15 * epoch,
                "metrics/mAP50-95(B)": 0.1 * epoch,
                "val/box_loss": 1.8 / epoch,
                "val/cls_loss": 1.3 / epoch,
                "val/dfl_loss": 1.1 / epoch,
            }
        )
    pd.DataFrame(rows).to_csv(source, index=False)

    metrics = load_metrics(source, expected_epochs=3)
    assert list(metrics.columns) == ["epoch", *METRIC_COLUMNS]
    assert metrics["epoch"].tolist() == [1, 2, 3]

    png_path = tmp_path / "metrics.png"
    pdf_path = tmp_path / "metrics.pdf"
    render_metrics(metrics, png_path=png_path, pdf_path=pdf_path)

    assert png_path.is_file() and png_path.stat().st_size > 10_000
    assert pdf_path.is_file() and pdf_path.stat().st_size > 1_000
    with Image.open(png_path) as image:
        assert image.width >= 3000
        assert image.height >= 1600
