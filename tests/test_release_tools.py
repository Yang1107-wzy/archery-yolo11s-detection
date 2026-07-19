from __future__ import annotations

import csv
import json
from pathlib import Path

from tools.release_tools import (
    build_file_manifest,
    normalize_portable_paths,
    scan_forbidden_text,
    summarize_results_csv,
)


def test_normalize_portable_paths_rewrites_nested_absolute_paths() -> None:
    payload = {
        "model": "/Users/researcher/work/archery_ml/artifacts/models/best.pt",  # pragma: allowlist publication-scan
        "nested": [
            "/Users/researcher/work/archery_ml/data/processed/target/data.yaml",  # pragma: allowlist publication-scan
            {"unchanged": "YOLO11s"},
        ],
    }
    mapping = {
        "/Users/researcher/work/archery_ml/artifacts/models/best.pt": "models/yolo11s-target-v6/best.pt",  # pragma: allowlist publication-scan
        "/Users/researcher/work/archery_ml/data/processed/target/data.yaml": "data/target-arrow-detection-v6/data.yaml",  # pragma: allowlist publication-scan
    }

    normalized = normalize_portable_paths(payload, mapping)

    assert normalized == {
        "model": "models/yolo11s-target-v6/best.pt",
        "nested": [
            "data/target-arrow-detection-v6/data.yaml",
            {"unchanged": "YOLO11s"},
        ],
    }


def test_summarize_results_requires_150_contiguous_epochs_and_selects_best(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "metrics/mAP50(B)", "metrics/mAP50-95(B)"],
        )
        writer.writeheader()
        for epoch in range(1, 151):
            writer.writerow(
                {
                    "epoch": epoch,
                    "metrics/mAP50(B)": epoch / 200,
                    "metrics/mAP50-95(B)": 0.9 if epoch == 142 else epoch / 1000,
                }
            )

    summary = summarize_results_csv(path)

    assert summary["epochs"] == 150
    assert summary["first_epoch"] == 1
    assert summary["last_epoch"] == 150
    assert summary["best_epoch"] == 142
    assert summary["best_map50_95"] == 0.9


def test_manifest_hashes_release_files_and_excludes_git_and_manifest(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("release\n", encoding="utf-8")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "best.pt").write_bytes(b"weights")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("private", encoding="utf-8")
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    manifest_path = release_dir / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")

    manifest = build_file_manifest(tmp_path, manifest_path=manifest_path)

    paths = [entry["path"] for entry in manifest["files"]]
    assert paths == ["README.md", "models/best.pt"]
    assert all(len(entry["sha256"]) == 64 for entry in manifest["files"])


def test_secret_path_scan_reports_absolute_user_paths(tmp_path: Path) -> None:
    (tmp_path / "clean.md").write_text("portable content", encoding="utf-8")
    (tmp_path / "leak.json").write_text(
        json.dumps({"path": "/Users/alvin/private/model.pt"}),  # pragma: allowlist publication-scan
        encoding="utf-8",
    )

    findings = scan_forbidden_text(tmp_path)

    assert findings == [{"path": "leak.json", "pattern": "/Users/"}]  # pragma: allowlist publication-scan


def test_release_scan_and_manifest_ignore_generated_outputs(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("public", encoding="utf-8")
    output = tmp_path / "outputs" / "example.json"
    output.parent.mkdir()
    output.write_text("/Users/private/generated", encoding="utf-8")  # pragma: allowlist publication-scan

    findings = scan_forbidden_text(tmp_path)
    manifest = build_file_manifest(tmp_path, manifest_path=tmp_path / "release/manifest.json")

    assert findings == []
    assert [entry["path"] for entry in manifest["files"]] == ["README.md"]
