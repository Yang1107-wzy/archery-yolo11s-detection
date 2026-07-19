from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from tools.release_tools import normalize_portable_paths, sha256_file, summarize_results_csv


RUN_ID = "replica_20260719_002149_184e8ce2_full"
DATASET_REL = Path("data/target-arrow-detection-v6")
MODEL_REL = Path("models/yolo11s-target-v6")
TRAINING_REL = Path("training/150epoch-seed42")

CODE_FILES = (
    "src/archery_ml/__init__.py",
    "src/archery_ml/registry.py",
    "src/archery_ml/data/__init__.py",
    "src/archery_ml/data/audit_yolo.py",
    "src/archery_ml/data/dataset_fingerprint.py",
    "src/archery_ml/models/__init__.py",
    "src/archery_ml/models/train_yolo.py",
    "src/archery_ml/models/predict_score.py",
    "src/archery_ml/evaluation/__init__.py",
    "src/archery_ml/evaluation/standard_ap.py",
    "src/archery_ml/evaluation/threshold_sweep.py",
    "src/archery_ml/viz/__init__.py",
    "src/archery_ml/viz/roboflow_train_metrics.py",
)

SOURCE_TEST_FILES = (
    "tests/test_replication_yolo_audit.py",
    "tests/test_replication_predict.py",
    "tests/test_replication_evaluation.py",
    "tests/test_roboflow_train_metrics_plot.py",
)

REPORT_FILES = (
    "reports/02_public_dataset_audit.json",
    "reports/02_public_dataset_audit.md",
    "reports/04_current_validation_selection.json",
    "reports/04_replication_metrics.json",
    "reports/04_replication_report.md",
    "reports/07_error_analysis.json",
    "reports/07_error_analysis.md",
    "reports/08_final_model_card.md",
)


def _copy_file(source: Path, target: Path, inventory: list[dict[str, Any]]) -> None:
    if not source.is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    inventory.append(
        {
            "source_role": source.name,
            "target": target.as_posix(),
            "bytes": source.stat().st_size,
            "source_sha256": sha256_file(source),
        }
    )


def _path_mapping(source_root: Path, source_aliases: tuple[Path, ...] = ()) -> dict[str, str]:
    model = source_root / "artifacts/models/replica_target_v6/default_aug_seed42"
    run = source_root / f"artifacts/training_runs/{RUN_ID}"
    data = source_root / "data/raw/target_arrow_v6"
    processed_data = source_root / "data/processed/target_arrow_v6/data.yaml"
    config = source_root / "configs/train/replica_target_v6_default_aug.yaml"
    mapping = {
        str(model / "best.pt"): f"{MODEL_REL.as_posix()}/best.pt",
        str(model / "last.pt"): f"{MODEL_REL.as_posix()}/last.pt",
        str(run / "weights/best.pt"): f"{MODEL_REL.as_posix()}/best.pt",
        str(run / "weights/last.pt"): f"{MODEL_REL.as_posix()}/last.pt",
        str(processed_data): f"{DATASET_REL.as_posix()}/data.yaml",
        str(data / "data.yaml"): f"{DATASET_REL.as_posix()}/data.yaml",
        str(config): "configs/train/yolo11s_target_v6_150e.yaml",
        str(run): TRAINING_REL.as_posix(),
        str(source_root): ".",
    }
    # The workspace may be reached through a Finder alias/symlink. Cover caller-provided aliases without publishing them.
    for alias in source_aliases:
        literal_alias = str(alias)
        if literal_alias == str(source_root):
            continue
        for key, value in list(mapping.items()):
            if key.startswith(str(source_root)):
                mapping[key.replace(str(source_root), literal_alias, 1)] = value
        mapping[literal_alias] = "."
    return mapping


def _write_portable_dataset_yaml(target: Path) -> None:
    payload = {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 12,
        "names": ["0", "1", "10", "2", "3", "4", "5", "6", "7", "8", "9", "target"],
        "source": {
            "provider": "Roboflow Universe user dataset",
            "workspace": "archery-zrbei",
            "project": "target-and-arrow-detection",
            "version": 6,
            "license": "MIT",
            "url": "https://universe.roboflow.com/archery-zrbei/target-and-arrow-detection/dataset/6",
        },
    }
    target.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _sanitize_structured_file(source: Path, target: Path, mapping: dict[str, str]) -> None:
    if source.suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        normalized = normalize_portable_paths(payload, mapping)
        target.write_text(json.dumps(normalized, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    if source.suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        normalized = normalize_portable_paths(payload, mapping)
        target.write_text(yaml.safe_dump(normalized, sort_keys=False, allow_unicode=True), encoding="utf-8")
        return
    text = source.read_text(encoding="utf-8", errors="replace")
    target.write_text(normalize_portable_paths(text, mapping), encoding="utf-8")


def sync_release(
    source: str | Path,
    destination: str | Path,
    *,
    include_code: bool = True,
) -> dict[str, Any]:
    source_input = Path(source).expanduser().absolute()
    source_root = source_input.resolve()
    release_root = Path(destination).expanduser().resolve()
    release_root.mkdir(parents=True, exist_ok=True)
    inventory: list[dict[str, Any]] = []
    mapping = _path_mapping(source_root, (source_input,))

    source_dataset = source_root / "data/raw/target_arrow_v6"
    target_dataset = release_root / DATASET_REL
    for split in ("train", "valid", "test"):
        for kind in ("images", "labels"):
            shutil.copytree(
                source_dataset / split / kind,
                target_dataset / split / kind,
                dirs_exist_ok=True,
            )
    for name in ("README.dataset.txt", "README.roboflow.txt"):
        _copy_file(source_dataset / name, target_dataset / "source_metadata" / name, inventory)
    ingest_source = source_dataset / "INGEST_MANIFEST.json"
    ingest_target = target_dataset / "source_metadata/INGEST_MANIFEST.json"
    if ingest_source.is_file():
        ingest_target.parent.mkdir(parents=True, exist_ok=True)
        _sanitize_structured_file(ingest_source, ingest_target, mapping)
        inventory.append(
            {
                "source_role": "INGEST_MANIFEST.json",
                "target": ingest_target.relative_to(release_root).as_posix(),
                "bytes": ingest_source.stat().st_size,
                "source_sha256": sha256_file(ingest_source),
            }
        )
    _copy_file(source_dataset / "data.yaml", target_dataset / "source_metadata/original_data.yaml", inventory)
    target_dataset.mkdir(parents=True, exist_ok=True)
    _write_portable_dataset_yaml(target_dataset / "data.yaml")

    source_model = source_root / "artifacts/models/replica_target_v6/default_aug_seed42"
    target_model = release_root / MODEL_REL
    for name in ("best.pt", "last.pt"):
        _copy_file(source_model / name, target_model / name, inventory)

    source_run = source_root / f"artifacts/training_runs/{RUN_ID}"
    target_run = release_root / TRAINING_REL
    for path in sorted(source_run.iterdir()):
        if path.name == "weights":
            continue
        target = target_run / path.name
        if path.is_dir():
            shutil.copytree(path, target, dirs_exist_ok=True)
            for text_path in target.rglob("*"):
                if text_path.is_file() and text_path.suffix.lower() in {".html", ".json", ".csv", ".md", ".txt", ".yaml", ".yml"}:
                    original = source_run / text_path.relative_to(target_run)
                    _sanitize_structured_file(original, text_path, mapping)
        elif path.name in {"args.yaml"}:
            target.parent.mkdir(parents=True, exist_ok=True)
            _sanitize_structured_file(path, target, mapping)
        else:
            _copy_file(path, target, inventory)

    manifest_source = source_root / f"artifacts/run_manifests/{RUN_ID}.json"
    target_manifest = target_run / "run_manifest.json"
    target_manifest.parent.mkdir(parents=True, exist_ok=True)
    _sanitize_structured_file(manifest_source, target_manifest, mapping)
    inventory.append(
        {
            "source_role": "run_manifest",
            "target": target_manifest.relative_to(release_root).as_posix(),
            "bytes": manifest_source.stat().st_size,
            "source_sha256": sha256_file(manifest_source),
        }
    )

    config_source = source_root / "configs/train/replica_target_v6_default_aug.yaml"
    config_target = release_root / "configs/train/yolo11s_target_v6_150e.yaml"
    config_target.parent.mkdir(parents=True, exist_ok=True)
    _sanitize_structured_file(config_source, config_target, mapping)

    for relative in REPORT_FILES:
        source_path = source_root / relative
        if not source_path.is_file():
            continue
        target_path = release_root / "evaluation/source_reports" / source_path.name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        _sanitize_structured_file(source_path, target_path, mapping)

    if include_code:
        for relative in (*CODE_FILES, *SOURCE_TEST_FILES):
            _copy_file(source_root / relative, release_root / relative, inventory)

    provenance = release_root / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    for entry in inventory:
        target_path = Path(str(entry["target"]))
        if target_path.is_absolute() and target_path.is_relative_to(release_root):
            entry["target"] = target_path.relative_to(release_root).as_posix()
    rewrite_payload = {
        "schema_version": 1,
        "policy": "Approved absolute workspace paths were rewritten to repository-relative paths.",
        "rules": [
            {"source_kind": "dataset_yaml", "replacement": "data/target-arrow-detection-v6/data.yaml"},
            {"source_kind": "best_checkpoint", "replacement": "models/yolo11s-target-v6/best.pt"},
            {"source_kind": "last_checkpoint", "replacement": "models/yolo11s-target-v6/last.pt"},
            {"source_kind": "training_run", "replacement": "training/150epoch-seed42"},
            {"source_kind": "workspace_root", "replacement": "."},
        ],
    }
    (provenance / "path_rewrite.json").write_text(
        json.dumps(rewrite_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (provenance / "source_inventory.json").write_text(
        json.dumps({"schema_version": 1, "files": inventory}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "dataset": {"images": 1645, "splits": {"train": 1482, "valid": 98, "test": 65}},
        "training": summarize_results_csv(target_run / "results.csv"),
        "published_model_files": ["best.pt", "last.pt"],
        "excluded_intermediate_checkpoints": True,
    }
    (provenance / "sync_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync the curated YOLO11s academic release mirror.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(sync_release(args.source, args.destination), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
