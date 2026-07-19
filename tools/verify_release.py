from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from tools.release_tools import scan_forbidden_text, sha256_file, summarize_results_csv


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class ReleaseContract:
    split_counts: dict[str, int]
    class_names: tuple[str, ...]
    best_sha256: str
    last_sha256: str


FORMAL_CONTRACT = ReleaseContract(
    split_counts={"train": 1482, "valid": 98, "test": 65},
    class_names=("0", "1", "10", "2", "3", "4", "5", "6", "7", "8", "9", "target"),
    best_sha256="699235268b229cbb5e401d4fb0559d788630de04267605d2927aad60aa262b20",
    last_sha256="6e1db278f809bfd9474c8a0873adb10dd2e34e5e1cab82055e8c32ff91ab0384",
)


def _issue(issues: list[dict[str, Any]], code: str, **details: Any) -> None:
    issues.append({"code": code, **details})


def _dataset_checks(root: Path, contract: ReleaseContract, issues: list[dict[str, Any]]) -> dict[str, Any]:
    dataset = root / "data/target-arrow-detection-v6"
    yaml_path = dataset / "data.yaml"
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) if yaml_path.is_file() else {}
    names = tuple(str(item) for item in (payload or {}).get("names", []))
    if names != contract.class_names:
        _issue(issues, "class_mapping", expected=list(contract.class_names), actual=list(names))
    total_images = 0
    total_labels = 0
    exif_images = 0
    gps_images = 0
    split_report: dict[str, Any] = {}
    for split, expected in contract.split_counts.items():
        image_dir = dataset / split / "images"
        label_dir = dataset / split / "labels"
        images = sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
        labels = sorted(label_dir.glob("*.txt"))
        total_images += len(images)
        total_labels += len(labels)
        if len(images) != expected or len(labels) != expected:
            _issue(
                issues,
                "split_count",
                split=split,
                expected=expected,
                images=len(images),
                labels=len(labels),
            )
        for image in images:
            with Image.open(image) as opened:
                exif = opened.getexif()
                if exif:
                    exif_images += 1
                if 34853 in exif:
                    gps_images += 1
        for label in labels:
            for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                fields = line.split()
                try:
                    class_id = int(fields[0])
                    coords = [float(value) for value in fields[1:]]
                except (ValueError, IndexError):
                    _issue(issues, "invalid_label", path=label.relative_to(root).as_posix(), line=line_number)
                    continue
                if len(fields) != 5 or class_id < 0 or class_id >= len(contract.class_names):
                    _issue(issues, "invalid_label", path=label.relative_to(root).as_posix(), line=line_number)
                if len(coords) != 4 or any(value < 0.0 or value > 1.0 for value in coords) or coords[2] <= 0 or coords[3] <= 0:
                    _issue(issues, "invalid_label", path=label.relative_to(root).as_posix(), line=line_number)
        split_report[split] = {"images": len(images), "labels": len(labels)}
    if exif_images or gps_images:
        _issue(issues, "image_metadata", exif_images=exif_images, gps_images=gps_images)
    license_path = dataset / "source_metadata/README.dataset.txt"
    if not license_path.is_file() or "License: MIT" not in license_path.read_text(encoding="utf-8", errors="ignore"):
        _issue(issues, "dataset_license")
    return {
        "images": total_images,
        "labels": total_labels,
        "splits": split_report,
        "class_names": list(names),
        "exif_images": exif_images,
        "gps_images": gps_images,
    }


def _verify_manifest(root: Path, issues: list[dict[str, Any]]) -> dict[str, Any]:
    path = root / "release/manifest.json"
    if not path.is_file():
        return {"present": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    checked = 0
    for entry in payload.get("files", []):
        candidate = root / str(entry["path"])
        if not candidate.is_file() or sha256_file(candidate) != entry.get("sha256"):
            _issue(issues, "manifest_mismatch", path=str(entry.get("path")))
        checked += 1
    return {"present": True, "checked": checked}


def verify_release(
    root: str | Path,
    *,
    contract: ReleaseContract = FORMAL_CONTRACT,
) -> dict[str, Any]:
    release_root = Path(root).expanduser().resolve()
    issues: list[dict[str, Any]] = []
    dataset = _dataset_checks(release_root, contract, issues)
    results_path = release_root / "training/150epoch-seed42/results.csv"
    try:
        training = summarize_results_csv(results_path)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        training = {"error": str(exc)}
        _issue(issues, "training_results", detail=str(exc))
    model_root = release_root / "models/yolo11s-target-v6"
    model_hashes: dict[str, str | None] = {}
    for filename, expected in (("best.pt", contract.best_sha256), ("last.pt", contract.last_sha256)):
        path = model_root / filename
        actual = sha256_file(path) if path.is_file() else None
        model_hashes[filename] = actual
        if actual != expected:
            _issue(issues, "model_hash", file=filename, expected=expected, actual=actual)
    intermediate = sorted(path.relative_to(release_root).as_posix() for path in release_root.rglob("epoch*.pt"))
    for path in intermediate:
        _issue(issues, "intermediate_checkpoint", path=path)
    forbidden = scan_forbidden_text(release_root)
    for finding in forbidden:
        _issue(issues, "forbidden_text", **finding)
    manifest = _verify_manifest(release_root, issues)
    return {
        "status": "passed" if not issues else "failed",
        "dataset": dataset,
        "training": training,
        "models": model_hashes,
        "privacy": {
            "forbidden_text_findings": forbidden,
            "exif_images": dataset["exif_images"],
            "gps_images": dataset["gps_images"],
        },
        "manifest": manifest,
        "issues": issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the complete public YOLO11s release contract.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify_release(args.root)
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
