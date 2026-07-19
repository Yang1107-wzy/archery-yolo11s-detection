from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

import yaml
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ROBOFLOW_HASH = re.compile(r"\.rf\.[0-9a-f]{32}$", re.IGNORECASE)
SAMPLE_LIMIT = 100


def _names(payload: dict[str, Any]) -> list[str]:
    raw = payload.get("names", [])
    if isinstance(raw, dict):
        return [str(raw[key]) for key in sorted(raw, key=lambda value: int(value))]
    return [str(value) for value in raw]


def _root(data_yaml: Path, payload: dict[str, Any]) -> Path:
    configured = payload.get("path")
    if not configured:
        return data_yaml.parent.resolve()
    path = Path(str(configured)).expanduser()
    return (path if path.is_absolute() else data_yaml.parent / path).resolve()


def _split_value(payload: dict[str, Any], split: str) -> Any:
    return payload.get("val" if split == "valid" else split)


def _image_dirs(root: Path, value: Any) -> list[Path]:
    values = value if isinstance(value, list) else [value]
    dirs: list[Path] = []
    for item in values:
        if item is None:
            continue
        path = Path(str(item))
        path = (path if path.is_absolute() else root / path).resolve()
        dirs.append(path)
    return dirs


def _label_path(image: Path) -> Path:
    parts = list(image.parts)
    if "images" in parts:
        reverse_index = parts[::-1].index("images")
        index = len(parts) - reverse_index - 1
        parts[index] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image.with_suffix(".txt")


def _summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None, "p05": None, "p95": None}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        return ordered[round((len(ordered) - 1) * fraction)]

    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
        "median": median(values),
        "p05": percentile(0.05),
        "p95": percentile(0.95),
    }


def _bbox_summary(rows: list[tuple[float, float]]) -> dict[str, Any]:
    widths = [width for width, _ in rows]
    heights = [height for _, height in rows]
    areas = [width * height for width, height in rows]
    aspects = [width / height for width, height in rows if height > 0]
    return {
        "count": len(rows),
        "width": _summarize(widths),
        "height": _summarize(heights),
        "area": _summarize(areas),
        "aspect_ratio": _summarize(aspects),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dhash(path: Path) -> int:
    with Image.open(path) as opened:
        pixels = list(opened.convert("L").resize((9, 8)).get_flattened_data())
    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value = (value << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return value


def _source_key(path: Path) -> str:
    return ROBOFLOW_HASH.sub("", path.stem)


def _duplicate_analysis(images: list[tuple[str, Path]], root: Path) -> dict[str, Any]:
    records = [
        {
            "split": split,
            "path": path.relative_to(root).as_posix(),
            "sha256": _sha256(path),
            "dhash": _dhash(path),
            "source_key": _source_key(path),
        }
        for split, path in images
    ]
    exact: dict[str, list[dict[str, Any]]] = {}
    sources: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        exact.setdefault(str(record["sha256"]), []).append(record)
        sources.setdefault(str(record["source_key"]), []).append(record)
    exact_groups = [group for group in exact.values() if len(group) > 1]
    exact_cross = [group for group in exact_groups if len({str(item["split"]) for item in group}) > 1]
    source_groups = [group for group in sources.values() if len(group) > 1]
    source_cross = [group for group in source_groups if len({str(item["split"]) for item in group}) > 1]
    near_pairs: list[dict[str, Any]] = []
    for index, left in enumerate(records):
        for right in records[index + 1 :]:
            if left["split"] == right["split"]:
                continue
            distance = int(left["dhash"] ^ right["dhash"]).bit_count()
            if distance <= 4:
                near_pairs.append(
                    {
                        "left": left["path"],
                        "left_split": left["split"],
                        "right": right["path"],
                        "right_split": right["split"],
                        "hamming_distance": distance,
                        "byte_identical": left["sha256"] == right["sha256"],
                    }
                )
    serialize_group = lambda group: [
        {"path": str(item["path"]), "split": str(item["split"])} for item in group
    ]
    return {
        "method": {
            "exact": "SHA-256 of encoded image bytes",
            "near": "64-bit difference hash, cross-split Hamming distance <= 4; includes byte-identical pairs",
            "offline_augmentation": "Roboflow filename grouped after stripping .rf.<32-hex> suffix",
        },
        "exact_group_count": len(exact_groups),
        "exact_cross_split_group_count": len(exact_cross),
        "exact_cross_split_samples": [serialize_group(group) for group in exact_cross[:SAMPLE_LIMIT]],
        "near_cross_split_pair_count": len(near_pairs),
        "near_cross_split_samples": near_pairs[:SAMPLE_LIMIT],
        "offline_augmentation_group_count": len(source_groups),
        "offline_augmentation_cross_split_group_count": len(source_cross),
        "offline_augmentation_cross_split_samples": [serialize_group(group) for group in source_cross[:SAMPLE_LIMIT]],
        "samples_truncated_at": SAMPLE_LIMIT,
    }


def audit_yolo_dataset(data_yaml: str | Path) -> dict[str, Any]:
    yaml_path = Path(data_yaml).expanduser().resolve()
    if not yaml_path.is_file():
        return {"status": "blocked", "issues": [{"code": "missing_data_yaml", "path": str(yaml_path)}]}
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    names = _names(payload)
    root = _root(yaml_path, payload)
    issues: list[dict[str, Any]] = []
    split_stats: dict[str, dict[str, int]] = {}
    instances: Counter[int] = Counter()
    bbox_rows: list[tuple[float, float]] = []
    bbox_by_split: dict[str, list[tuple[float, float]]] = {split: [] for split in ("train", "valid", "test")}
    bbox_by_class: dict[int, list[tuple[float, float]]] = {index: [] for index in range(len(names))}
    image_records: list[tuple[str, Path]] = []
    for split in ("train", "valid", "test"):
        dirs = _image_dirs(root, _split_value(payload, split))
        images: list[Path] = []
        for directory in dirs:
            if not directory.exists():
                issues.append({"code": "missing_split_path", "split": split, "path": str(directory)})
                continue
            if directory.is_file():
                images.extend(
                    Path(line.strip()) for line in directory.read_text(encoding="utf-8").splitlines() if line.strip()
                )
            else:
                images.extend(path for path in directory.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
        image_records.extend((split, image) for image in images)
        labels_found = 0
        empty_labels = 0
        for image in sorted(images, key=lambda item: item.as_posix()):
            try:
                with Image.open(image) as opened:
                    opened.verify()
            except Exception as exc:
                issues.append({"code": "corrupt_image", "path": str(image), "detail": str(exc)})
            label = _label_path(image)
            if not label.is_file():
                issues.append({"code": "image_without_label", "path": str(image)})
                continue
            labels_found += 1
            lines = [line.strip() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                empty_labels += 1
            for line_number, line in enumerate(lines, start=1):
                fields = line.split()
                if len(fields) != 5:
                    issues.append({"code": "invalid_field_count", "path": str(label), "line": line_number})
                    continue
                try:
                    class_id = int(fields[0])
                    x, y, width, height = (float(value) for value in fields[1:])
                except ValueError:
                    issues.append({"code": "non_numeric_label", "path": str(label), "line": line_number})
                    continue
                if class_id < 0 or class_id >= len(names):
                    issues.append({"code": "class_id_out_of_range", "path": str(label), "line": line_number})
                else:
                    instances[class_id] += 1
                    bbox_rows.append((width, height))
                    bbox_by_split[split].append((width, height))
                    bbox_by_class[class_id].append((width, height))
                if any(value < 0.0 or value > 1.0 for value in (x, y, width, height)):
                    issues.append({"code": "coordinate_out_of_range", "path": str(label), "line": line_number})
                if width <= 0.0 or height <= 0.0:
                    issues.append({"code": "non_positive_bbox", "path": str(label), "line": line_number})
                if x - width / 2 < 0 or x + width / 2 > 1 or y - height / 2 < 0 or y + height / 2 > 1:
                    issues.append({"code": "bbox_outside_image", "path": str(label), "line": line_number})
        split_stats[split] = {"images": len(images), "labels": labels_found, "empty_labels": empty_labels}
    blocking_codes = {
        "missing_split_path",
        "corrupt_image",
        "invalid_field_count",
        "non_numeric_label",
        "class_id_out_of_range",
        "coordinate_out_of_range",
        "non_positive_bbox",
        "bbox_outside_image",
    }
    status = "failed" if any(item["code"] in blocking_codes for item in issues) else "passed"
    return {
        "status": status,
        "data_yaml": str(yaml_path),
        "root": str(root),
        "names": names,
        "splits": split_stats,
        "class_instances": {name: instances[index] for index, name in enumerate(names)},
        "bbox_statistics": {
            "overall": _bbox_summary(bbox_rows),
            "by_split": {split: _bbox_summary(rows) for split, rows in bbox_by_split.items()},
            "by_class": {names[index]: _bbox_summary(bbox_by_class[index]) for index in range(len(names))},
        },
        "duplicates": _duplicate_analysis(image_records, root),
        "issues": issues,
    }
