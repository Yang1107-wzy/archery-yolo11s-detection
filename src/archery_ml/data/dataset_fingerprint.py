from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_dataset_root(data_yaml: Path, payload: dict[str, Any]) -> Path:
    configured = payload.get("path")
    if configured:
        candidate = Path(str(configured)).expanduser()
        return (candidate if candidate.is_absolute() else data_yaml.parent / candidate).resolve()
    return data_yaml.parent.resolve()


def fingerprint_yolo_dataset(data_yaml: str | Path) -> dict[str, Any]:
    yaml_path = Path(data_yaml).expanduser().resolve()
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    root = _resolve_dataset_root(yaml_path, payload)
    files: list[dict[str, str]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".txt"}:
            files.append({"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)})
    canonical = json.dumps(
        {"data_yaml_sha256": sha256_file(yaml_path), "files": files},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "data_yaml": str(yaml_path),
        "data_yaml_sha256": sha256_file(yaml_path),
        "file_count": len(files),
        "files": files,
        "overall_fingerprint": hashlib.sha256(canonical).hexdigest(),
    }
