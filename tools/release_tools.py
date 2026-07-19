from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any


TEXT_SUFFIXES = {
    "",
    ".cff",
    ".csv",
    ".html",
    ".json",
    ".md",
    ".py",
    ".sha256",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
ALLOWLIST_PRAGMA = "pragma: allowlist publication-scan"
USER_PATH_MARKER = "/" + "Users" + "/"
GITHUB_TOKEN_MARKER = "gh" + "o_"
OPENAI_KEY_MARKER = "s" + "k-"
SLACK_TOKEN_MARKER = "xo" + "xb-"
AWS_KEY_MARKER = "AK" + "IA"
IGNORED_DIRECTORY_NAMES = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "outputs",
}


def _is_ignored(relative: Path) -> bool:
    return any(part in IGNORED_DIRECTORY_NAMES or part.startswith(".venv") for part in relative.parts)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_portable_paths(value: Any, mapping: dict[str, str]) -> Any:
    """Recursively replace approved absolute source paths with release-relative paths."""
    if isinstance(value, dict):
        return {key: normalize_portable_paths(item, mapping) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_portable_paths(item, mapping) for item in value]
    if isinstance(value, tuple):
        return tuple(normalize_portable_paths(item, mapping) for item in value)
    if isinstance(value, str):
        normalized = value
        for source, replacement in sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True):
            normalized = normalized.replace(source, replacement)
        return normalized
    return value


def summarize_results_csv(path: str | Path) -> dict[str, int | float]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 150:
        raise ValueError(f"expected 150 result rows, found {len(rows)}")
    epochs = [int(row["epoch"]) for row in rows]
    if epochs != list(range(1, 151)):
        raise ValueError("epochs must be contiguous from 1 through 150")
    best = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    return {
        "epochs": len(rows),
        "first_epoch": epochs[0],
        "last_epoch": epochs[-1],
        "best_epoch": int(best["epoch"]),
        "best_map50": float(best["metrics/mAP50(B)"]),
        "best_map50_95": float(best["metrics/mAP50-95(B)"]),
    }


def build_file_manifest(root: str | Path, *, manifest_path: str | Path) -> dict[str, Any]:
    release_root = Path(root).resolve()
    excluded_manifest = Path(manifest_path).resolve()
    files: list[dict[str, Any]] = []
    for path in sorted(candidate for candidate in release_root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(release_root)
        if _is_ignored(relative):
            continue
        if path.resolve() == excluded_manifest:
            continue
        files.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {"schema_version": 1, "file_count": len(files), "files": files}


def scan_forbidden_text(
    root: str | Path,
    patterns: tuple[str, ...] = (
        USER_PATH_MARKER,
        GITHUB_TOKEN_MARKER,
        OPENAI_KEY_MARKER,
        SLACK_TOKEN_MARKER,
        AWS_KEY_MARKER,
    ),
) -> list[dict[str, str]]:
    release_root = Path(root).resolve()
    findings: list[dict[str, str]] = []
    for path in sorted(candidate for candidate in release_root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(release_root)
        if _is_ignored(relative):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > 20 * 1024 * 1024:
            continue
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for pattern in patterns:
            if any(pattern in line and ALLOWLIST_PRAGMA not in line for line in lines):
                findings.append({"path": relative.as_posix(), "pattern": pattern})
    return findings
