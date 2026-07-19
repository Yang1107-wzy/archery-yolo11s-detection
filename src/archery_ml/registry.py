from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


REQUIRED_RUN_FIELDS = {
    "run_id",
    "config_hash",
    "dataset_fingerprint",
    "checkpoint",
    "metrics",
    "seed",
    "mode",
}
SECRET_FRAGMENTS = ("api_key", "apikey", "secret", "token", "password")


def _assert_no_secrets(value: Any, path: str = "record") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower()
            if any(fragment in normalized for fragment in SECRET_FRAGMENTS):
                raise ValueError(f"secret-like field is not allowed: {path}.{key}")
            _assert_no_secrets(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_secrets(child, f"{path}[{index}]")


def append_run_record(path: str | Path, record: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_RUN_FIELDS - set(record))
    if missing:
        raise ValueError(f"missing required run fields: {', '.join(missing)}")
    _assert_no_secrets(record)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(record)
    payload.setdefault("recorded_at", datetime.now().astimezone().isoformat(timespec="seconds"))
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def update_run_record(path: str | Path, run_id: str, updates: dict[str, Any]) -> None:
    _assert_no_secrets(updates)
    target = Path(path)
    rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]
    matches = [index for index, row in enumerate(rows) if row.get("run_id") == run_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one registry line for {run_id}, found {len(matches)}")
    index = matches[0]
    rows[index] = {**rows[index], **updates, "run_id": run_id}
    missing = sorted(REQUIRED_RUN_FIELDS - set(rows[index]))
    if missing:
        raise ValueError(f"missing required run fields after update: {', '.join(missing)}")
    target.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def update_data_registry(path: str | Path, dataset_name: str, state: dict[str, Any]) -> None:
    _assert_no_secrets(state)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "datasets": {}}
    if target.is_file() and target.stat().st_size:
        payload = json.loads(target.read_text(encoding="utf-8"))
        payload.setdefault("schema_version", 1)
        payload.setdefault("datasets", {})
    entry = dict(state)
    entry["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    payload["datasets"][dataset_name] = entry
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_project_memory(path: str | Path, entry: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    prefix = ""
    if target.is_file() and target.stat().st_size:
        existing = target.read_text(encoding="utf-8")
        prefix = "" if existing.endswith("\n\n") else ("\n" if existing.endswith("\n") else "\n\n")
    with target.open("a", encoding="utf-8") as handle:
        handle.write(prefix + entry.rstrip() + "\n")
