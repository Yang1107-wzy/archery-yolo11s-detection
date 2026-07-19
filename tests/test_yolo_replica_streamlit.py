from __future__ import annotations

from pathlib import Path

from archery_ml.demo.yolo_replica_streamlit import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    PAGE_TABS,
    build_yolo_streamlit_command,
)


ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_command_uses_independent_entrypoint_and_port() -> None:
    command = build_yolo_streamlit_command()

    assert command[:3] == ["streamlit", "run", "app/yolo_replica_demo.py"]
    assert command[command.index("--server.address") + 1] == "::"
    assert command[command.index("--server.port") + 1] == "8504"
    assert DEFAULT_HOST == "::"
    assert DEFAULT_PORT == 8504


def test_page_contract_contains_single_batch_and_metrics_tabs() -> None:
    assert PAGE_TABS == ("单张推理", "批量推理", "模型指标")


def test_public_app_and_launcher_exist() -> None:
    assert (ROOT / "app/yolo_replica_demo.py").is_file()
    assert (ROOT / "scripts/run_yolo_replica_demo.sh").is_file()
