from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_academic_documents_are_present() -> None:
    required = {
        "README.md",
        "README.zh-CN.md",
        "MODEL_CARD.md",
        "MODEL_CARD.zh-CN.md",
        "DATASET_CARD.md",
        "DATASET_CARD.zh-CN.md",
        "RESULTS.md",
        "RESULTS.zh-CN.md",
        "REPRODUCIBILITY.md",
        "REPRODUCIBILITY.zh-CN.md",
        "THIRD_PARTY_NOTICES.md",
        "CITATION.cff",
        "LICENSE",
        "Makefile",
        ".github/workflows/verify.yml",
        "release/release_notes_v1.0.0.md",
    }
    missing = sorted(path for path in required if not (ROOT / path).is_file())
    assert not missing


def test_readmes_record_the_result_boundaries_and_authors() -> None:
    english = (ROOT / "README.md").read_text(encoding="utf-8")
    chinese = (ROOT / "README.zh-CN.md").read_text(encoding="utf-8")
    combined = english + chinese
    for marker in (
        "Zhengyang Wang",
        "Jiacheng Yao",
        "Epoch 142",
        "80.12%",
        "43.65%",
        "75.90%",
        "41.31%",
        "78.16%",
        "67.23%",
        "72.29%",
    ):
        assert marker in combined
    assert "README.zh-CN.md" in english
    assert "README.md" in chinese
    assert "not the original Roboflow checkpoint" in english
    assert "不是 Roboflow 原作者的 checkpoint" in chinese
    assert "bounding-box center" in english
    assert "检测框中心" in chinese


def test_lfs_and_ci_contracts() -> None:
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/verify.yml").read_text(encoding="utf-8")
    launcher = (ROOT / "scripts/run_yolo_replica_demo.sh").read_text(encoding="utf-8")
    ci_requirements = (ROOT / "requirements-ci.lock").read_text(encoding="utf-8")
    assert "data/target-arrow-detection-v6/**/images/**" in attributes
    assert "models/yolo11s-target-v6/*.pt" in attributes
    assert "lfs: false" in workflow
    assert ".venv_streamlit_demo" not in launcher
    assert "scipy==1.17.1" in ci_requirements
