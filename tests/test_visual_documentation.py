from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAGES = [ROOT / "VISUAL_ANALYSIS.md", ROOT / "VISUAL_ANALYSIS.zh-CN.md"]


def _local_images(markdown: str) -> list[str]:
    return [target for target in re.findall(r"!\[[^]]*\]\(([^)]+)\)", markdown) if "://" not in target]


def test_bilingual_visual_analysis_contract() -> None:
    for page in PAGES:
        assert page.exists()
        text = page.read_text(encoding="utf-8")
        for required in [
            "Epoch 142",
            "0.25",
            "0.50",
            "target 0.94",
            "65",
        ]:
            assert required in text
        for image_target in _local_images(text):
            assert (page.parent / image_target).resolve().exists(), image_target


def test_english_claim_boundaries() -> None:
    text = PAGES[0].read_text(encoding="utf-8").lower()
    for required in [
        "validation",
        "standard ap test",
        "qualitative",
        "dense overlap",
        "bounding-box center",
        "not physical impact",
    ]:
        assert required in text


def test_chinese_claim_boundaries() -> None:
    text = PAGES[1].read_text(encoding="utf-8")
    for required in [
        "validation",
        "标准 AP test",
        "定性案例",
        "密集重叠",
        "检测框中心",
        "不等同于物理撞击点",
    ]:
        assert required in text


def test_readmes_embed_visual_summary_and_language_matched_link() -> None:
    english = (ROOT / "README.md").read_text(encoding="utf-8")
    chinese = (ROOT / "README.zh-CN.md").read_text(encoding="utf-8")
    assert "## Visual evidence" in english
    assert "[Detailed visual analysis](VISUAL_ANALYSIS.md)" in english
    assert "## 可视化证据" in chinese
    assert "[完整可视化分析](VISUAL_ANALYSIS.zh-CN.md)" in chinese
    for text in [english, chinese]:
        assert "docs/assets/visuals/training_dynamics.png" in text
        assert "docs/assets/visuals/local_model_inference_streamlit.png" in text
