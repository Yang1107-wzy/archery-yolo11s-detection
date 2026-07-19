# Visual Evidence Implementation Plan

> **For agentic workers:** REQUIRED WORKFLOW: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan one task at a time. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add source-backed model, dataset, training, evaluation and real local-inference visuals to the bilingual GitHub repository, then verify and publish them to `main`.

**Architecture:** A focused `archery_ml.viz.repository_visuals` module will extract canonical data and render deterministic static figures. A thin script will orchestrate generation. Bilingual Markdown pages and README sections will embed those outputs and preserve the distinction between validation, standard AP test, locked-threshold test and qualitative local inference.

**Tech Stack:** Python 3.12, Matplotlib, Pillow, pandas, NumPy, PyYAML, pytest, Git LFS, Markdown.

## Global Constraints

- Use only tracked Target v6 labels, `results.csv`, evaluation JSON, published checkpoints and the supplied local screenshot.
- Do not fabricate, manually smooth or visually alter quantitative values.
- Preserve the screenshot's detection content and aspect ratio.
- Label the screenshot as qualitative local inference, not independent test evidence.
- Identify positions as bounding-box centers, not physical arrow-impact annotations.
- Keep all paths repository-relative and pass the existing privacy scan.
- English and Chinese documentation must carry equivalent evidence boundaries.
- Push only to `Yang1107-wzy/archery-yolo11s-detection`, branch `main`, after fresh local verification.

---

### Task 1: Source extraction contracts

**Files:**
- Create: `src/archery_ml/viz/repository_visuals.py`
- Create: `tests/test_repository_visuals.py`

**Interfaces:**
- Produces: `collect_dataset_statistics(dataset_root: Path) -> dict`, `load_training_series(csv_path: Path) -> pandas.DataFrame`, `load_evaluation_protocols(summary_path: Path) -> dict`.
- Consumes: YOLO labels, 150-row Ultralytics CSV and `evaluation/results_summary.json`.

- [ ] **Step 1: Write failing tests**

Add tests that assert split counts `1482/98/65`, class order `0,1,10,2,3,4,5,6,7,8,9,target`, 150 contiguous epochs, best Epoch 142, and exact standard-test mAP50 `0.7590066637476979`.

- [ ] **Step 2: Verify RED**

Run `PYTHONPATH=src:. python3 -m pytest tests/test_repository_visuals.py -q` and confirm import or missing-function failure.

- [ ] **Step 3: Implement extraction functions**

Parse every tracked label line, validate class IDs, count instances by split/class, normalize CSV column whitespace, and return protocol dictionaries without changing values.

- [ ] **Step 4: Verify GREEN**

Run the focused test and confirm all extraction assertions pass.

### Task 2: Deterministic figure generation

**Files:**
- Modify: `src/archery_ml/viz/repository_visuals.py`
- Create: `scripts/generate_repository_visuals.py`
- Modify: `tests/test_repository_visuals.py`
- Create outputs under: `docs/assets/visuals/`
- Create source tables under: `evaluation/visualization_sources/`

**Interfaces:**
- Produces: `render_dataset_overview`, `render_training_dynamics`, `render_evaluation_comparison`, `render_model_pipeline`, `render_sample_montage`, and `generate_all(root: Path) -> list[Path]`.
- Each renderer receives reviewed data structures rather than reading unrelated global state.

- [ ] **Step 1: Write failing output tests**

Assert that `generate_all` creates five named PNG files, two CSV source tables and one JSON manifest; assert PNG dimensions are at least 1200 by 700 and source metrics match canonical values.

- [ ] **Step 2: Verify RED**

Run the focused test and confirm failure because renderers are absent.

- [ ] **Step 3: Implement minimal renderers**

Use a fixed white/charcoal/blue/gold palette. Use zero-baseline bars for counts, raw-plus-labelled-smooth lines for training only when raw values remain visible, a separate locked-AP panel, direct labels, Epoch 142 and resume-boundary annotations, and deterministic sample selection.

- [ ] **Step 4: Generate tracked outputs**

Run `PYTHONPATH=src:. python3 scripts/generate_repository_visuals.py --root .` and inspect its output manifest.

- [ ] **Step 5: Verify GREEN**

Run focused tests and confirm outputs and exact source values pass.

### Task 3: Local inference evidence asset

**Files:**
- Add: `docs/assets/visuals/local_model_inference_streamlit.png`
- Modify: `tests/test_repository_visuals.py`

**Interfaces:**
- Consumes: `/var/folders/dd/cdsf_djs0yj236fjyvhx3vsr0000gn/T/codex-clipboard-ef969386-2afc-4ff0-91e8-3bd4eccd8379.png` during local assembly only.
- Produces: a repository asset with no embedded absolute source path.

- [ ] **Step 1: Write failing asset test**

Assert the destination exists, is a PNG, preserves the source dimensions/aspect ratio and has no EXIF/GPS metadata.

- [ ] **Step 2: Verify RED**

Run the asset test and confirm the destination is missing.

- [ ] **Step 3: Copy and sanitize the image**

Decode the supplied PNG and re-encode it without EXIF metadata while preserving pixels and dimensions. Do not draw over the inference result.

- [ ] **Step 4: Verify GREEN and visually inspect**

Run the asset test, then inspect the original and repository asset side by side.

### Task 4: Bilingual visual-analysis documentation

**Files:**
- Create: `VISUAL_ANALYSIS.md`
- Create: `VISUAL_ANALYSIS.zh-CN.md`
- Modify: `README.md`
- Modify: `README.zh-CN.md`
- Create: `tests/test_visual_documentation.py`

**Interfaces:**
- Consumes: generated assets and evidence summaries.
- Produces: GitHub-renderable relative links and bilingual captions.

- [ ] **Step 1: Write failing documentation-contract tests**

Assert both pages and README sections exist; every linked asset exists; captions contain Epoch 142, validation/test separation, `confidence=0.25`, `NMS IoU=0.50`, `target 0.94`, qualitative-case wording, dense-overlap limitation and bounding-box-center boundary.

- [ ] **Step 2: Verify RED**

Run `PYTHONPATH=src:. python3 -m pytest tests/test_visual_documentation.py -q` and confirm missing-document failures.

- [ ] **Step 3: Write bilingual pages and README summaries**

Introduce model flow, dataset composition, class distribution, training curves, evaluation protocols, confusion/PR/F1 diagnostics, sample montage and the supplied local inference screenshot. Link each README to the language-matched detailed page.

- [ ] **Step 4: Verify GREEN**

Run documentation-contract tests and a relative-link checker.

### Task 5: Release verification and publication

**Files:**
- Modify: `release/manifest.json`
- Modify only if required by new static assets: `.gitattributes`

**Interfaces:**
- Consumes: the final repository tree.
- Produces: updated SHA-256 manifest, verified commit and public GitHub update.

- [ ] **Step 1: Inspect visuals**

Open every generated PNG at original resolution and verify labels, axes, colors and image content are legible and unclipped.

- [ ] **Step 2: Run fresh verification**

Run `make test`, regenerate `release/manifest.json`, run `make verify`, `git diff --check`, the privacy scan, and `git lfs fsck`.

- [ ] **Step 3: Review scope and commit**

Inspect `git status`, staged diff and asset sizes; commit only visual-evidence files with a focused message.

- [ ] **Step 4: Push authorized target**

Push `main` to `git@github.com:Yang1107-wzy/archery-yolo11s-detection.git`.

- [ ] **Step 5: Verify GitHub**

Confirm remote commit, README paths, visual assets and successful GitHub Actions run. Report any non-blocking warnings separately.
