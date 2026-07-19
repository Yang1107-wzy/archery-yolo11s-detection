# Visual Evidence Design for the YOLO11s Academic Repository

Date: 2026-07-20  
Status: approved for implementation planning  
Repository: `Yang1107-wzy/archery-yolo11s-detection`

## Objective

Add a compact visual summary to both README files and a detailed bilingual visual-analysis document. Every quantitative chart must be generated from tracked experiment evidence. The supplied Streamlit screenshot must be presented as a qualitative local-model inference example, not as test-set evidence.

## Audience and surfaces

The primary audiences are research-group members, supervisors and GitHub visitors who need to understand the model, data and evaluation without opening raw logs.

The release will use two levels of detail:

1. `README.md` and `README.zh-CN.md`: a short visual overview with three high-value figures and links to the detailed analysis.
2. `VISUAL_ANALYSIS.md` and `VISUAL_ANALYSIS.zh-CN.md`: a complete, section-by-section explanation suitable for a group meeting.

Static PNG figures are the canonical GitHub surface. Each generated quantitative figure will also have a reproducible script and a machine-readable source table or JSON where applicable.

## Visual set

### 1. Project overview

A single architecture-and-evidence flow diagram will explain:

`Target v6 image -> YOLO11s Backbone -> multi-scale Neck -> Detect Head -> NMS -> boxes/classes -> evaluation and local application`

This is an explanatory diagram. It will not claim an exact layer-by-layer reproduction of an unpublished Roboflow checkpoint. The caption will state that the repository uses the Ultralytics YOLO11s architecture and independently trained weights.

### 2. Dataset overview

The dataset figure will contain:

- a three-bar split-count chart: train 1,482, validation 98 and test 65;
- a ranked horizontal class-instance chart computed directly from the tracked YOLO label files;
- a small sample montage drawn only from tracked dataset images, with split and filename provenance recorded.

Counts will be recomputed during generation rather than copied from prose. The figure and caption will note the risk of visually related or offline-augmented variants across public splits.

### 3. Training dynamics

The training figure will be regenerated from `training/150epoch-seed42/results.csv` and will show:

- train box, classification and DFL losses;
- validation box and classification losses;
- precision, recall, mAP50 and mAP50-95;
- a vertical annotation at Epoch 142, the maximum validation mAP50-95 checkpoint;
- a visible resume boundary between Epochs 80 and 81.

Raw epoch values remain visible. Any optional smoothing must be explicitly labelled and must not replace the raw series.

### 4. Evaluation comparison

A grouped dot or bar comparison will separate three protocols:

- best validation checkpoint at Epoch 142;
- independent standard-AP test with confidence floor 0.001 and NMS IoU 0.70;
- independent locked operating-point test with confidence 0.25 and NMS IoU 0.50.

Precision and recall may be compared across all three protocols. mAP values from the locked operating point will be placed in a separately labelled panel because they were recomputed after confidence filtering and are not directly comparable to standard AP. F1 will be shown only where it was formally calculated.

Canonical sources are `evaluation/results_summary.json` and `evaluation/standard_ap_test.json`.

### 5. Diagnostic evidence

The detailed analysis will embed and explain the tracked confusion matrices, precision-recall curve, precision curve, recall curve and F1 curve. Captions will define the axes and explain what a reader should inspect. Existing Ultralytics figures remain source evidence; no visual retouching may alter their values.

### 6. Local-model qualitative inference

The user-supplied Streamlit screenshot will be copied into `docs/assets/visuals/local_model_inference_streamlit.png` without altering the detection content. It will be labelled:

- source: real local inference by Zhengyang Wang;
- model: repository `best.pt` from the 150-epoch YOLO11s run;
- operating point: confidence 0.25 and NMS IoU 0.50;
- `target 0.94`: confidence for the full target-face box;
- score labels such as `8 0.79`: predicted score-region class and confidence;
- overlapping white boxes: a visible dense-scene localization/readability limitation;
- position semantics: bounding-box centers, not physically annotated arrow-impact points;
- evidence boundary: a qualitative application example, not a substitute for the 65-image independent test evaluation.

The screenshot will retain its original aspect ratio. A captioned derivative may add an outer numbered callout strip, but must not cover, remove or redraw original detections.

## Visual style

- white or near-white chart backgrounds;
- dark charcoal text and quiet grey grids;
- blue as the primary quantitative color and gold/orange as the sole comparison accent;
- color is supplemented by markers, labels and line styles;
- no gradients, decorative 3D effects or truncated bar baselines;
- English figures use English labels; Chinese pages provide complete Chinese captions and interpretation;
- exported figures target readable GitHub width and remain legible when scaled to approximately 900 pixels.

## File layout and interfaces

Planned additions:

- `VISUAL_ANALYSIS.md`
- `VISUAL_ANALYSIS.zh-CN.md`
- `docs/assets/visuals/`
- `scripts/generate_repository_visuals.py`
- `evaluation/visualization_sources/`
- tests for source counts, metric values, required captions and output-file existence.

The generation script will accept repository-relative inputs and a repository-relative output directory. It must not require network access or embed local absolute paths. Running it twice on unchanged inputs should produce the same source tables and semantically identical figures.

## README integration

Both README files will receive a visual-results section after the main-result table. The section will show:

1. evaluation comparison;
2. training overview;
3. local-model inference screenshot.

Each image will have a one-paragraph interpretation and a link to the corresponding bilingual detailed-analysis page. README claims must continue to separate validation, standard test AP and the locked deployment operating point.

## Validation and quality assurance

Implementation is complete only when:

- all plotted counts and metrics match the canonical tracked sources;
- generated source tables are machine-readable;
- all figures render without clipped text, overlapping labels or misleading axes;
- the local inference screenshot caption contains every evidence boundary listed above;
- both README variants render valid relative links;
- the release manifest is regenerated;
- existing and new tests pass;
- privacy and absolute-path scans pass;
- final images are visually inspected at original size and GitHub-like scaled size.

GitHub push is a separate external write. Local implementation, verification and commit may proceed after design and plan approval; pushing changes requires a fresh explicit confirmation of repository, branch and content.
