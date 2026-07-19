# 基于 YOLO11s 的射箭靶面与箭支检测

[English](README.md) · [模型卡](MODEL_CARD.zh-CN.md) · [数据集卡](DATASET_CARD.zh-CN.md) · [结果](RESULTS.zh-CN.md) · [复现说明](REPRODUCIBILITY.zh-CN.md)

这是一个以证据链和可复现性为核心的 YOLO11s 150-epoch 学术发布包，用于检测射箭靶面和计分区域。仓库包含 Target and Arrow Detection v6 完整解压数据、训练与评估代码、Streamlit 演示程序、最终权重、150 个 epoch 的完整记录、训练图表和独立 test 结果。

**作者：** Zhengyang Wang、Jiacheng Yao；单位：北京师范大学-香港浸会大学联合国际学院（BNBU）。

## 研究范围与结论边界

- 本项目是在 Target v6 上训练的独立 YOLO11s 复现，**不是 Roboflow 原作者的 checkpoint**。
- validation 用于选择模型；test 单独报告，不能把 validation 指标当成 test 指标。
- 网页曲线截图的估读仅用于探索性比较，不属于正式复现证据。
- 当前计分点取自预测**检测框中心**，并不等同于经过物理标注的箭支真实撞击点。
- Epoch 81 的 `time` 重新计时表示训练进程恢复；模型与优化器状态已恢复，并非重新初始化。

## 主要结果

| 评价方式 | 数据划分 / 选择方式 | Precision | Recall | F1 | mAP50 | mAP50–95 |
|---|---|---:|---:|---:|---:|---:|
| 最佳 validation checkpoint | validation，Epoch 142 | 73.94% | 78.58% | — | 80.12% | 43.65% |
| 标准 AP 评估 | 独立 test，置信度下限 0.001，NMS IoU 0.70 | 78.11% | 67.27% | — | 75.90% | 41.31% |
| 锁定部署工作点 | 独立 test，`conf=0.25`、`iou=0.50` | 78.16% | 67.23% | 72.29% | 67.50%* | 37.74%* |

\* 锁定工作点的 AP 是在置信度过滤后重新计算的，不能与标准 AP 直接横向比较，详见 [RESULTS.zh-CN.md](RESULTS.zh-CN.md)。

## 数据集

Target and Arrow Detection v6 共 1,645 张 \(640\times640\) 图片，每张图片均有对应 YOLO 标签：train 1,482、validation 98、test 65。12 类顺序为 `0, 1, 10, 2, 3, 4, 5, 6, 7, 8, 9, target`。

数据来源为 [Roboflow Universe: Target and Arrow Detection v6](https://universe.roboflow.com/archery-zrbei/target-and-arrow-detection/dataset/6)，保留原提供者署名与 MIT 数据集元数据。完整解压图片通过 Git LFS 管理；不重复上传 187 MiB 原始 ZIP，只在数据集卡中保存其 SHA-256。

## 模型结构与训练

YOLO11s 可以零基础理解为三部分：Backbone 从图片提取边缘、纹理和语义特征；Neck 融合不同尺度特征；Detect Head 在多个分辨率上预测类别和边界框。训练参数为 150 epochs、batch 8、输入 640、seed 42。Ultralytics 在 `optimizer=auto` 下选择 AdamW，有效初始学习率为 0.000625；配置中的 `lr0=0.01` 不是自动优化器最终采用的有效初始学习率。

训练在 Epoch 80 与 81 之间恢复。`best.pt` 由 validation mAP50–95 在 Epoch 142 选出。仓库不发布 150 个中间 checkpoint，但保留 `best.pt`、`last.pt`、配置、优化器摘要、图表和完整 150 行 `results.csv`。

## 安装与运行

要求 Python 3.12，并在克隆前安装 Git LFS：

```bash
git lfs install
git clone https://github.com/Yang1107-wzy/archery-yolo11s-detection.git
cd archery-yolo11s-detection
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[demo,dev]'
make verify
```

训练接口带有防误触保护：

```bash
python -m archery_ml.cli train --config configs/train/yolo11s_target_v6_150e.yaml --smoke
ARCHERY_RUN_FULL=1 python -m archery_ml.cli train --config configs/train/yolo11s_target_v6_150e.yaml
```

推理会输出 JSON 和标注图：

```bash
python -m archery_ml.cli infer --model models/yolo11s-target-v6/best.pt --source data/target-arrow-detection-v6/test/images --output outputs/example --device cpu
```

还可使用 `make test`、`make verify`、`make infer-example` 和 `make demo`。

## 局限性

公开数据划分可能在不同 split 间包含视觉近似图片或离线增强版本，因此这些 test 结果不能表述为跨被试、跨场地或跨拍摄会话泛化。数据规模较小、类别不均衡，本模型也未针对正式比赛判分或安全关键应用完成校准。若要估计真实物理撞击点，需要专门的点标注与几何标定。

## 许可与引用

项目代码和微调权重采用 GNU AGPL-3.0；数据文件保留来源数据集的 MIT 条款与署名。详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 [CITATION.cff](CITATION.cff)。本版本不填写 ORCID、邮箱或 DOI。

