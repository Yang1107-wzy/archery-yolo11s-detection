# 可视化证据与指标解读

[English](VISUAL_ANALYSIS.md) · [返回中文 README](README.zh-CN.md)

本页把每张图对应到仓库内可追溯的真实来源：Target v6 标签、150 行训练日志、分协议评估 JSON，或作者本地推理界面。定量图由 `scripts/generate_repository_visuals.py` 生成；机器可读源表与图像哈希保存在 `evaluation/visualization_sources/`。

## 1. 模型实现流程

![YOLO11s 模型流程](docs/assets/visuals/model_pipeline.png)

输入图像先按比例填充到 \(640\times640\)。Backbone 提取由浅到深的视觉特征；Neck 融合高分辨率细节和深层语义；Detect Head 在三个尺度上预测 12 类与边界框；NMS 再按照置信度和 IoU 删除重复框。仓库的 `best.pt` 实现了这条流程，它是在 Target v6 上独立训练的复现权重，不是 Roboflow 原作者 checkpoint。当前点位来自**检测框中心，不等同于物理撞击点**真值。

## 2. 数据集组成

![数据划分与类别分布](docs/assets/visuals/dataset_overview.png)

完整数据包含 1,645 组图片与标签：train 1,482、validation 98、独立 test 65。脚本从全部 YOLO 标签重新统计出 4,583 个检测框，其中 `target` 有 1,791 个实例，各计分区域类别为 135–407 个，说明类别不均衡确实存在。

![三个 split 的固定样例](docs/assets/visuals/dataset_samples.png)

样例图对每个 split 固定选取一张图片，只用于展示数据形态，不用于估计精度，也不能证明不同 split 之间完全跨场地或跨拍摄会话独立。

## 3. 训练过程

![训练与 validation 曲线](docs/assets/visuals/training_dynamics.png)

模型设置为 150 epochs、batch 8、seed 42、输入尺寸 640。细线保留真实逐 epoch 数值；粗线是明确标注的 7-epoch 移动平均，只用于提高可读性。训练损失持续下降，validation mAP50–95 在 **Epoch 142** 达到最高值 0.43646。Epoch 81 虚线表示恢复训练边界：`time` 重新计时，但模型和优化器状态已经恢复，并未重新初始化。

这些曲线说明训练基本收敛，但仅凭曲线不能证明泛化能力。checkpoint 由 validation 选择，独立 test 必须另行报告。

## 4. validation 与独立 test 对比

![严格分开的评估协议](docs/assets/visuals/evaluation_comparison.png)

三种结果回答不同问题：

- **最佳 validation：** 在 Epoch 142 选择 `best.pt`；Precision 0.7394、Recall 0.7858、mAP50 0.8012、mAP50–95 0.4365。
- **标准 AP test：** 在全部 65 张 test 图片上，以置信度下限 0.001、NMS IoU 0.70 积分；Precision 0.7811、Recall 0.6727、mAP50 0.7590、mAP50–95 0.4131。
- **锁定工作点 test：** 使用 `confidence=0.25`、`NMS IoU=0.50` 描述真实部署行为；Precision 0.7816、Recall 0.6723、F1 0.7229。置信度过滤后的 AP 被单独绘制，不能改称标准 AP。

test 的 Recall 和 mAP 低于 validation，说明存在可测量的泛化差距。小规模 65 张 test、类别不均衡、拍摄条件差异，以及公开 split 中可能存在视觉近似样本，都是合理风险；目前证据不能断言其中某一个因素已经被单独证明为根因。

## 5. 类别级诊断

![归一化混淆矩阵](training/150epoch-seed42/confusion_matrix_normalized.png)

![Precision-Recall 曲线](training/150epoch-seed42/BoxPR_curve.png)

![F1-Confidence 曲线](training/150epoch-seed42/BoxF1_curve.png)

这些 Ultralytics 原始图用于观察类别混淆，以及置信度变化对 Precision、Recall、F1 的影响。它们能补充总体指标，但来自 validation 的阈值曲线不能被包装成新的独立 test 指标。

## 6. 本地模型真实推理

![Zhengyang Wang 的本地 Streamlit 真实推理](docs/assets/visuals/local_model_inference_streamlit.png)

**来源标注：** Zhengyang Wang 使用本仓库 `models/yolo11s-target-v6/best.pt` 在本地真实运行。左侧参数为 `confidence=0.25`、`NMS IoU=0.50`。紫色大框 `target 0.94` 表示完整靶面检测，置信度 0.94；`8 0.79` 表示一个 8 分区域预测，置信度 0.79。

这是一张**定性案例**，不能替代 65 张独立 test 的定量评估。它说明模型在户外光照、透视变化和背景干扰下仍能找到完整靶面；同时也暴露了**密集重叠**局限：中心区域多个白色小框与文字相互遮挡，可读性下降，小区域定位也更难人工核验。图中位置属于预测检测框中心，**不等同于物理撞击点**标注，也不能直接作为正式比赛判分。

## 复现定量图

```bash
PYTHONPATH=src:. python3 scripts/generate_repository_visuals.py --root .
```

本地推理截图属于作者提供的定性证据，因此 CI 不自动生成；其余定量图均可由仓库内受版本控制的真实源文件复现。
