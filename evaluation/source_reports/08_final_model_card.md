# Model card：YOLO11s Target v6 local replica

## 推荐模型

- 名称：`replica_target_v6/default_aug_seed42`
- architecture：YOLO11s detection
- checkpoint：`artifacts/models/replica_target_v6/default_aug_seed42/best.pt`
- checkpoint SHA-256：`699235268b229cbb5e401d4fb0559d788630de04267605d2927aad60aa262b20`
- run_id：`replica_20260719_002149_184e8ce2_full`
- config hash：`184e8ce20505bd90924613fc0035b3143dfabd7168ecf1b87cc6a1b34399ab49`
- dataset fingerprint：`b99178e6c76f9fe94f2b1d961de4209c9e5eeb062e33cd2ded1cb17798367a8f`
- 推荐 operating point：confidence 0.25、NMS IoU 0.50。

## 数据与训练

- 官方 Roboflow Target and Arrow Detection v6：1482 train / 98 valid / 65 public test。
- 输入 640、batch 8、seed 42、MPS、pretrained `yolo11s.pt`。
- 完成 150 epoch；第 80 epoch 后因工作区被 iCloud 移动而从 `last.pt` 续训。
- 数据集 README 标注 MIT；模型和第三方权重的分发仍需同时遵守其各自许可证。

## 指标

- best validation：P 0.73944、R 0.78580、mAP50 0.80121、mAP50-95 0.43646。
- locked valid operating point：P 0.75990、R 0.80369、F1 0.78118、mAP50 0.74383、mAP50-95 0.40743。
- official public test：P 0.78164、R 0.67233、F1 0.72287、mAP50 0.67503、mAP50-95 0.37737。
- public-test target 类：P/R/F1 0.98462、AP50 0.98069、AP50-95 0.93743。

## 适用范围

- 识别本数据分布中的箭靶与 0–10 score 类检测框。
- 输出的 score 坐标为检测框中心，不是经过人工 GT 验证的物理撞击点。
- 真实部署应保留人工复核，尤其是类别 6、10、7、1、8、9、重叠目标、旧孔与环线附近。

## 不适用与风险

- 不是原作者网页模型的私有 `best.pt`，也不是原权重复原。
- 12 类 public-test aggregate 未达到网页参考 P/R/mAP50；只能声明数据与架构复刻以及本地实际结果。
- 官方 split 存在相似图/同源候选，可能使 official test 高估独立场景泛化。
- legacy 5 类 zero-shot 没有 GT，只可人工审阅，不可转写成准确率。
- 模型仍属 experimental，不用于安全关键或自动裁决场景。
