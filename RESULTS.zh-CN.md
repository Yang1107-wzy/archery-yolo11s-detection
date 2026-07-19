# 结果与评价协议

完整 150 行 `results.csv` 中，Epoch 142 的 validation mAP50–95 最高：Precision 0.73944、Recall 0.78580、mAP50 0.80121、mAP50–95 0.43646。validation 用于选模型，不是独立 test 结论。

标准 AP test 使用 test split、置信度下限 0.001、NMS IoU 0.70，共 65 张图片、200 个实例：Precision 0.7811169、Recall 0.6726637、mAP50 0.7590067、mAP50–95 0.4131430，机器可读证据见 `evaluation/standard_ap_test.json`。

validation-only 选择后，部署工作点锁定为 confidence 0.25、NMS IoU 0.50。test Precision 为 0.7816353、Recall 为 0.6723284、F1 为 0.7228731。过滤后的 mAP50 0.6750289、mAP50–95 0.3773739 是在置信度过滤后重新计算的，不能直接与标准 AP 比较。计分区 macro F1 为 0.6920871，`target` F1 为 0.9846154。

标准 AP test 相比最佳 validation，mAP50 低 4.22 个百分点，mAP50–95 低 2.33 个百分点。这与 test 子集更难或分布不同相符，但现有数据不足以作因果归因。逐类别结果与错误分析见 `evaluation/source_reports/`。

