# Results and Evaluation Protocols

## Validation model selection

Epoch 142 is the maximum validation mAP50–95 row in the complete 150-row `results.csv`: precision 0.73944, recall 0.78580, mAP50 0.80121 and mAP50–95 0.43646. Validation selected the checkpoint and is not an independent test claim.

## Independent test: standard AP

The standard Ultralytics test run used the test split, confidence floor 0.001 and NMS IoU 0.70. It produced precision 0.7811169, recall 0.6726637, mAP50 0.7590067 and mAP50–95 0.4131430 over 65 images and 200 instances. Machine-readable evidence is in `evaluation/standard_ap_test.json`.

## Independent test: locked operating point

After validation-only selection, the deployment point was locked to confidence 0.25 and NMS IoU 0.50. Test precision was 0.7816353, recall 0.6723284 and F1 0.7228731. Its filtered mAP50 0.6750289 and mAP50–95 0.3773739 were recomputed after confidence filtering and must not be compared directly with standard AP. The score-region macro F1 was 0.6920871 and `target` F1 was 0.9846154.

## Interpretation

The independent standard-AP test is 4.22 percentage points lower than best validation for mAP50 and 2.33 points lower for mAP50–95. This gap is consistent with a harder or differently distributed test subset, but the dataset does not support a causal attribution. Per-class tables and error analysis are retained under `evaluation/source_reports/`.

