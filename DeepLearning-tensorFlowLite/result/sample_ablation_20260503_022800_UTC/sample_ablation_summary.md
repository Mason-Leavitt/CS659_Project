# Sample Ablation Summary

## Study Type
This is a sample-based no-retraining sensitivity study for saved TFLite and HOG+SVM artifacts.

## Inputs
Manifest: `D:\BackupFiles\School2023-Present\05_Spring2026\CS659\CS659_Project\DeepLearning-tensorFlowLite\result\manifests\plantnet_300K_manifest_test.csv`
Output directory: `D:\BackupFiles\School2023-Present\05_Spring2026\CS659\CS659_Project\DeepLearning-tensorFlowLite\result\sample_ablation_20260503_022800_UTC`
Created at: `2026-05-03T02:28:00Z`

## Models Evaluated
TFLite available: True
HOG+SVM available: True
Labels available: True

## Ablation Variables
TFLite color correction: none, gray_world, max_rgb
TFLite top-k: 1, 3, 5
HOG+SVM top-k: 1, 3, 5

## Number of Images Attempted
10 manifest row(s) were attempted and 10 image(s) were successfully evaluated.

## Key Results
Model agreement rate was not computed.
TFLite predictions were stable across color correction was not computed.
TFLite top-1 accuracy was not computed.
HOG+SVM top-1 accuracy: 0.0%.
TFLite top-k accuracy was not computed.
HOG+SVM top-k accuracy: 0.0%.

## Failures / Skipped Items
90 failure record(s) were captured in `failure_report.csv`.
No components were skipped.

## Caveats
Model agreement rate was not computed because both models were not available on the same evaluated images.
TFLite color stability was not computed because complete color-correction predictions were unavailable.