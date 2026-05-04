# Sample Ablation Summary

## Study Type
This is a sample-based no-retraining sensitivity study for saved TFLite and HOG+SVM artifacts.

## Inputs
Manifest: `D:\BackupFiles\School2023-Present\05_Spring2026\CS659\CS659_Project\DeepLearning-tensorFlowLite\result\parameter_sweep_20260504_042043_UTC\point_004_top_k_8\point_004_top_k_8_manifest.csv`
Output directory: `D:\BackupFiles\School2023-Present\05_Spring2026\CS659\CS659_Project\DeepLearning-tensorFlowLite\result\parameter_sweep_20260504_042043_UTC\point_004_top_k_8`
Created at: `2026-05-04T04:21:33Z`

## Models Evaluated
TFLite available: True
HOG+SVM available: True
Labels available: True

## Ablation Variables
Sampling mode: balanced
Sampling seed: 50
Max images per class: 5
TFLite color correction: none, gray_world, max_rgb
TFLite top-k: 1, 8
HOG+SVM top-k: 1, 8

## Number of Images Attempted
100 manifest row(s) were attempted and 100 image(s) were successfully evaluated.

## Key Results
Model agreement rate: 1.0% (1/100 images).
TFLite predictions were stable across color correction: 25.0% (25/100 images).
TFLite top-1 accuracy: 25.0%.
HOG+SVM top-1 accuracy: 0.0%.
TFLite top-k accuracy: 61.0%.
HOG+SVM top-k accuracy: 0.0%.

## Observations
Model agreement rate: 1.0%.
TFLite color stability: 25.0%.
TFLite top-k accuracy exceeded HOG+SVM by 61.0 percentage points on this sample.
Sample-size caveat: this run evaluated 100 image(s) using balanced sampling, so small samples may still underrepresent the full test split.

## Failures / Skipped Items
No per-image failures were recorded.
No components were skipped.

## Caveats
Model agreement rate: 1.0% (1/100 images).
TFLite predictions were stable across color correction: 25.0% (25/100 images).