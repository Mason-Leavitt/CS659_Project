# Sample Ablation Summary

## Study Type
This is a sample-based no-retraining sensitivity study for saved TFLite and HOG+SVM artifacts.

## Inputs
Manifest: `D:\BackupFiles\School2023-Present\05_Spring2026\CS659\CS659_Project\DeepLearning-tensorFlowLite\result\manifests\plantnet_300K_manifest_test.csv`
Output directory: `D:\BackupFiles\School2023-Present\05_Spring2026\CS659\CS659_Project\DeepLearning-tensorFlowLite\result\sample_ablation_20260503_023041_UTC`
Created at: `2026-05-03T02:30:41Z`

## Models Evaluated
TFLite available: True
HOG+SVM available: True
Labels available: True

## Ablation Variables
TFLite color correction: none, gray_world, max_rgb
TFLite top-k: 1, 3, 5
HOG+SVM top-k: 1, 3, 5

## Number of Images Attempted
200 manifest row(s) were attempted and 200 image(s) were successfully evaluated.

## Key Results
Model agreement rate: 3.0% (6/200 images).
TFLite predictions were stable across color correction: 22.0% (44/200 images).
TFLite top-1 accuracy: 27.0%.
HOG+SVM top-1 accuracy: 6.0%.
TFLite top-k accuracy: 56.5%.
HOG+SVM top-k accuracy: 36.0%.

## Failures / Skipped Items
No per-image failures were recorded.
No components were skipped.

## Caveats
Model agreement rate: 3.0% (6/200 images).
TFLite predictions were stable across color correction: 22.0% (44/200 images).