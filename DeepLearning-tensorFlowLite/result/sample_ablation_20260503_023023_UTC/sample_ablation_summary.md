# Sample Ablation Summary

## Study Type
This is a sample-based no-retraining sensitivity study for saved TFLite and HOG+SVM artifacts.

## Inputs
Manifest: `D:\BackupFiles\School2023-Present\05_Spring2026\CS659\CS659_Project\DeepLearning-tensorFlowLite\result\manifests\plantnet_300K_manifest_test.csv`
Output directory: `D:\BackupFiles\School2023-Present\05_Spring2026\CS659\CS659_Project\DeepLearning-tensorFlowLite\result\sample_ablation_20260503_023023_UTC`
Created at: `2026-05-03T02:30:23Z`

## Models Evaluated
TFLite available: True
HOG+SVM available: True
Labels available: True

## Ablation Variables
TFLite color correction: none, gray_world, max_rgb
TFLite top-k: 1, 3, 5
HOG+SVM top-k: 1, 3, 5

## Number of Images Attempted
50 manifest row(s) were attempted and 50 image(s) were successfully evaluated.

## Key Results
Model agreement rate: 2.0% (1/50 images).
TFLite predictions were stable across color correction: 16.0% (8/50 images).
TFLite top-1 accuracy: 6.0%.
HOG+SVM top-1 accuracy: 0.0%.
TFLite top-k accuracy: 34.0%.
HOG+SVM top-k accuracy: 6.0%.

## Failures / Skipped Items
No per-image failures were recorded.
No components were skipped.

## Caveats
Model agreement rate: 2.0% (1/50 images).
TFLite predictions were stable across color correction: 16.0% (8/50 images).