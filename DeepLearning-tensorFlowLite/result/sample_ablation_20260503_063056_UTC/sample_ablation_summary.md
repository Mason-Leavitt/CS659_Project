# Sample Ablation Summary

## Study Type
This is a sample-based no-retraining sensitivity study for saved TFLite and HOG+SVM artifacts.

## Inputs
Manifest: `D:\BackupFiles\School2023-Present\05_Spring2026\CS659\CS659_Project\DeepLearning-tensorFlowLite\result\manifests\plantnet_300K_manifest_test.csv`
Output directory: `D:\BackupFiles\School2023-Present\05_Spring2026\CS659\CS659_Project\DeepLearning-tensorFlowLite\result\sample_ablation_20260503_063056_UTC`
Created at: `2026-05-03T06:30:56Z`

## Models Evaluated
TFLite available: True
HOG+SVM available: True
Labels available: True

## Ablation Variables
Sampling mode: balanced
Sampling seed: 42
Max images per class: 1
TFLite color correction: none, gray_world, max_rgb
TFLite top-k: 1, 3, 5
HOG+SVM top-k: 1, 3, 5

## Number of Images Attempted
200 manifest row(s) were attempted and 200 image(s) were successfully evaluated.

## Key Results
Model agreement rate: 0.5% (1/200 images).
TFLite predictions were stable across color correction: 23.5% (47/200 images).
TFLite top-1 accuracy: 10.5%.
HOG+SVM top-1 accuracy: 0.0%.
TFLite top-k accuracy: 26.5%.
HOG+SVM top-k accuracy: 0.5%.

## Observations
Model agreement rate: 0.5%.
TFLite color stability: 23.5%.
TFLite top-k accuracy exceeded HOG+SVM by 26.0 percentage points on this sample.
Sample-size caveat: this run evaluated 200 image(s) using balanced sampling, so small samples may still underrepresent the full test split.

## Failures / Skipped Items
No per-image failures were recorded.
No components were skipped.

## Caveats
Model agreement rate: 0.5% (1/200 images).
TFLite predictions were stable across color correction: 23.5% (47/200 images).