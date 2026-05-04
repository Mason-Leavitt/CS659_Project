# Sample Ablation Summary

## Study Type
This is a sample-based no-retraining sensitivity study for saved TFLite and HOG+SVM artifacts.

## Inputs
Manifest: `D:\BackupFiles\School2023-Present\05_Spring2026\CS659\CS659_Project\DeepLearning-tensorFlowLite\result\manifests\plantnet_300K_manifest_test.csv`
Output directory: `D:\BackupFiles\School2023-Present\05_Spring2026\CS659\CS659_Project\DeepLearning-tensorFlowLite\result\sample_ablation_20260503_063202_UTC`
Created at: `2026-05-03T06:32:02Z`

## Models Evaluated
TFLite available: True
HOG+SVM available: True
Labels available: True

## Ablation Variables
Sampling mode: balanced
Sampling seed: 7
Max images per class: 1
TFLite color correction: none, gray_world, max_rgb
TFLite top-k: 1, 3, 5
HOG+SVM top-k: 1, 3, 5

## Number of Images Attempted
200 manifest row(s) were attempted and 200 image(s) were successfully evaluated.

## Key Results
Model agreement rate: 1.5% (3/200 images).
TFLite predictions were stable across color correction: 21.5% (43/200 images).
TFLite top-1 accuracy: 15.0%.
HOG+SVM top-1 accuracy: 0.0%.
TFLite top-k accuracy: 28.5%.
HOG+SVM top-k accuracy: 2.0%.

## Observations
Model agreement rate: 1.5%.
TFLite color stability: 21.5%.
TFLite top-k accuracy exceeded HOG+SVM by 26.5 percentage points on this sample.
Sample-size caveat: this run evaluated 200 image(s) using balanced sampling, so small samples may still underrepresent the full test split.

## Failures / Skipped Items
No per-image failures were recorded.
No components were skipped.

## Caveats
Model agreement rate: 1.5% (3/200 images).
TFLite predictions were stable across color correction: 21.5% (43/200 images).