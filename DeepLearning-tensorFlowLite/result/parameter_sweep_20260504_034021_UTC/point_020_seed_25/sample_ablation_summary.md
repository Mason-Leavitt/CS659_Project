# Sample Ablation Summary

## Study Type
This is a sample-based no-retraining sensitivity study for saved TFLite and HOG+SVM artifacts.

## Inputs
Manifest: `D:\BackupFiles\School2023-Present\05_Spring2026\CS659\CS659_Project\DeepLearning-tensorFlowLite\result\parameter_sweep_20260504_034021_UTC\point_020_seed_25\point_020_seed_25_manifest.csv`
Output directory: `D:\BackupFiles\School2023-Present\05_Spring2026\CS659\CS659_Project\DeepLearning-tensorFlowLite\result\parameter_sweep_20260504_034021_UTC\point_020_seed_25`
Created at: `2026-05-04T03:45:01Z`

## Models Evaluated
TFLite available: True
HOG+SVM available: True
Labels available: True

## Ablation Variables
Sampling mode: balanced
Sampling seed: 25
Max images per class: 1
TFLite color correction: none, gray_world, max_rgb
TFLite top-k: 1, 5
HOG+SVM top-k: 1, 5

## Number of Images Attempted
25 manifest row(s) were attempted and 25 image(s) were successfully evaluated.

## Key Results
Model agreement rate: 4.0% (1/25 images).
TFLite predictions were stable across color correction: 32.0% (8/25 images).
TFLite top-1 accuracy: 20.0%.
HOG+SVM top-1 accuracy: 0.0%.
TFLite top-k accuracy: 64.0%.
HOG+SVM top-k accuracy: 0.0%.

## Observations
Model agreement rate: 4.0%.
TFLite color stability: 32.0%.
TFLite top-k accuracy exceeded HOG+SVM by 64.0 percentage points on this sample.
Sample-size caveat: this run evaluated 25 image(s) using balanced sampling, so small samples may still underrepresent the full test split.

## Failures / Skipped Items
No per-image failures were recorded.
No components were skipped.

## Caveats
Model agreement rate: 4.0% (1/25 images).
TFLite predictions were stable across color correction: 32.0% (8/25 images).