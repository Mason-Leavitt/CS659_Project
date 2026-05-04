# Sample Ablation Summary

## Study Type
This is a sample-based no-retraining sensitivity study for saved TFLite and HOG+SVM artifacts.

## Inputs
Manifest: `D:\BackupFiles\School2023-Present\05_Spring2026\CS659\CS659_Project\DeepLearning-tensorFlowLite\result\parameter_sweep_20260504_033108_UTC\point_014_seed_42\point_014_seed_42_manifest.csv`
Output directory: `D:\BackupFiles\School2023-Present\05_Spring2026\CS659\CS659_Project\DeepLearning-tensorFlowLite\result\parameter_sweep_20260504_033108_UTC\point_014_seed_42`
Created at: `2026-05-04T03:33:19Z`

## Models Evaluated
TFLite available: True
HOG+SVM available: True
Labels available: True

## Ablation Variables
Sampling mode: balanced
Sampling seed: 42
Max images per class: 1
TFLite color correction: none, gray_world, max_rgb
TFLite top-k: 1, 5
HOG+SVM top-k: 1, 5

## Number of Images Attempted
25 manifest row(s) were attempted and 25 image(s) were successfully evaluated.

## Key Results
Model agreement rate: 0.0% (0/25 images).
TFLite predictions were stable across color correction: 36.0% (9/25 images).
TFLite top-1 accuracy: 20.0%.
HOG+SVM top-1 accuracy: 0.0%.
TFLite top-k accuracy: 52.0%.
HOG+SVM top-k accuracy: 0.0%.

## Observations
Model agreement rate: 0.0%.
TFLite color stability: 36.0%.
TFLite top-k accuracy exceeded HOG+SVM by 52.0 percentage points on this sample.
Sample-size caveat: this run evaluated 25 image(s) using balanced sampling, so small samples may still underrepresent the full test split.

## Failures / Skipped Items
No per-image failures were recorded.
No components were skipped.

## Caveats
Model agreement rate: 0.0% (0/25 images).
TFLite predictions were stable across color correction: 36.0% (9/25 images).