# Sample Ablation Summary

## Study Type
This is a sample-based no-retraining sensitivity study for saved TFLite and HOG+SVM artifacts.

## Inputs
Manifest: `D:\BackupFiles\School2023-Present\05_Spring2026\CS659\CS659_Project\DeepLearning-tensorFlowLite\result\parameter_sweep_20260504_034021_UTC\point_017_max_images_per_class_7\point_017_max_images_per_class_7_manifest.csv`
Output directory: `D:\BackupFiles\School2023-Present\05_Spring2026\CS659\CS659_Project\DeepLearning-tensorFlowLite\result\parameter_sweep_20260504_034021_UTC\point_017_max_images_per_class_7`
Created at: `2026-05-04T03:42:55Z`

## Models Evaluated
TFLite available: True
HOG+SVM available: True
Labels available: True

## Ablation Variables
Sampling mode: balanced
Sampling seed: 42
Max images per class: 7
TFLite color correction: none, gray_world, max_rgb
TFLite top-k: 1, 5
HOG+SVM top-k: 1, 5

## Number of Images Attempted
175 manifest row(s) were attempted and 175 image(s) were successfully evaluated.

## Key Results
Model agreement rate: 0.6% (1/175 images).
TFLite predictions were stable across color correction: 30.3% (53/175 images).
TFLite top-1 accuracy: 26.3%.
HOG+SVM top-1 accuracy: 0.0%.
TFLite top-k accuracy: 52.6%.
HOG+SVM top-k accuracy: 0.0%.

## Observations
Model agreement rate: 0.6%.
TFLite color stability: 30.3%.
TFLite top-k accuracy exceeded HOG+SVM by 52.6 percentage points on this sample.
Sample-size caveat: this run evaluated 175 image(s) using balanced sampling, so small samples may still underrepresent the full test split.

## Failures / Skipped Items
No per-image failures were recorded.
No components were skipped.

## Caveats
Model agreement rate: 0.6% (1/175 images).
TFLite predictions were stable across color correction: 30.3% (53/175 images).