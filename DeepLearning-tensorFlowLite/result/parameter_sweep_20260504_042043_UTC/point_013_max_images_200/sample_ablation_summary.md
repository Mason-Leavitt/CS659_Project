# Sample Ablation Summary

## Study Type
This is a sample-based no-retraining sensitivity study for saved TFLite and HOG+SVM artifacts.

## Inputs
Manifest: `D:\BackupFiles\School2023-Present\05_Spring2026\CS659\CS659_Project\DeepLearning-tensorFlowLite\result\parameter_sweep_20260504_042043_UTC\point_013_max_images_200\point_013_max_images_200_manifest.csv`
Output directory: `D:\BackupFiles\School2023-Present\05_Spring2026\CS659\CS659_Project\DeepLearning-tensorFlowLite\result\parameter_sweep_20260504_042043_UTC\point_013_max_images_200`
Created at: `2026-05-04T04:23:30Z`

## Models Evaluated
TFLite available: True
HOG+SVM available: True
Labels available: True

## Ablation Variables
Sampling mode: balanced
Sampling seed: 50
Max images per class: 5
TFLite color correction: none, gray_world, max_rgb
TFLite top-k: 1, 5
HOG+SVM top-k: 1, 5

## Number of Images Attempted
125 manifest row(s) were attempted and 125 image(s) were successfully evaluated.

## Key Results
Model agreement rate: 0.8% (1/125 images).
TFLite predictions were stable across color correction: 27.2% (34/125 images).
TFLite top-1 accuracy: 26.4%.
HOG+SVM top-1 accuracy: 0.0%.
TFLite top-k accuracy: 51.2%.
HOG+SVM top-k accuracy: 0.0%.

## Observations
Model agreement rate: 0.8%.
TFLite color stability: 27.2%.
TFLite top-k accuracy exceeded HOG+SVM by 51.2 percentage points on this sample.
Sample-size caveat: this run evaluated 125 image(s) using balanced sampling, so small samples may still underrepresent the full test split.

## Failures / Skipped Items
No per-image failures were recorded.
No components were skipped.

## Caveats
Model agreement rate: 0.8% (1/125 images).
TFLite predictions were stable across color correction: 27.2% (34/125 images).