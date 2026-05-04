# Parameter Sweep Summary

This is an inference-only one-factor-at-a-time parameter sweep. No model retraining was performed.

## Sweep Metadata
Sweep ID: `parameter_sweep_20260504_033108_UTC`
Output directory: `D:\BackupFiles\School2023-Present\05_Spring2026\CS659\CS659_Project\DeepLearning-tensorFlowLite\result\parameter_sweep_20260504_033108_UTC`
Dataset path: `dataset_sample`
Split: `test`

## Baseline Values
- `top_k`: `5`
- `sampling_mode`: `balanced`
- `max_images`: `200`
- `max_images_per_class`: `1`
- `seed`: `42`

## Parameters Swept
top_k, sampling_mode, max_images, max_images_per_class, seed

## Selected Metrics
tflite_top1_accuracy, hog_top1_accuracy

## Execution Status
Planned runs: 15
Completed runs: 15
Failed runs: 0

## Generated Charts
- `sweep_top_k_metrics.png` | varied parameter: `top_k` | metrics: TFLite top-1 accuracy, HOG+SVM top-1 accuracy | baseline held fixed: sampling_mode=balanced, max_images=200, max_images_per_class=1, seed=42
- `sweep_sampling_mode_metrics.png` | varied parameter: `sampling_mode` | metrics: TFLite top-1 accuracy, HOG+SVM top-1 accuracy | baseline held fixed: top_k=5, max_images=200, max_images_per_class=1, seed=42
- `sweep_max_images_metrics.png` | varied parameter: `max_images` | metrics: TFLite top-1 accuracy, HOG+SVM top-1 accuracy | baseline held fixed: top_k=5, sampling_mode=balanced, max_images_per_class=1, seed=42
- `sweep_max_images_per_class_metrics.png` | varied parameter: `max_images_per_class` | metrics: TFLite top-1 accuracy, HOG+SVM top-1 accuracy | baseline held fixed: top_k=5, sampling_mode=balanced, max_images=200, seed=42
- `sweep_seed_metrics.png` | varied parameter: `seed` | metrics: TFLite top-1 accuracy, HOG+SVM top-1 accuracy | baseline held fixed: top_k=5, sampling_mode=balanced, max_images=200, max_images_per_class=1

## Notes
- Full per-point artifacts are stored in per-point subdirectories under the sweep output directory.
- Interactive chart viewers are planned for a later step.