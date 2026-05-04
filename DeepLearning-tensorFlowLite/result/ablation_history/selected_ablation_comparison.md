# Ablation Comparison Summary

## Comparison Validity
These runs are not directly comparable because core comparison fields differ: sampling_mode, max_images_per_class. Matching core fields: dataset_path, split, max_images.

Differing fields: sampling_mode, max_images_per_class, num_classes, seed

- These runs are not directly comparable because core comparison fields differ: sampling_mode, max_images_per_class.
- Observed evaluation coverage differs across runs: num_classes.
- Seed differences make this comparison more exploratory even though the core configuration may match.

## Compared Runs
- sample_ablation_20260503_063255_090739_UTC: split=test sampling=balanced max_images=200 TFLite top-k=0.285 HOG top-k=0.02.
- sample_ablation_20260503_063347_619633_UTC: split=test sampling=balanced max_images=200 TFLite top-k=0.265 HOG top-k=0.005.
- sample_ablation_20260503_063440_109357_UTC: split=test sampling=random max_images=200 TFLite top-k=0.735 HOG top-k=0.045.