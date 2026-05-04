# Ablation History Summary Report

## Scope / Data Source
- Generated at: `2026-05-03T18:06:54Z`
- Recorded history records: `6`
- Comparable groups: `5`
- Corrupted history lines skipped: `0`
- Source of truth: `ablation_history.jsonl` remains the canonical append-only record.

## Executive Summary
- Recommendation-ready history: `yes`
- Runs with metric payloads: `6` of `6`
- Best observed tflite top-1 accuracy: `46.5%` in `sample_ablation_20260503_063440_109357_UTC`
- Best observed tflite top-k accuracy: `73.5%` in `sample_ablation_20260503_063440_109357_UTC`
- Best observed hog top-1 accuracy: `2.0%` in `sample_ablation_20260503_063440_109357_UTC`
- Best observed hog top-k accuracy: `4.5%` in `sample_ablation_20260503_063440_109357_UTC`
- Best observed model agreement rate: `2.0%` in `sample_ablation_20260503_063440_109357_UTC`
- Best observed tflite color stability: `34.5%` in `sample_ablation_20260503_063440_109357_UTC`
- Comparability note: results are only directly comparable within matching dataset, split, sampling mode, max_images, and max_images_per_class groups.

## Best Observed Runs
| Metric | Value | Run ID | Dataset | Split | Sampling | Max Images | Per-Class Cap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TFLite top-1 accuracy | 46.5% | sample_ablation_20260503_063440_109357_UTC | plantnet_300K | test | random | 200 | n/a |
| TFLite top-k accuracy | 73.5% | sample_ablation_20260503_063440_109357_UTC | plantnet_300K | test | random | 200 | n/a |
| HOG top-1 accuracy | 2.0% | sample_ablation_20260503_063440_109357_UTC | plantnet_300K | test | random | 200 | n/a |
| HOG top-k accuracy | 4.5% | sample_ablation_20260503_063440_109357_UTC | plantnet_300K | test | random | 200 | n/a |
| Model agreement rate | 2.0% | sample_ablation_20260503_063440_109357_UTC | plantnet_300K | test | random | 200 | n/a |
| TFLite color stability | 34.5% | sample_ablation_20260503_063440_109357_UTC | plantnet_300K | test | random | 200 | n/a |

## Comparable Group Summary
The table below shows up to the top `5` comparable groups ranked by repeated evidence and stability. Use the CSV/JSON exports for the full history.
| Dataset | Split | Sampling | Max Images | Per-Class Cap | Runs | Evaluated Images Range | Class Count Range | Best TFLite Top-1 | Best HOG Top-1 | Best Agreement | Best Stability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| plantnet_300K | test | balanced | 200 | 1 | 2 | 200 | 200 | 15.0% | 0.0% | 1.5% | 23.5% |
| plantnet_300K | test | balanced | 10 | 1 | 1 | 10 | 10 | 10.0% | 0.0% | 0.0% | 20.0% |
| plantnet_300K | test | balanced | 200 | 2 | 1 | 200 | 200 | 10.5% | 0.0% | 0.5% | 23.5% |
| plantnet_300K | test | balanced | 50 | 1 | 1 | 50 | 50 | 10.0% | 0.0% | 0.0% | 28.0% |
| plantnet_300K | test | random | 200 | n/a | 1 | 200 | 114 | 46.5% | 2.0% | 2.0% | 34.5% |

## Recommendation Readiness / Evidence
- Sufficient history for evidence-based recommendations: `yes`
- At least one directly comparable group has repeated runs, but the evidence remains limited because support is only two runs or key grouping fields are missing.
- Recommendation confidence is based on repeated directly comparable groups rather than total global run count.
- Strongest repeated groups in recorded history:
  - plantnet_300K | split=test | sampling=balanced | max_images=200 | per_class=1 | runs=2

## Caveats and Interpretation Notes
- Ablations are inference-only; no model retraining is performed.
- Cross-dataset or cross-sampling comparisons are exploratory rather than directly comparable.
- Accuracy metrics are only meaningful when manifest labels are available for the evaluated images.
- Sample-based runs are not full-dataset evaluations unless a full test split was explicitly used.
- Balanced and random runs are not directly comparable here because their observed class coverage differs substantially.

## Generated Artifacts
- `ablation_history.jsonl`
- `ablation_history_table.csv`
- `ablation_history_summary.json`
- `ablation_history_summary.md`
- `ablation_history_metrics.png`
