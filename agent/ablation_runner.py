#!/usr/bin/env python3
"""
No-retraining sample ablation runner for saved plant-classification artifacts.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt

from agent.ablation_history import append_ablation_history
from agent.dataset_manifest import load_manifest_metadata
from agent.model_imports import ensure_model_project_on_path

TFLITE_COLOR_MODES = ("none", "gray_world", "max_rgb")
TOP_K_VALUES = (1, 3, 5)
ProgressCallback = Callable[[dict[str, Any]], None]


def _emit_progress(
    callback: ProgressCallback | None,
    *,
    stage: str,
    current: int,
    total: int,
    message: str,
) -> None:
    if callback is None:
        return
    callback(
        {
            "stage": stage,
            "current": int(current),
            "total": int(total),
            "message": message,
        }
    )


def _load_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return [dict(row) for row in reader]


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _get_predicted_label(entry: dict[str, Any]) -> str | None:
    value = entry.get("label")
    if value is None:
        return None
    return str(value)


def _load_tflite_runner() -> tuple[Any | None, Path | None, str | None]:
    try:
        ensure_model_project_on_path()
        import app_config
        import infer_plant_tflite

        if not app_config.DEFAULT_TFLITE_MODEL_PATH.is_file():
            return None, None, "TFLite ablations were skipped because the TFLite model artifact is missing."
        if not app_config.DEFAULT_EXPORT_LABELS_PATH.is_file():
            return None, None, "TFLite accuracy metrics were skipped because plant_labels_export.txt is missing."
        return infer_plant_tflite, app_config.DEFAULT_EXPORT_LABELS_PATH, None
    except ModuleNotFoundError as exc:
        if getattr(exc, "name", "") == "tensorflow":
            return None, None, "TFLite ablations were skipped because TensorFlow is unavailable."
        return None, None, str(exc)
    except Exception as exc:
        return None, None, str(exc)


def _load_hog_runner() -> tuple[Any | None, str | None]:
    try:
        ensure_model_project_on_path()
        import app_config
        import infer_hog_svm

        status = app_config.get_artifact_status()
        model_info = status.get("hog_svm_model", {})
        labels_info = status.get("hog_svm_labels", {})
        if not model_info.get("exists"):
            return None, "HOG+SVM ablations were skipped because the saved HOG model artifact is missing."
        if not labels_info.get("exists"):
            return None, "HOG+SVM ablations were skipped because the saved HOG labels artifact is missing."
        return infer_hog_svm, None
    except Exception as exc:
        return None, str(exc)


def _make_variant_key(model_name: str, top_k: int, color_correct: str | None = None) -> str:
    if color_correct is None:
        return f"{model_name}:top_k={top_k}"
    return f"{model_name}:color_correct={color_correct}:top_k={top_k}"


def _init_metric_bucket() -> dict[str, Any]:
    return {
        "attempted": 0,
        "successful": 0,
        "failed": 0,
        "top1_correct": 0,
        "topk_correct": 0,
        "score_type": None,
    }


def _save_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _save_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")


def _rate_text(label: str, count: int, total: int) -> str:
    if total <= 0:
        return f"{label} was not computed."
    return f"{label}: {100.0 * count / total:.1f}% ({count}/{total} images)."


def _compute_summary_metrics(
    *,
    agreement_count: int,
    agreement_total: int,
    stability_count: int,
    stability_total: int,
    tflite_top1_correct: int | None,
    tflite_top1_total: int,
    hog_top1_correct: int | None,
    hog_top1_total: int,
    tflite_topk_correct: int | None,
    tflite_topk_total: int,
    hog_topk_correct: int | None,
    hog_topk_total: int,
    labels_available: bool,
    metrics_notes: list[str],
) -> dict[str, Any]:
    def _rate(count: int | None, total: int) -> float | None:
        if count is None or total <= 0:
            return None
        return count / total

    if not labels_available:
        metrics_notes.append("Accuracy was not computed because labels were missing.")
    if agreement_total <= 0:
        metrics_notes.append("Model agreement rate was not computed because both models were not available on the same evaluated images.")
    if stability_total <= 0:
        metrics_notes.append("TFLite color stability was not computed because complete color-correction predictions were unavailable.")

    return {
        "model_agreement_rate": _rate(agreement_count, agreement_total),
        "model_agreement_count": agreement_count if agreement_total > 0 else None,
        "model_agreement_total": agreement_total if agreement_total > 0 else None,
        "tflite_color_stability_rate": _rate(stability_count, stability_total),
        "tflite_color_stability_count": stability_count if stability_total > 0 else None,
        "tflite_color_stability_total": stability_total if stability_total > 0 else None,
        "tflite_top1_accuracy": _rate(tflite_top1_correct, tflite_top1_total),
        "hog_top1_accuracy": _rate(hog_top1_correct, hog_top1_total),
        "tflite_topk_accuracy": _rate(tflite_topk_correct, tflite_topk_total),
        "hog_topk_accuracy": _rate(hog_topk_correct, hog_topk_total),
        "metrics_notes": metrics_notes,
    }


def _save_markdown(path: Path, summary: dict[str, Any]) -> None:
    metrics = summary.get("metrics", {})
    notes = list(metrics.get("metrics_notes", []))
    evaluated = int(summary.get("num_images_evaluated", 0))
    attempted = int(summary.get("num_images_attempted", 0))
    skipped = summary.get("skipped", {})
    tflite_top_k_values = summary.get("tflite_top_k_values", list(TOP_K_VALUES))
    hog_top_k_values = summary.get("hog_top_k_values", list(TOP_K_VALUES))

    lines = [
        "# Sample Ablation Summary",
        "",
        "## Study Type",
        f"This is a sample-based no-retraining sensitivity study for saved TFLite and HOG+SVM artifacts.",
        "",
        "## Inputs",
        f"Manifest: `{summary.get('manifest_path')}`",
        f"Output directory: `{summary.get('output_dir')}`",
        f"Created at: `{summary.get('created_at')}`",
        "",
        "## Models Evaluated",
        f"TFLite available: {summary.get('tflite_available')}",
        f"HOG+SVM available: {summary.get('hog_available')}",
        f"Labels available: {summary.get('labels_available')}",
        "",
        "## Ablation Variables",
        f"Sampling mode: {summary.get('sampling_mode', 'unknown')}",
        f"Sampling seed: {summary.get('seed', 'unknown')}",
        f"Max images per class: {summary.get('max_images_per_class') if summary.get('max_images_per_class') is not None else 'None'}",
        f"TFLite color correction: {', '.join(TFLITE_COLOR_MODES)}",
        f"TFLite top-k: {', '.join(str(v) for v in tflite_top_k_values)}",
        f"HOG+SVM top-k: {', '.join(str(v) for v in hog_top_k_values)}",
        "",
        "## Number of Images Attempted",
        f"{attempted} manifest row(s) were attempted and {evaluated} image(s) were successfully evaluated.",
        "",
        "## Key Results",
    ]

    if evaluated <= 0:
        lines.append("No images were evaluated, so no accuracy, agreement, or stability metrics were computed.")
    else:
        agreement_count = metrics.get("model_agreement_count") or 0
        agreement_total = metrics.get("model_agreement_total") or 0
        stability_count = metrics.get("tflite_color_stability_count") or 0
        stability_total = metrics.get("tflite_color_stability_total") or 0
        lines.append(_rate_text("Model agreement rate", agreement_count, agreement_total))
        lines.append(
            _rate_text(
                "TFLite predictions were stable across color correction",
                stability_count,
                stability_total,
            )
        )

        def _accuracy_line(label: str, value: float | None) -> str:
            if value is None:
                return f"{label} was not computed."
            return f"{label}: {100.0 * value:.1f}%."

        lines.append(_accuracy_line("TFLite top-1 accuracy", metrics.get("tflite_top1_accuracy")))
        lines.append(_accuracy_line("HOG+SVM top-1 accuracy", metrics.get("hog_top1_accuracy")))
        lines.append(_accuracy_line("TFLite top-k accuracy", metrics.get("tflite_topk_accuracy")))
        lines.append(_accuracy_line("HOG+SVM top-k accuracy", metrics.get("hog_topk_accuracy")))

    lines.extend(["", "## Observations"])
    agreement_rate = metrics.get("model_agreement_rate")
    stability_rate = metrics.get("tflite_color_stability_rate")
    if agreement_rate is not None:
        lines.append(f"Model agreement rate: {100.0 * float(agreement_rate):.1f}%.")
    if stability_rate is not None:
        lines.append(f"TFLite color stability: {100.0 * float(stability_rate):.1f}%.")
    if metrics.get("tflite_topk_accuracy") is not None and metrics.get("hog_topk_accuracy") is not None:
        delta = float(metrics["tflite_topk_accuracy"]) - float(metrics["hog_topk_accuracy"])
        if delta > 0:
            lines.append(f"TFLite top-k accuracy exceeded HOG+SVM by {100.0 * delta:.1f} percentage points on this sample.")
        elif delta < 0:
            lines.append(f"HOG+SVM top-k accuracy exceeded TFLite by {100.0 * abs(delta):.1f} percentage points on this sample.")
        else:
            lines.append("TFLite and HOG+SVM achieved the same top-k accuracy on this sample.")
    lines.append(
        f"Sample-size caveat: this run evaluated {evaluated} image(s) using {summary.get('sampling_mode', 'unknown')} sampling, so small samples may still underrepresent the full test split."
    )

    lines.extend(["", "## Failures / Skipped Items"])
    if summary.get("num_failures", 0) > 0:
        lines.append(f"{summary.get('num_failures', 0)} failure record(s) were captured in `failure_report.csv`.")
    else:
        lines.append("No per-image failures were recorded.")
    if skipped:
        for key, reason in skipped.items():
            lines.append(f"{key}: {reason}")
    else:
        lines.append("No components were skipped.")

    lines.extend(["", "## Caveats"])
    if notes:
        lines.extend(notes)
    else:
        lines.append("No additional caveats were recorded.")

    path.write_text("\n".join(lines), encoding="utf-8")


def _save_bar_plot(
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    data: dict[str, float | int],
    subtitle: str | None = None,
    footer: str | None = None,
) -> bool:
    if not data or sum(float(v) for v in data.values()) <= 0.0:
        return False
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=100)
    keys = list(data.keys())
    values = [float(data[key]) for key in keys]
    ax.bar(range(len(keys)), values, color="#2E8B57")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=25, ha="right")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if subtitle:
        fig.text(0.5, 0.94, subtitle, ha="center", va="top", fontsize=9)
    if footer:
        fig.text(0.5, 0.02, footer, ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 0.9))
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


def run_sample_ablation(
    manifest_path: str,
    max_images: int | None = None,
    progress_callback: ProgressCallback | None = None,
    top_k_values: tuple[int, ...] | list[int] | None = None,
    summary_top_k: int | None = None,
    output_dir: str | None = None,
    append_history: bool = True,
    write_charts: bool = True,
) -> dict[str, Any]:
    ensure_model_project_on_path()
    import app_config

    manifest = Path(manifest_path)
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    result: dict[str, Any] = {
        "success": False,
        "warning": None,
        "manifest_path": str(manifest),
        "output_dir": None,
        "num_images_requested": 0,
        "num_images_attempted": 0,
        "num_images_evaluated": 0,
        "num_failures": 0,
        "skipped": {},
        "files": {},
        "summary": {},
        "error": None,
    }

    _emit_progress(
        progress_callback,
        stage="loading_manifest",
        current=0,
        total=1,
        message="Loading manifest.",
    )

    try:
        if not manifest.is_file():
            raise FileNotFoundError(f"Manifest CSV not found: {manifest}")

        rows = _load_manifest_rows(manifest)
        if max_images is not None:
            rows = rows[: max(int(max_images), 0)]
        result["num_images_requested"] = len(rows)
        manifest_metadata = load_manifest_metadata(manifest)
        metadata_ok = bool(manifest_metadata.get("success"))

        valid_rows = [
            row for row in rows
            if str(row.get("image_path", "")).strip()
        ]
        class_distribution = dict(sorted(Counter(str(row.get("label", "")).strip() for row in valid_rows if str(row.get("label", "")).strip()).items()))
        num_classes = len(class_distribution)

        requested_top_k_values = tuple(
            sorted(
                {
                    int(value)
                    for value in (
                        top_k_values
                        if top_k_values is not None
                        else TOP_K_VALUES
                    )
                }
            )
        )
        if not requested_top_k_values:
            raise ValueError("top_k_values must include at least one positive integer.")
        if any(value < 1 for value in requested_top_k_values):
            raise ValueError("top_k_values must contain only positive integers.")
        if 1 not in requested_top_k_values:
            requested_top_k_values = tuple(sorted(set(requested_top_k_values) | {1}))

        resolved_summary_top_k = int(summary_top_k) if summary_top_k is not None else max(requested_top_k_values)
        if resolved_summary_top_k not in requested_top_k_values:
            requested_top_k_values = tuple(sorted(set(requested_top_k_values) | {resolved_summary_top_k}))

        if output_dir is not None:
            out_dir = app_config.ensure_dir(Path(output_dir))
        else:
            out_dir = app_config.ensure_dir(
                app_config.DEFAULT_RESULT_DIR / f"sample_ablation_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC')}"
            )
        result["output_dir"] = str(out_dir)

        _emit_progress(
            progress_callback,
            stage="initializing_models",
            current=0,
            total=1,
            message="Checking available models and labels.",
        )

        tflite_module, tflite_labels_path, tflite_error = _load_tflite_runner()
        hog_module, hog_error = _load_hog_runner()

        labels_available = any(str(row.get("label", "")).strip() for row in valid_rows)
        if tflite_error:
            result["skipped"]["tflite"] = tflite_error
        if hog_error:
            result["skipped"]["hog_svm"] = hog_error

        all_rows: list[dict[str, Any]] = []
        failure_rows: list[dict[str, Any]] = []
        tflite_metrics: dict[str, dict[str, Any]] = {}
        hog_metrics: dict[str, dict[str, Any]] = {}
        for color_correct in TFLITE_COLOR_MODES:
            for top_k in requested_top_k_values:
                tflite_metrics[_make_variant_key("tflite", top_k, color_correct)] = _init_metric_bucket()
        for top_k in requested_top_k_values:
            hog_metrics[_make_variant_key("hog_svm", top_k)] = _init_metric_bucket()

        agreement_count = 0
        disagreement_count = 0
        agreement_total = 0
        stability_count = 0
        stability_total = 0
        overlap_scores: list[int] = []
        figure_notes: list[str] = []
        metrics_notes: list[str] = []

        if not valid_rows:
            result["warning"] = "Manifest contains no valid image_path rows."
            metrics_notes.append("No images were evaluated because the manifest did not contain any valid image paths.")

        total_images = len(valid_rows)
        for idx, row in enumerate(valid_rows, start=1):
            image_path = Path(str(row.get("image_path", "")))
            true_label = str(row.get("label", "")).strip()
            result["num_images_attempted"] += 1
            _emit_progress(
                progress_callback,
                stage="evaluating_images",
                current=idx,
                total=total_images,
                message=f"Evaluating image {idx} of {total_images}: {image_path.name}",
            )

            if not image_path.is_file():
                failure_rows.append(
                    {
                        "image_path": str(image_path),
                        "model": "manifest",
                        "variant": "missing_image",
                        "error": "Image not found",
                    }
                )
                continue

            image_evaluated = False
            tflite_cache: dict[tuple[str, int], dict[str, Any]] = {}
            hog_cache: dict[int, dict[str, Any]] = {}

            if tflite_module is not None and tflite_labels_path is not None:
                for color_correct in TFLITE_COLOR_MODES:
                    for top_k in requested_top_k_values:
                        variant_key = _make_variant_key("tflite", top_k, color_correct)
                        bucket = tflite_metrics[variant_key]
                        bucket["attempted"] += 1
                        infer_result = tflite_module.run_tflite_inference(
                            model_path=app_config.DEFAULT_TFLITE_MODEL_PATH,
                            image_path=image_path,
                            labels_path=tflite_labels_path,
                            top_k=top_k,
                            color_correct=color_correct,
                        )
                        tflite_cache[(color_correct, top_k)] = infer_result
                        if not infer_result.get("success"):
                            bucket["failed"] += 1
                            failure_rows.append(
                                {
                                    "image_path": str(image_path),
                                    "model": "tflite",
                                    "variant": variant_key,
                                    "error": infer_result.get("error") or "Inference failed",
                                }
                            )
                            continue

                        image_evaluated = True
                        bucket["successful"] += 1
                        bucket["score_type"] = "probability"
                        predictions = infer_result.get("predictions", [])
                        top1_label = _get_predicted_label(predictions[0]) if predictions else None
                        topk_labels = [
                            str(pred.get("label"))
                            for pred in predictions
                            if pred.get("label") is not None
                        ]
                        top1_correct = bool(labels_available and true_label and top1_label == true_label)
                        topk_correct = bool(labels_available and true_label and true_label in topk_labels)
                        if top1_correct:
                            bucket["top1_correct"] += 1
                        if topk_correct:
                            bucket["topk_correct"] += 1
                        all_rows.append(
                            {
                                "image_path": str(image_path),
                                "label": true_label,
                                "display_label": row.get("display_label", true_label),
                                "model": "tflite",
                                "variant": variant_key,
                                "top_k": top_k,
                                "color_correct": color_correct,
                                "score_type": "probability",
                                "prediction_label": top1_label or "",
                                "prediction_display_label": top1_label or "",
                                "top1_correct": top1_correct if labels_available else "",
                                "topk_correct": topk_correct if labels_available else "",
                                "error": "",
                            }
                        )

                none_pred = tflite_cache.get(("none", 1), {}).get("predictions", [])
                gray_pred = tflite_cache.get(("gray_world", 1), {}).get("predictions", [])
                max_pred = tflite_cache.get(("max_rgb", 1), {}).get("predictions", [])
                none_label = _get_predicted_label(none_pred[0]) if none_pred else None
                gray_label = _get_predicted_label(gray_pred[0]) if gray_pred else None
                max_label = _get_predicted_label(max_pred[0]) if max_pred else None
                if none_label is not None and gray_label is not None and max_label is not None:
                    stability_total += 1
                    if none_label == gray_label == max_label:
                        stability_count += 1

            if hog_module is not None:
                for top_k in requested_top_k_values:
                    variant_key = _make_variant_key("hog_svm", top_k)
                    bucket = hog_metrics[variant_key]
                    bucket["attempted"] += 1
                    hog_result = hog_module.run_hog_svm_inference(
                        image_path=image_path,
                        top_k=top_k,
                    )
                    hog_cache[top_k] = hog_result
                    if not hog_result.get("success"):
                        bucket["failed"] += 1
                        failure_rows.append(
                            {
                                "image_path": str(image_path),
                                "model": "hog_svm",
                                "variant": variant_key,
                                "error": hog_result.get("error") or "Inference failed",
                            }
                        )
                        continue

                    image_evaluated = True
                    bucket["successful"] += 1
                    bucket["score_type"] = hog_result.get("score_type", "label_only")
                    predictions = hog_result.get("predictions", [])
                    top1_label = _get_predicted_label(predictions[0]) if predictions else None
                    topk_labels = [
                        str(pred.get("label"))
                        for pred in predictions
                        if pred.get("label") is not None
                    ]
                    top1_correct = bool(labels_available and true_label and top1_label == true_label)
                    topk_correct = bool(labels_available and true_label and true_label in topk_labels)
                    if top1_correct:
                        bucket["top1_correct"] += 1
                    if topk_correct:
                        bucket["topk_correct"] += 1
                    top1_pred = predictions[0] if predictions else {}
                    all_rows.append(
                        {
                            "image_path": str(image_path),
                            "label": true_label,
                            "display_label": row.get("display_label", true_label),
                            "model": "hog_svm",
                            "variant": variant_key,
                            "top_k": top_k,
                            "color_correct": "",
                            "score_type": hog_result.get("score_type", "label_only"),
                            "prediction_label": top1_label or "",
                            "prediction_display_label": top1_pred.get("display_label", top1_label or ""),
                            "top1_correct": top1_correct if labels_available else "",
                            "topk_correct": topk_correct if labels_available else "",
                            "error": "",
                        }
                    )

                base_tflite = tflite_cache.get(("none", resolved_summary_top_k))
                base_hog = hog_cache.get(resolved_summary_top_k)
                if base_tflite and base_tflite.get("success") and base_hog and base_hog.get("success"):
                    tflite_preds = base_tflite.get("predictions", [])
                    hog_preds = base_hog.get("predictions", [])
                    if tflite_preds and hog_preds:
                        agreement_total += 1
                        tflite_top1 = str(tflite_preds[0].get("label"))
                        hog_top1 = str(hog_preds[0].get("label"))
                        if tflite_top1 == hog_top1:
                            agreement_count += 1
                        else:
                            disagreement_count += 1
                        overlap_scores.append(
                            len(
                                {
                                    str(pred.get("label"))
                                    for pred in tflite_preds
                                    if pred.get("label") is not None
                                }
                                & {
                                    str(pred.get("label"))
                                    for pred in hog_preds
                                    if pred.get("label") is not None
                                }
                            )
                        )

            if image_evaluated:
                result["num_images_evaluated"] += 1

        _emit_progress(
            progress_callback,
            stage="saving_outputs",
            current=0,
            total=1,
            message="Saving ablation outputs.",
        )

        no_images_message = "No images were evaluated, so no accuracy, agreement, or stability metrics were computed."
        if result["num_images_evaluated"] <= 0:
            metrics_notes.append(no_images_message)
            if result["warning"] is None:
                result["warning"] = no_images_message

        tflite_top1_bucket = tflite_metrics[_make_variant_key("tflite", 1, "none")]
        tflite_topk_bucket = tflite_metrics[_make_variant_key("tflite", resolved_summary_top_k, "none")]
        hog_top1_bucket = hog_metrics[_make_variant_key("hog_svm", 1)]
        hog_topk_bucket = hog_metrics[_make_variant_key("hog_svm", resolved_summary_top_k)]

        metrics = _compute_summary_metrics(
            agreement_count=agreement_count,
            agreement_total=agreement_total,
            stability_count=stability_count,
            stability_total=stability_total,
            tflite_top1_correct=tflite_top1_bucket["top1_correct"] if labels_available else None,
            tflite_top1_total=tflite_top1_bucket["successful"],
            hog_top1_correct=hog_top1_bucket["top1_correct"] if labels_available else None,
            hog_top1_total=hog_top1_bucket["successful"],
            tflite_topk_correct=tflite_topk_bucket["topk_correct"] if labels_available else None,
            tflite_topk_total=tflite_topk_bucket["successful"],
            hog_topk_correct=hog_topk_bucket["topk_correct"] if labels_available else None,
            hog_topk_total=hog_topk_bucket["successful"],
            labels_available=labels_available,
            metrics_notes=metrics_notes,
        )

        if "tflite" in result["skipped"]:
            metrics["metrics_notes"].append(result["skipped"]["tflite"])
        if "hog_svm" in result["skipped"]:
            metrics["metrics_notes"].append(result["skipped"]["hog_svm"])

        if agreement_total > 0:
            metrics["metrics_notes"].append(_rate_text("Model agreement rate", agreement_count, agreement_total))
        if stability_total > 0:
            metrics["metrics_notes"].append(
                _rate_text("TFLite predictions were stable across color correction", stability_count, stability_total)
            )

        summary = {
            "study_type": "sample_ablation_sensitivity_analysis",
            "run_id": f"sample_ablation_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f_UTC')}",
            "created_at": created_at,
            "dataset_path": manifest_metadata.get("dataset_path") if metadata_ok else None,
            "split": manifest_metadata.get("split") if metadata_ok else None,
            "manifest_path": str(manifest),
            "output_dir": str(out_dir),
            "max_images": max_images if max_images is not None else result["num_images_requested"],
            "sampling_mode": manifest_metadata.get("sampling_mode", "unknown") if metadata_ok else "unknown",
            "seed": manifest_metadata.get("seed") if metadata_ok else None,
            "max_images_per_class": manifest_metadata.get("max_images_per_class") if metadata_ok else None,
            "tflite_color_correct_values": list(TFLITE_COLOR_MODES),
            "tflite_top_k_values": list(requested_top_k_values),
            "hog_top_k_values": list(requested_top_k_values),
            "summary_top_k": resolved_summary_top_k,
            "num_images_requested": result["num_images_requested"],
            "num_images_attempted": result["num_images_attempted"],
            "num_images_evaluated": result["num_images_evaluated"],
            "num_failures": len(failure_rows),
            "num_classes": num_classes,
            "class_distribution": class_distribution,
            "tflite_available": tflite_module is not None,
            "hog_available": hog_module is not None,
            "labels_available": labels_available,
            "metrics": metrics,
            "skipped": result["skipped"],
            "tflite_metrics": tflite_metrics,
            "hog_metrics": hog_metrics,
            "comparison": {
                "agreement_count": agreement_count,
                "disagreement_count": disagreement_count,
                "agreement_total": agreement_total,
                "mean_topk_overlap": (sum(overlap_scores) / len(overlap_scores)) if overlap_scores else None,
            },
            "caveats": figure_notes,
            "generated_files": {},
            "failures_summary": {
                "count": len(failure_rows),
                "by_model": dict(Counter(str(row.get("model", "unknown")) for row in failure_rows)),
            },
        }

        results_csv = out_dir / "sample_ablation_results.csv"
        summary_json = out_dir / "sample_ablation_summary.json"
        summary_md = out_dir / "sample_ablation_summary.md"
        stability_png = out_dir / "tflite_color_correction_stability_summary.png"
        agreement_png = out_dir / "model_top1_agreement_summary.png"
        failure_csv = out_dir / "failure_report.csv"

        if all_rows:
            _save_csv(
                results_csv,
                all_rows,
                [
                    "image_path",
                    "label",
                    "display_label",
                    "model",
                    "variant",
                    "top_k",
                    "color_correct",
                    "score_type",
                    "prediction_label",
                    "prediction_display_label",
                    "top1_correct",
                    "topk_correct",
                    "error",
                ],
            )
        _save_json(summary_json, summary)
        _save_markdown(summary_md, summary)
        if failure_rows:
            _save_csv(failure_csv, failure_rows, ["image_path", "model", "variant", "error"])

        chart_context = (
            f"Run: {summary.get('run_id', 'unknown')} | split={summary.get('split', 'unknown')} | "
            f"sampling={summary.get('sampling_mode', 'unknown')} | max_images={summary.get('max_images', 'n/a')} | "
            f"per_class={summary.get('max_images_per_class', 'n/a')} | evaluated={summary.get('num_images_evaluated', 'n/a')} | "
            f"classes={summary.get('num_classes', 'n/a')}"
        )
        chart_footer = "Inference-only ablation summary; no model retraining was performed."

        stability_created = False
        agreement_created = False
        if write_charts:
            stability_created = _save_bar_plot(
                stability_png,
                f"TFLite Color-Correction Stability (n={stability_total})",
                "Stability outcome",
                "Evaluated image count",
                {
                    "stable": stability_count,
                    "changed": max(stability_total - stability_count, 0),
                },
                subtitle=chart_context,
                footer=chart_footer,
            )
            if not stability_created:
                figure_notes.append("Skipped TFLite stability figure because there were not enough successful color-correction comparisons.")

            agreement_created = _save_bar_plot(
                agreement_png,
                f"Model Top-1 Agreement Summary (n={agreement_total})",
                "Agreement outcome",
                "Evaluated image count",
                {
                    "agree": agreement_count,
                    "disagree": disagreement_count,
                },
                subtitle=chart_context,
                footer=chart_footer,
            )
            if not agreement_created:
                figure_notes.append("Skipped model agreement figure because both models were not available on the same evaluated images.")
        else:
            figure_notes.append("Skipped ablation charts because chart writing was disabled for this run.")

        summary["caveats"] = figure_notes
        _save_json(summary_json, summary)
        _save_markdown(summary_md, summary)

        result["files"] = {
            "csv": str(results_csv) if results_csv.is_file() else None,
            "json": str(summary_json),
            "markdown": str(summary_md),
            "stability_plot": str(stability_png) if stability_png.is_file() else None,
            "agreement_plot": str(agreement_png) if agreement_png.is_file() else None,
            "failure_csv": str(failure_csv) if failure_csv.is_file() else None,
        }
        summary["generated_files"] = dict(result["files"])
        result["summary"] = summary
        result["num_failures"] = len(failure_rows)
        result["success"] = True
        if append_history:
            append_ablation_history(summary)

        _emit_progress(
            progress_callback,
            stage="done",
            current=max(result["num_images_attempted"], 1),
            total=max(result["num_images_attempted"], 1),
            message="Sample ablation completed.",
        )
        return result
    except Exception as exc:
        result["error"] = str(exc)
        return result
