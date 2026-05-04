#!/usr/bin/env python3
"""
Plain tool-wrapper functions for the future agent layer.

This module is intentionally usable without LangChain being installed. When
LangChain is available, StructuredTool wrappers are exposed via AGENT_TOOLS.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.model_imports import ensure_model_project_on_path

LANGCHAIN_AVAILABLE = False
AGENT_TOOLS: list[Any] = []


def _friendly_error(exc: Exception) -> str:
    text = str(exc)
    if isinstance(exc, ModuleNotFoundError) and getattr(exc, "name", "") == "tensorflow":
        return "TensorFlow is not installed, so TFLite classification is unavailable in this environment."
    return text


def _load_app_config() -> Any:
    ensure_model_project_on_path()
    import app_config

    return app_config


def _load_tflite_module() -> Any:
    ensure_model_project_on_path()
    import infer_plant_tflite

    return infer_plant_tflite


def _load_hog_module() -> Any:
    ensure_model_project_on_path()
    import infer_hog_svm

    return infer_hog_svm


def _load_ablation_module() -> Any:
    import agent.ablation_utils as ablation_utils

    return ablation_utils


def _load_dataset_manifest_module() -> Any:
    import agent.dataset_manifest as dataset_manifest

    return dataset_manifest


def _load_ablation_runner_module() -> Any:
    import agent.ablation_runner as ablation_runner

    return ablation_runner


def _load_ablation_history_module() -> Any:
    import agent.ablation_history as ablation_history

    return ablation_history


def _load_ablation_planner_module() -> Any:
    import agent.ablation_planner as ablation_planner

    return ablation_planner


def _load_parameter_sweep_planner_module() -> Any:
    import agent.parameter_sweep_planner as parameter_sweep_planner

    return parameter_sweep_planner


def _load_parameter_sweep_runner_module() -> Any:
    import agent.parameter_sweep_runner as parameter_sweep_runner

    return parameter_sweep_runner


def classify_with_tflite(
    image_path: str,
    top_k: int = 5,
    color_correct: str = "none",
) -> dict[str, Any]:
    """Classify one image with the default TFLite model and label file."""
    try:
        app_config = _load_app_config()
        infer_plant_tflite = _load_tflite_module()
        return infer_plant_tflite.run_tflite_inference(
            model_path=app_config.DEFAULT_TFLITE_MODEL_PATH,
            image_path=image_path,
            labels_path=app_config.get_default_labels_path(),
            top_k=top_k,
            color_correct=color_correct,
        )
    except Exception as exc:
        return {
            "success": False,
            "image_path": image_path,
            "model_path": None,
            "labels_path": None,
            "top_k": int(top_k),
            "color_correct": color_correct,
            "input_shape": [],
            "predictions": [],
            "error": _friendly_error(exc),
        }


def classify_with_hog_svm(image_path: str, top_k: int = 5) -> dict[str, Any]:
    """Classify one image with the default HOG+SVM artifacts when available."""
    try:
        app_config = _load_app_config()
        artifact_status = app_config.get_artifact_status()
        hog_model = artifact_status.get("hog_svm_model", {})
        hog_labels = artifact_status.get("hog_svm_labels", {})
        if not hog_model.get("exists"):
            return {
                "success": False,
                "image_path": image_path,
                "model_path": str(hog_model.get("path")) if hog_model.get("path") is not None else None,
                "labels_path": str(hog_labels.get("path")) if hog_labels.get("path") is not None else None,
                "prediction": {
                    "label": None,
                    "display_label": None,
                    "confidence": None,
                    "score": None,
                },
                "predictions": [],
                "score_type": "label_only",
                "preprocessing": {
                    "img_size": 224,
                    "grayscale": True,
                    "hog": {
                        "orientations": 9,
                        "pixels_per_cell": 16,
                        "cells_per_block": 2,
                        "block_norm": "L2-Hys",
                    },
                },
                "warning": None,
                "error": "HOG+SVM model artifact not available. Expected hog_svm_model.joblib in DeepLearning-tensorFlowLite/ or under result/.",
            }
        if not hog_labels.get("exists"):
            return {
                "success": False,
                "image_path": image_path,
                "model_path": str(hog_model.get("path")) if hog_model.get("path") is not None else None,
                "labels_path": str(hog_labels.get("path")) if hog_labels.get("path") is not None else None,
                "prediction": {
                    "label": None,
                    "display_label": None,
                    "confidence": None,
                    "score": None,
                },
                "predictions": [],
                "score_type": "label_only",
                "preprocessing": {
                    "img_size": 224,
                    "grayscale": True,
                    "hog": {
                        "orientations": 9,
                        "pixels_per_cell": 16,
                        "cells_per_block": 2,
                        "block_norm": "L2-Hys",
                    },
                },
                "warning": None,
                "error": "HOG+SVM labels artifact not available. Expected hog_svm_labels.txt in DeepLearning-tensorFlowLite/ or under result/.",
            }
        infer_hog_svm = _load_hog_module()
        return infer_hog_svm.run_hog_svm_inference(image_path=image_path, top_k=top_k)
    except Exception as exc:
        return {
            "success": False,
            "image_path": image_path,
            "model_path": None,
            "labels_path": None,
            "prediction": {
                "label": None,
                "display_label": None,
                "confidence": None,
                "score": None,
            },
            "predictions": [],
            "score_type": "label_only",
            "preprocessing": {
                "img_size": 224,
                "grayscale": True,
                "hog": {
                    "orientations": 9,
                    "pixels_per_cell": 16,
                    "cells_per_block": 2,
                    "block_norm": "L2-Hys",
                },
            },
            "warning": None,
            "error": _friendly_error(exc),
        }


def compare_classifiers(
    image_path: str,
    top_k: int = 5,
    color_correct: str = "none",
) -> dict[str, Any]:
    """Compare TFLite top-1 output against HOG+SVM output when both are available."""
    tflite_result = classify_with_tflite(
        image_path=image_path,
        top_k=top_k,
        color_correct=color_correct,
    )
    hog_svm_result = classify_with_hog_svm(image_path=image_path, top_k=top_k)

    comparison: dict[str, Any] = {
        "success": False,
        "image_path": image_path,
        "tflite_result": tflite_result,
        "hog_svm_result": hog_svm_result,
        "agreement": None,
        "explanation": "",
        "error": None,
    }

    if not tflite_result.get("success"):
        comparison["error"] = tflite_result.get("error") or "TFLite classification failed."
        comparison["explanation"] = (
            "Comparison is not available because the TFLite classifier could not produce a result."
        )
        return comparison

    if not hog_svm_result.get("success"):
        comparison["success"] = True
        comparison["explanation"] = (
            "TFLite classification succeeded, but HOG+SVM comparison is not available because "
            f"the baseline artifacts or inference result are unavailable: {hog_svm_result.get('error') or 'not available'}"
        )
        return comparison

    tflite_predictions = tflite_result.get("predictions", [])
    tflite_top1 = tflite_predictions[0]["label"] if tflite_predictions else None
    hog_label = hog_svm_result.get("prediction", {}).get("label")

    agreement = None
    if tflite_top1 is not None and hog_label is not None:
        agreement = tflite_top1 == hog_label

    comparison["success"] = True
    comparison["agreement"] = agreement
    comparison["explanation"] = (
        f"TFLite top-1 prediction is '{tflite_top1}' and HOG+SVM prediction is '{hog_label}'. "
        f"Agreement: {agreement}."
    )
    return comparison


def get_artifact_status() -> dict[str, Any]:
    """Return current filesystem-backed artifact status from app_config."""
    app_config = _load_app_config()
    return app_config.get_artifact_status()


def get_metrics_summary() -> dict[str, Any]:
    """
    Return a real summary of existing metrics artifacts without fabricating data.

    This inspects result/ for HOG+SVM runs with summary.json and CNN/TFLite runs
    with metrics_summary.json, including nested single_split/final_retrain/fold_*
    subdirectories used by the existing training code.
    """
    app_config = _load_app_config()

    result_dir = Path(app_config.DEFAULT_RESULT_DIR)
    known_names = {
        "metrics_summary.json",
        "metrics_classification_report.txt",
        "confusion_matrix_normalized.png",
        "confusion_row_correlation.png",
        "roc_curves.png",
        "summary.json",
        "classification_report.txt",
    }

    if not result_dir.is_dir():
        return {
            "success": False,
            "result_dir": str(result_dir),
            "available": False,
            "runs": [],
            "error": "Metrics not available: result directory does not exist.",
        }

    def _relative_known_files(run_dir: Path) -> list[str]:
        return sorted(
            str(p.relative_to(run_dir))
            for p in run_dir.rglob("*")
            if p.is_file() and p.name in known_names
        )

    def _parse_json_file(path: Path) -> tuple[dict[str, Any], str | None]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return {}, f"Summary JSON is not an object: {path}"
            return data, None
        except Exception as exc:
            return {}, f"Could not parse summary JSON {path}: {exc}"

    runs: list[dict[str, Any]] = []
    parse_errors: list[str] = []
    seen_summary_paths: set[Path] = set()

    summary_candidates = sorted(result_dir.rglob("summary.json")) + sorted(result_dir.rglob("metrics_summary.json"))
    for summary_path in summary_candidates:
        summary_path = summary_path.resolve()
        if summary_path in seen_summary_paths:
            continue
        seen_summary_paths.add(summary_path)

        run_dir = summary_path.parent
        method = "unknown"
        if summary_path.name == "metrics_summary.json":
            method = "cnn_tflite"
        elif summary_path.name == "summary.json":
            method = "hog_svm"

        metrics, run_error = _parse_json_file(summary_path)
        if method == "unknown" and metrics.get("method") == "hog_svm":
            method = "hog_svm"

        run_entry: dict[str, Any] = {
            "run_dir": str(run_dir),
            "method": method,
            "summary_file": str(summary_path),
            "metrics": metrics,
            "files": _relative_known_files(run_dir),
        }
        if run_error is not None:
            run_entry["error"] = run_error
            parse_errors.append(run_error)
        runs.append(run_entry)

    available = bool(runs)
    error: str | None = None
    if not available:
        error = "Metrics not available: no summary.json or metrics_summary.json files were found under result/."
    elif parse_errors:
        error = "Some metrics files were found but could not be parsed."

    return {
        "success": available,
        "result_dir": str(result_dir),
        "available": available,
        "runs": runs,
        "error": error,
    }


def get_ablation_feasibility() -> dict[str, Any]:
    """Return a safe feasibility summary for future ablation work."""
    try:
        ablation_utils = _load_ablation_module()
        return ablation_utils.get_ablation_feasibility()
    except Exception as exc:
        return {
            "success": False,
            "hardware": {},
            "possible_now": [],
            "requires_dataset": [],
            "requires_training": [],
            "skipped": [],
            "explanation": "",
            "error": _friendly_error(exc),
        }


def build_dataset_manifest(
    dataset_path: str,
    split: str = "test",
    max_images: int | None = None,
    sampling_mode: str = "balanced",
    seed: int = 42,
    max_images_per_class: int | None = None,
) -> dict[str, Any]:
    """Build a CSV manifest for a PlantNet-style dataset split."""
    try:
        app_config = _load_app_config()
        dataset_manifest = _load_dataset_manifest_module()
        out_dir = app_config.ensure_dir(app_config.DEFAULT_RESULT_DIR / "manifests")
        suffix = f"manifest_{split}.csv"
        safe_name = Path(str(dataset_path)).name or "dataset"
        out_csv = out_dir / f"{safe_name}_{suffix}"
        return dataset_manifest.build_manifest(
            dataset_path=dataset_path,
            split=split,
            out_csv=str(out_csv),
            max_images=max_images,
            sampling_mode=sampling_mode,
            seed=seed,
            max_images_per_class=max_images_per_class,
        )
    except Exception as exc:
        return {
            "success": False,
            "dataset_path": str(dataset_path),
            "resolved_image_root": None,
            "split": split,
            "num_images": 0,
            "num_classes": 0,
            "manifest_path": None,
            "preview": [],
            "sampling_mode": str(sampling_mode or "balanced"),
            "seed": int(seed),
            "max_images_per_class": int(max_images_per_class) if max_images_per_class is not None else None,
            "class_distribution": {},
            "error": _friendly_error(exc),
        }


def get_ablation_history(limit: int | None = 10) -> dict[str, Any]:
    try:
        ablation_history = _load_ablation_history_module()
        return ablation_history.load_ablation_history(limit=limit)
    except Exception as exc:
        return {
            "success": False,
            "history_path": None,
            "entries": [],
            "count": 0,
            "corrupted_lines": [],
            "error": _friendly_error(exc),
        }


def get_ablation_results(limit: int | None = 10) -> dict[str, Any]:
    result = get_ablation_history(limit=limit)
    if isinstance(result, dict) and "runs" not in result:
        result["runs"] = result.get("entries", [])
    return result


def get_latest_ablation_result() -> dict[str, Any]:
    try:
        ablation_history = _load_ablation_history_module()
        return ablation_history.get_ablation_run(latest=True)
    except Exception as exc:
        return {
            "success": False,
            "run": None,
            "error": _friendly_error(exc),
        }


def get_ablation_result_by_id(run_id: str) -> dict[str, Any]:
    try:
        ablation_history = _load_ablation_history_module()
        return ablation_history.get_ablation_run(run_id=run_id, latest=False)
    except Exception as exc:
        return {
            "success": False,
            "run": None,
            "error": _friendly_error(exc),
        }


def compare_ablation_results(run_ids: list[str] | None = None) -> dict[str, Any]:
    try:
        ablation_history = _load_ablation_history_module()
        return ablation_history.compare_ablation_runs(run_ids=run_ids)
    except Exception as exc:
        return {
            "success": False,
            "runs": [],
            "comparison_rows": [],
            "selected_run_ids": [],
            "explanation": "",
            "error": _friendly_error(exc),
        }


def export_ablation_results() -> dict[str, Any]:
    try:
        ablation_history = _load_ablation_history_module()
        return ablation_history.export_ablation_history_artifacts()
    except Exception as exc:
        return {
            "success": False,
            "artifacts": {},
            "count": 0,
            "corrupted_lines": [],
            "error": _friendly_error(exc),
        }


def get_ablation_recommendations() -> dict[str, Any]:
    try:
        ablation_history = _load_ablation_history_module()
        return ablation_history.get_recommendations_from_history()
    except Exception as exc:
        return {
            "success": False,
            "enough_history": False,
            "recommendations": {},
            "constraints": [],
            "rationale": [],
            "history_summary": {},
            "error": _friendly_error(exc),
        }


def plan_ablation_study(message: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        ablation_planner = _load_ablation_planner_module()
        return ablation_planner.plan_ablation_from_request(message=message, context=context)
    except Exception as exc:
        return {
            "success": False,
            "needs_more_info": False,
            "missing_fields": [],
            "prompt": None,
            "plan": {},
            "warnings": [],
            "parsed": {},
            "error": _friendly_error(exc),
        }


def run_planned_ablation(
    dataset_path: str,
    split: str = "test",
    max_images: int | None = 50,
    sampling_mode: str = "balanced",
    seed: int = 42,
    max_images_per_class: int | None = 1,
) -> dict[str, Any]:
    try:
        manifest_result = build_dataset_manifest(
            dataset_path=dataset_path,
            split=split,
            max_images=max_images,
            sampling_mode=sampling_mode,
            seed=seed,
            max_images_per_class=max_images_per_class,
        )
        if not manifest_result.get("success") or not manifest_result.get("manifest_path"):
            return {
                "success": False,
                "manifest": manifest_result,
                "ablation": None,
                "error": manifest_result.get("error") or "Manifest generation failed.",
            }
        ablation_result = run_sample_ablation(
            manifest_path=str(manifest_result["manifest_path"]),
            max_images=max_images,
        )
        return {
            "success": bool(ablation_result.get("success")),
            "manifest": manifest_result,
            "ablation": ablation_result,
            "summary": ablation_result.get("summary", {}),
            "files": ablation_result.get("files", {}),
            "error": ablation_result.get("error"),
        }
    except Exception as exc:
        return {
            "success": False,
            "manifest": None,
            "ablation": None,
            "summary": {},
            "files": {},
            "error": _friendly_error(exc),
        }


def run_sample_ablation(
    manifest_path: str,
    max_images: int | None = None,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    """Run no-retraining sample ablations from a manifest CSV."""
    try:
        ablation_runner = _load_ablation_runner_module()
        return ablation_runner.run_sample_ablation(
            manifest_path=manifest_path,
            max_images=max_images,
            progress_callback=progress_callback,
        )
    except Exception as exc:
        return {
            "success": False,
            "manifest_path": manifest_path,
            "output_dir": None,
            "num_images_requested": 0,
            "num_images_attempted": 0,
            "num_images_evaluated": 0,
            "num_failures": 0,
            "skipped": {},
            "files": {},
            "summary": {},
            "error": _friendly_error(exc),
        }


def parse_parameter_sweep_request_tool(request: str) -> dict[str, Any]:
    """Parse a natural-language inference-only parameter sweep request."""
    try:
        parameter_sweep_planner = _load_parameter_sweep_planner_module()
        return parameter_sweep_planner.parse_parameter_sweep_request(request)
    except Exception as exc:
        return {
            "success": False,
            "text": str(request or ""),
            "parameter_ranges": {},
            "baseline_values": {},
            "selected_metrics": [],
            "warnings": [],
            "error": _friendly_error(exc),
        }


def get_parameter_sweep_support() -> dict[str, Any]:
    """Describe the supported inference-only OFAT parameter sweep surface."""
    try:
        parameter_sweep_planner = _load_parameter_sweep_planner_module()
        return {
            "success": True,
            "study_type": "parameter_sweep",
            "inference_only": True,
            "supported_parameters": list(parameter_sweep_planner.SUPPORTED_SWEEP_PARAMETERS),
            "supported_metrics": list(parameter_sweep_planner.SUPPORTED_SWEEP_METRICS),
            "default_selected_metrics": list(parameter_sweep_planner.DEFAULT_SELECTED_METRICS),
            "default_baseline_values": dict(parameter_sweep_planner.DEFAULT_BASELINE_VALUES),
            "safety_limits": dict(parameter_sweep_planner.DEFAULT_SWEEP_LIMITS),
            "unsupported_controls": [
                "color_correct",
                "svm_c",
                "svm_kernel",
                "svm_gamma",
                "hog_orientations",
                "hog_pixels_per_cell",
                "hog_cells_per_block",
            ],
            "notes": [
                "This sweep varies inference-time controls only.",
                "Training-time HOG/SVM hyperparameters are not currently swept.",
            ],
        }
    except Exception as exc:
        return {
            "success": False,
            "study_type": "parameter_sweep",
            "inference_only": True,
            "supported_parameters": [],
            "supported_metrics": [],
            "default_selected_metrics": [],
            "default_baseline_values": {},
            "safety_limits": {},
            "unsupported_controls": [],
            "notes": [],
            "error": _friendly_error(exc),
        }


def plan_parameter_sweep(
    request: str | None = None,
    dataset_path: str | None = None,
    split: str | None = None,
    parameter_ranges: dict[str, Any] | None = None,
    baseline_values: dict[str, Any] | None = None,
    selected_metrics: list[str] | None = None,
    max_total_runs: int | None = None,
    max_values_per_parameter: int | None = None,
) -> dict[str, Any]:
    """Build an inference-only one-factor parameter sweep plan from text or structured inputs."""
    try:
        parameter_sweep_planner = _load_parameter_sweep_planner_module()
        parsed = (
            parameter_sweep_planner.parse_parameter_sweep_request(request)
            if str(request or "").strip()
            else {
                "success": True,
                "text": str(request or ""),
                "parameter_ranges": {},
                "baseline_values": {},
                "selected_metrics": [],
                "warnings": [],
            }
        )

        merged_parameter_ranges = dict(parsed.get("parameter_ranges", {}))
        merged_baseline_values = dict(parsed.get("baseline_values", {}))
        if isinstance(parameter_ranges, dict):
            merged_parameter_ranges.update(parameter_ranges)
        if isinstance(baseline_values, dict):
            merged_baseline_values.update(baseline_values)

        effective_metrics = selected_metrics
        if effective_metrics is None:
            parsed_metrics = parsed.get("selected_metrics", [])
            if isinstance(parsed_metrics, list) and parsed_metrics:
                effective_metrics = list(parsed_metrics)

        plan = parameter_sweep_planner.build_parameter_sweep_plan(
            dataset_path=dataset_path,
            split=split,
            parameter_ranges=merged_parameter_ranges,
            baseline_values=merged_baseline_values,
            selected_metrics=effective_metrics,
            max_total_runs=max_total_runs,
            max_values_per_parameter=max_values_per_parameter,
        )
        plan["request"] = str(request or "")
        plan["parsed_request"] = parsed
        return plan
    except Exception as exc:
        return {
            "success": False,
            "study_type": "parameter_sweep",
            "errors": [_friendly_error(exc)],
            "warnings": [],
            "require_confirmation": False,
            "dataset_path": dataset_path,
            "split": split or "test",
            "supported_parameters": [],
            "parameter_ranges": {},
            "baseline_values": {},
            "baseline_sources": {},
            "generated_sweep_points": [],
            "total_planned_runs": 0,
            "safety_limits": {},
            "selected_metrics": [],
            "request": str(request or ""),
            "parsed_request": {},
        }


def run_parameter_sweep_tool(
    request: str | None = None,
    plan: dict[str, Any] | None = None,
    dataset_path: str | None = None,
    split: str | None = None,
    parameter_ranges: dict[str, Any] | None = None,
    baseline_values: dict[str, Any] | None = None,
    selected_metrics: list[str] | None = None,
    max_total_runs: int | None = None,
    max_values_per_parameter: int | None = None,
    allow_unsafe: bool = False,
    write_charts: bool = True,
) -> dict[str, Any]:
    """Run an inference-only one-factor parameter sweep after validating the plan."""
    try:
        runnable_plan = dict(plan or {})
        if not runnable_plan:
            runnable_plan = plan_parameter_sweep(
                request=request,
                dataset_path=dataset_path,
                split=split,
                parameter_ranges=parameter_ranges,
                baseline_values=baseline_values,
                selected_metrics=selected_metrics,
                max_total_runs=max_total_runs,
                max_values_per_parameter=max_values_per_parameter,
            )

        if not runnable_plan.get("success"):
            return {
                "success": False,
                "status": "invalid_plan",
                "sweep_id": None,
                "output_dir": None,
                "plan": runnable_plan,
                "results": [],
                "summary": {},
                "artifacts": {},
                "errors": list(runnable_plan.get("errors", []) or ["Parameter sweep plan is invalid."]),
                "warnings": list(runnable_plan.get("warnings", [])),
            }

        if runnable_plan.get("require_confirmation") and not allow_unsafe:
            return {
                "success": False,
                "status": "requires_confirmation",
                "sweep_id": None,
                "output_dir": None,
                "plan": runnable_plan,
                "results": [],
                "summary": {},
                "artifacts": {},
                "errors": [
                    "This parameter sweep exceeds a safety limit and requires confirmation before execution."
                ],
                "warnings": list(runnable_plan.get("warnings", [])),
            }

        parameter_sweep_runner = _load_parameter_sweep_runner_module()
        return parameter_sweep_runner.run_parameter_sweep(
            runnable_plan,
            allow_unsafe=allow_unsafe,
            write_charts=write_charts,
        )
    except Exception as exc:
        return {
            "success": False,
            "status": "runner_error",
            "sweep_id": None,
            "output_dir": None,
            "plan": plan or {},
            "results": [],
            "summary": {},
            "artifacts": {},
            "errors": [_friendly_error(exc)],
            "warnings": [],
        }


def _ablation_control_entry(
    *,
    name: str,
    label: str,
    description: str,
    applies_to: list[str],
    control_type: str,
    examples: list[str],
    default: Any = None,
    values: list[Any] | None = None,
    suggested_values: list[Any] | None = None,
    suggested_value: Any = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "name": name,
        "label": label,
        "description": description,
        "applies_to": applies_to,
        "type": control_type,
        "examples": examples,
        "default": default,
        "notes": list(notes or []),
    }
    if values is not None:
        entry["values"] = values
    if suggested_values is not None:
        entry["suggested_values"] = suggested_values
    if suggested_value is not None:
        entry["suggested_value"] = suggested_value
    return entry


def _normalize_control_lookup_text(value: str) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("colour", "color")
    text = text.replace("-", " ")
    text = text.replace("_", " ")
    text = " ".join(text.split())
    return text


def get_supported_ablation_controls() -> dict[str, Any]:
    """Return the currently implemented no-retraining ablation controls."""
    split = "test"
    sampling_modes = ["balanced", "random", "sorted"]
    seed = 42
    max_images_quick = 50
    max_images_robust = 200
    max_images_per_class = 1
    top_k_values = [1, 3, 5]
    tflite_color_correct_values = ["none", "gray_world", "max_rgb"]

    try:
        ablation_planner = _load_ablation_planner_module()
        split = str(ablation_planner.SAFE_DEFAULTS.get("split", split))
        sampling_modes = ["balanced", "random", "sorted"]
        seed = int(ablation_planner.SAFE_DEFAULTS.get("seed", seed))
        max_images_quick = int(ablation_planner.SAFE_DEFAULTS.get("max_images_quick", max_images_quick))
        max_images_robust = int(ablation_planner.SAFE_DEFAULTS.get("max_images_robust", max_images_robust))
        max_images_per_class = int(
            ablation_planner.SAFE_DEFAULTS.get("max_images_per_class", max_images_per_class)
        )
    except Exception:
        pass

    try:
        ablation_runner = _load_ablation_runner_module()
        top_k_values = list(getattr(ablation_runner, "TOP_K_VALUES", top_k_values))
        tflite_color_correct_values = list(
            getattr(ablation_runner, "TFLITE_COLOR_MODES", tflite_color_correct_values)
        )
    except Exception:
        pass

    shared_sampling_control_details = [
        _ablation_control_entry(
            name="dataset_path",
            label="Dataset path",
            description="Dataset root used to build the manifest and select evaluation images.",
            applies_to=["HOG+SVM", "TFLite"],
            control_type="path",
            examples=[r"E:\school\659\plantnet_300K\plantnet_300K", "dataset_sample"],
            notes=[
                "Changing dataset_path changes the evaluation population.",
                "Comparisons are only direct when dataset_path also matches.",
            ],
        ),
        _ablation_control_entry(
            name="split",
            label="Dataset split",
            description="Which dataset split to evaluate.",
            applies_to=["HOG+SVM", "TFLite"],
            control_type="choice",
            examples=[split, "train", "val"],
            default=split,
            values=[split, "train", "val", "validation"],
            notes=["The test split is the safest default for evaluation-only studies."],
        ),
        _ablation_control_entry(
            name="sampling_mode",
            label="Sampling mode",
            description="How images are selected into the manifest before inference runs.",
            applies_to=["HOG+SVM", "TFLite"],
            control_type="choice",
            examples=sampling_modes,
            default="balanced" if "balanced" in sampling_modes else (sampling_modes[0] if sampling_modes else None),
            values=sampling_modes,
            notes=[
                "Balanced sampling is preferred for small representative studies because it spreads coverage across classes.",
                "Random sampling is useful for stochastic checks but can underrepresent some classes.",
                "Sorted sampling is deterministic but can be biased by class and filename ordering.",
            ],
        ),
        _ablation_control_entry(
            name="max_images",
            label="Maximum images",
            description="Global sample size for the ablation run.",
            applies_to=["HOG+SVM", "TFLite"],
            control_type="count",
            examples=[str(max_images_quick), str(max_images_robust), "500"],
            suggested_values=[max_images_quick, max_images_robust],
            notes=[
                "Smaller values are faster but noisier.",
                "Larger values usually provide stronger representative evidence.",
            ],
        ),
        _ablation_control_entry(
            name="max_images_per_class",
            label="Maximum images per class",
            description="Limits how many images can be selected from each class before or during sampling.",
            applies_to=["HOG+SVM", "TFLite"],
            control_type="count_or_none",
            examples=["1", "2", "5", "None"],
            suggested_value=max_images_per_class,
            default=max_images_per_class,
            notes=[
                "Useful for balanced small-sample ablations.",
                "A value of 1 pushes the sample toward broader class coverage.",
            ],
        ),
        _ablation_control_entry(
            name="seed",
            label="Sampling seed",
            description="Random seed used for reproducible manifest sampling.",
            applies_to=["HOG+SVM", "TFLite"],
            control_type="integer",
            examples=[str(seed), "7", "99"],
            default=seed,
            notes=[
                "Changing the seed can change the sampled images without changing the core ablation configuration."
            ],
        ),
    ]

    classical_control_details = [
        _ablation_control_entry(
            name="top_k",
            label="Top-k depth",
            description="Controls how many ranked HOG+SVM class predictions are returned and evaluated in top-k summaries.",
            applies_to=["HOG+SVM"],
            control_type="choice",
            examples=[str(value) for value in top_k_values],
            values=top_k_values,
            default=top_k_values[-1] if top_k_values else None,
            notes=[
                "This is an inference/evaluation setting, not a retraining setting.",
                "It changes ranked evaluation depth, not the saved HOG feature extractor or SVM weights.",
            ],
        )
    ]

    deep_learning_control_details = [
        _ablation_control_entry(
            name="top_k",
            label="Top-k depth",
            description="Controls how many ranked TFLite class predictions are returned and evaluated in top-k summaries.",
            applies_to=["TFLite"],
            control_type="choice",
            examples=[str(value) for value in top_k_values],
            values=top_k_values,
            default=top_k_values[-1] if top_k_values else None,
            notes=["This changes evaluation depth, not the saved TFLite model weights."],
        ),
        _ablation_control_entry(
            name="color_correct",
            label="Color correction mode",
            description="Preprocessing color-correction mode applied before TFLite inference.",
            applies_to=["TFLite"],
            control_type="choice",
            examples=[str(value) for value in tflite_color_correct_values],
            values=tflite_color_correct_values,
            default=tflite_color_correct_values[0] if tflite_color_correct_values else None,
            notes=["Supported code-visible modes are none, gray_world, and max_rgb."],
        ),
    ]

    not_currently_swept_details = [
        _ablation_control_entry(
            name="hog_feature_extraction_settings",
            label="HOG feature extraction settings",
            description="Training-time HOG preprocessing choices baked into the saved classical artifact.",
            applies_to=["HOG+SVM"],
            control_type="training_time_only",
            examples=["orientations=9", "pixels_per_cell=16", "cells_per_block=2"],
            notes=["Not currently exposed by the inference-only ablation runner."],
        ),
        _ablation_control_entry(
            name="hog_orientations",
            label="HOG orientations",
            description="Number of orientation bins used during HOG feature extraction.",
            applies_to=["HOG+SVM"],
            control_type="training_time_only",
            examples=["9"],
            notes=["Not currently swept by the inference-only ablation runner."],
        ),
        _ablation_control_entry(
            name="hog_pixels_per_cell",
            label="HOG pixels per cell",
            description="Cell size used when extracting HOG features.",
            applies_to=["HOG+SVM"],
            control_type="training_time_only",
            examples=["16"],
            notes=["Would require alternate preprocessing or retrained/exported artifacts."],
        ),
        _ablation_control_entry(
            name="hog_cells_per_block",
            label="HOG cells per block",
            description="Block size used when normalizing HOG features.",
            applies_to=["HOG+SVM"],
            control_type="training_time_only",
            examples=["2"],
            notes=["Not currently exposed by the inference-only ablation runner."],
        ),
        _ablation_control_entry(
            name="svm_c",
            label="SVM C",
            description="Regularization strength for the saved SVM classifier.",
            applies_to=["HOG+SVM"],
            control_type="training_time_only",
            examples=["1.0"],
            notes=["Not currently swept; changing it would require a different trained artifact."],
        ),
        _ablation_control_entry(
            name="svm_kernel",
            label="SVM kernel",
            description="Kernel choice used by the saved SVM classifier.",
            applies_to=["HOG+SVM"],
            control_type="training_time_only",
            examples=["linear", "rbf"],
            notes=["Not currently exposed by the inference-only ablation runner."],
        ),
        _ablation_control_entry(
            name="svm_gamma",
            label="SVM gamma",
            description="Kernel coefficient for non-linear SVM kernels.",
            applies_to=["HOG+SVM"],
            control_type="training_time_only",
            examples=["scale", "auto"],
            notes=["Only relevant when training/exporting alternate SVM artifacts."],
        ),
        _ablation_control_entry(
            name="svm_class_weight",
            label="Class weighting",
            description="Class weighting strategy used during SVM training.",
            applies_to=["HOG+SVM"],
            control_type="training_time_only",
            examples=["balanced"],
            notes=["Not currently changed by the inference-only ablation system."],
        ),
        _ablation_control_entry(
            name="feature_scaling_or_preprocessing_pipeline",
            label="Feature scaling and preprocessing pipeline",
            description="Training-time scaling or preprocessing pipeline choices baked into the saved classical artifact.",
            applies_to=["HOG+SVM"],
            control_type="training_time_only",
            examples=["StandardScaler + SVM pipeline"],
            notes=["Would require a separate training or artifact-generation workflow."],
        ),
    ]

    return {
        "success": True,
        "system_type": "inference_only_ablation",
        "inference_only": True,
        "shared_sampling_controls": [entry["name"] for entry in shared_sampling_control_details],
        "shared_sampling_control_details": shared_sampling_control_details,
        "classical_controls": [entry["name"] for entry in classical_control_details],
        "classical_control_details": classical_control_details,
        "deep_learning_controls": [entry["name"] for entry in deep_learning_control_details],
        "deep_learning_control_details": deep_learning_control_details,
        "not_currently_swept": [entry["name"] for entry in not_currently_swept_details],
        "not_currently_swept_details": not_currently_swept_details,
        "caveats": [
            "The current ablation system is inference-only and does not retrain models.",
            "The classical HOG+SVM path evaluates saved artifacts rather than sweeping training-time hyperparameters.",
            "TFLite color-correction ablations are separate and do not apply to HOG+SVM.",
        ],
        "recommended_next_steps": [
            "Use balanced sampling for small representative studies.",
            "Repeat the same configuration with different seeds when you need stronger evidence.",
        ],
    }


def get_ablation_control_detail(query: str) -> dict[str, Any]:
    """Look up one ablation control or unswept setting by name or common alias."""
    controls = get_supported_ablation_controls()
    if not controls.get("success"):
        return {
            "success": False,
            "query": query,
            "matched_name": None,
            "matched_label": None,
            "category": None,
            "detail": None,
            "is_supported_control": False,
            "is_not_currently_swept": False,
            "message": "Ablation control metadata is unavailable.",
        }

    shared_details = list(controls.get("shared_sampling_control_details", []))
    classical_details = list(controls.get("classical_control_details", []))
    deep_details = list(controls.get("deep_learning_control_details", []))
    unswept_details = list(controls.get("not_currently_swept_details", []))

    detail_groups = [
        ("shared_sampling_control", True, False, shared_details),
        ("classical_control", True, False, classical_details),
        ("deep_learning_control", True, False, deep_details),
        ("not_currently_swept", False, True, unswept_details),
    ]

    alias_map = {
        "dataset path": "dataset_path",
        "split": "split",
        "dataset split": "split",
        "sampling mode": "sampling_mode",
        "balanced sampling": "sampling_mode",
        "random sampling": "sampling_mode",
        "sorted sampling": "sampling_mode",
        "max images": "max_images",
        "maximum images": "max_images",
        "max images per class": "max_images_per_class",
        "maximum images per class": "max_images_per_class",
        "seed": "seed",
        "sampling seed": "seed",
        "top k": "top_k",
        "topk": "top_k",
        "top k depth": "top_k",
        "color correction": "color_correct",
        "color correct": "color_correct",
        "color_correct": "color_correct",
        "svm c": "svm_c",
        "svm kernel": "svm_kernel",
        "svm gamma": "svm_gamma",
        "hog feature extraction settings": "hog_feature_extraction_settings",
        "hog orientations": "hog_orientations",
        "hog pixels per cell": "hog_pixels_per_cell",
        "hog cells per block": "hog_cells_per_block",
        "class weighting": "svm_class_weight",
        "feature scaling": "feature_scaling_or_preprocessing_pipeline",
        "preprocessing pipeline": "feature_scaling_or_preprocessing_pipeline",
    }

    normalized_query = _normalize_control_lookup_text(query)
    if "why is balanced sampling preferred" in normalized_query:
        normalized_query = "balanced sampling"
    elif normalized_query.startswith("what does "):
        normalized_query = normalized_query[len("what does ") :].strip()
        if " mean" in normalized_query:
            normalized_query = normalized_query.split(" mean", 1)[0].strip()
        elif " do" in normalized_query:
            normalized_query = normalized_query.split(" do", 1)[0].strip()
    elif normalized_query.startswith("what is "):
        normalized_query = normalized_query[len("what is ") :].strip(" .?")
    elif normalized_query.startswith("explain "):
        normalized_query = normalized_query[len("explain ") :].strip(" .?")
    elif normalized_query.startswith("define "):
        normalized_query = normalized_query[len("define ") :].strip(" .?")

    for suffix in (" for this ablation", " in an ablation", " for the ablation", " in this ablation"):
        if normalized_query.endswith(suffix):
            normalized_query = normalized_query[: -len(suffix)].strip()

    target_name = alias_map.get(normalized_query, normalized_query.replace(" ", "_"))

    for category, is_supported, is_unswept, entries in detail_groups:
        for detail in entries:
            if not isinstance(detail, dict):
                continue
            detail_name = str(detail.get("name") or "")
            if detail_name == target_name:
                return {
                    "success": True,
                    "query": query,
                    "matched_name": detail_name,
                    "matched_label": detail.get("label"),
                    "category": category,
                    "detail": detail,
                    "is_supported_control": is_supported,
                    "is_not_currently_swept": is_unswept,
                    "message": None,
                }

    return {
        "success": False,
        "query": query,
        "matched_name": None,
        "matched_label": None,
        "category": None,
        "detail": None,
        "is_supported_control": False,
        "is_not_currently_swept": False,
        "message": (
            "That was not recognized as a supported ablation control or known unswept classical setting. "
            "Ask about available ablation controls or try a known term like sampling_mode, max_images_per_class, top_k, seed, color_correct, or svm_c."
        ),
    }


def explain_ablation_control(query: str) -> dict[str, Any]:
    """Tool-friendly wrapper for explaining one ablation control or unswept setting."""
    return get_ablation_control_detail(query)


def _build_langchain_tools() -> list[Any]:
    try:
        from langchain_core.tools import StructuredTool
    except Exception:
        try:
            from langchain.tools import StructuredTool
        except Exception:
            return []

    try:
        return [
            StructuredTool.from_function(
                func=classify_with_tflite,
                name="classify_with_tflite",
                description="Classify a plant image with the default TensorFlow Lite model.",
            ),
            StructuredTool.from_function(
                func=classify_with_hog_svm,
                name="classify_with_hog_svm",
                description="Classify a plant image with the default HOG+SVM artifacts when available.",
            ),
            StructuredTool.from_function(
                func=compare_classifiers,
                name="compare_classifiers",
                description="Compare TFLite and HOG+SVM predictions for the same image.",
            ),
            StructuredTool.from_function(
                func=get_artifact_status,
                name="get_artifact_status",
                description="Inspect whether required model and label artifacts are available.",
            ),
            StructuredTool.from_function(
                func=get_metrics_summary,
                name="get_metrics_summary",
                description="Inspect which metrics files currently exist under the result directory.",
            ),
            StructuredTool.from_function(
                func=get_ablation_feasibility,
                name="get_ablation_feasibility",
                description="Report which ablation or experiment ideas appear feasible now versus what still requires dataset access or retraining.",
            ),
            StructuredTool.from_function(
                func=build_dataset_manifest,
                name="build_dataset_manifest",
                description="Build a CSV manifest for a PlantNet-style dataset split such as test, train, val, or validation.",
            ),
            StructuredTool.from_function(
                func=get_ablation_history,
                name="get_ablation_history",
                description="Load recent persistent ablation runs from the append-only history file.",
            ),
            StructuredTool.from_function(
                func=get_ablation_results,
                name="get_ablation_results",
                description="Load recent ablation results from the persistent append-only history store.",
            ),
            StructuredTool.from_function(
                func=get_latest_ablation_result,
                name="get_latest_ablation_result",
                description="Load the most recent ablation run from persistent history.",
            ),
            StructuredTool.from_function(
                func=get_ablation_result_by_id,
                name="get_ablation_result_by_id",
                description="Load one specific ablation run from persistent history by run_id.",
            ),
            StructuredTool.from_function(
                func=compare_ablation_results,
                name="compare_ablation_results",
                description="Compare explicit run IDs or the most recent comparable ablation runs from persistent history.",
            ),
            StructuredTool.from_function(
                func=export_ablation_results,
                name="export_ablation_results",
                description="Regenerate the canonical ablation history CSV, JSON, Markdown, and chart artifacts from JSONL history.",
            ),
            StructuredTool.from_function(
                func=get_ablation_recommendations,
                name="get_ablation_recommendations",
                description="Return conservative ablation recommendations based on persistent run history when enough runs exist.",
            ),
            StructuredTool.from_function(
                func=get_supported_ablation_controls,
                name="get_supported_ablation_controls",
                description="Describe the currently implemented inference-only ablation controls and caveats for classical HOG+SVM and TFLite paths.",
            ),
            StructuredTool.from_function(
                func=explain_ablation_control,
                name="explain_ablation_control",
                description="Explain a specific ablation control or unswept hyperparameter, such as sampling_mode, max_images_per_class, top_k, color_correct, SVM C, or HOG pixels per cell. Use this when the user asks what an ablation setting means or whether it is currently supported.",
            ),
            StructuredTool.from_function(
                func=get_parameter_sweep_support,
                name="get_parameter_sweep_support",
                description="Describe the inference-only one-factor-at-a-time parameter sweep feature, including supported parameters, supported metrics, defaults, and safety limits. Supported sweep parameters are top_k, sampling_mode, max_images, max_images_per_class, and seed. Unsupported HOG/SVM training hyperparameters are not swept.",
            ),
            StructuredTool.from_function(
                func=parse_parameter_sweep_request_tool,
                name="parse_parameter_sweep_request_tool",
                description="Parse a natural-language inference-only parameter sweep request into candidate parameter ranges, baseline values, and selected metrics without running anything.",
            ),
            StructuredTool.from_function(
                func=plan_parameter_sweep,
                name="plan_parameter_sweep",
                description="Plan an inference-only one-factor-at-a-time parameter sweep over top_k, sampling_mode, max_images, max_images_per_class, and seed. Use this before running a sweep, especially when the user provides ranges like top_k 1,3,5 or max_images 50,100,200. Unsupported HOG/SVM training hyperparameters are not swept.",
            ),
            StructuredTool.from_function(
                func=run_parameter_sweep_tool,
                name="run_parameter_sweep_tool",
                description="Run a validated inference-only one-factor-at-a-time parameter sweep over top_k, sampling_mode, max_images, max_images_per_class, and seed. Use planning before running when possible. Do not use this for unsupported HOG/SVM training-time hyperparameters or retraining workflows.",
            ),
            StructuredTool.from_function(
                func=plan_ablation_study,
                name="plan_ablation_study",
                description="Parse an ablation-study request and identify missing parameters, safe defaults, and warnings.",
            ),
            StructuredTool.from_function(
                func=run_sample_ablation,
                name="run_sample_ablation",
                description="Run no-retraining sample ablations from a previously built dataset manifest CSV.",
            ),
            StructuredTool.from_function(
                func=run_planned_ablation,
                name="run_planned_ablation",
                description="Build a manifest and run a no-retraining ablation study using the provided dataset path and sampling settings.",
            ),
        ]
    except Exception:
        return []


AGENT_TOOLS = _build_langchain_tools()
LANGCHAIN_AVAILABLE = bool(AGENT_TOOLS)
