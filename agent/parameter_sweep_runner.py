#!/usr/bin/env python3
"""
Backend runner for inference-only one-factor-at-a-time parameter sweeps.
"""
from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from agent.ablation_runner import run_sample_ablation
from agent.dataset_manifest import build_manifest
from agent.model_imports import ensure_model_project_on_path
from agent.parameter_sweep_planner import (
    DEFAULT_SELECTED_METRICS,
    SUPPORTED_SWEEP_METRICS,
    build_parameter_sweep_plan,
)

_METRIC_LABELS = {
    "tflite_top1_accuracy": "TFLite top-1 accuracy",
    "tflite_topk_accuracy": "TFLite top-k accuracy",
    "hog_top1_accuracy": "HOG+SVM top-1 accuracy",
    "hog_topk_accuracy": "HOG+SVM top-k accuracy",
    "model_agreement_rate": "Model agreement rate",
}


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")


def _parameter_sweep_output_root() -> Path:
    ensure_model_project_on_path()
    import app_config

    return Path(app_config.DEFAULT_RESULT_DIR)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")


def _save_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _selected_metrics_from_plan(plan: dict[str, Any]) -> list[str]:
    selected = [str(metric).strip() for metric in plan.get("selected_metrics", []) if str(metric).strip()]
    return selected or list(DEFAULT_SELECTED_METRICS)


def _metric_label(metric_name: str) -> str:
    return _METRIC_LABELS.get(metric_name, metric_name.replace("_", " "))


def _safe_metric_value(value: Any) -> float:
    if value is None:
        return math.nan
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else math.nan
    text = str(value).strip()
    if not text or text.lower() in {"n/a", "nan", "none"}:
        return math.nan
    try:
        numeric = float(text)
    except Exception:
        return math.nan
    return numeric if math.isfinite(numeric) else math.nan


def _validate_runnable_plan(plan: dict[str, Any], *, allow_unsafe: bool) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = list(plan.get("warnings", []))

    if not isinstance(plan, dict):
        return {
            "success": False,
            "errors": ["Parameter sweep plan must be a dictionary."],
            "warnings": [],
        }

    if plan.get("study_type") not in {None, "parameter_sweep"}:
        errors.append("Parameter sweep runner only accepts plans with study_type `parameter_sweep`.")
    if not plan.get("success"):
        errors.extend(plan.get("errors", []) or ["Parameter sweep plan is not marked successful."])
    if not plan.get("generated_sweep_points"):
        errors.append("Parameter sweep plan does not contain any generated_sweep_points to execute.")
    if plan.get("require_confirmation") and not allow_unsafe:
        errors.append("Parameter sweep plan requires confirmation before execution; rerun with allow_unsafe=True to override.")
    if not plan.get("dataset_path"):
        errors.append("Parameter sweep plan must include dataset_path before execution.")

    selected_metrics = _selected_metrics_from_plan(plan)
    invalid_metrics = [metric for metric in selected_metrics if metric not in SUPPORTED_SWEEP_METRICS]
    if invalid_metrics:
        errors.append(
            "Parameter sweep plan includes unsupported selected_metrics values: "
            + ", ".join(invalid_metrics)
            + "."
        )

    return {
        "success": not errors,
        "errors": errors,
        "warnings": warnings,
    }


def _point_output_name(index: int, varied_parameter: str, varied_value: Any) -> str:
    safe_value = str(varied_value).replace(" ", "_").replace("/", "_").replace("\\", "_")
    return f"point_{index:03d}_{varied_parameter}_{safe_value}"


def _top_k_values_for_point(parameter_values: dict[str, Any]) -> tuple[int, ...]:
    top_k = int(parameter_values["top_k"])
    if top_k == 1:
        return (1,)
    return (1, top_k)


def _run_sweep_point(
    *,
    plan: dict[str, Any],
    sweep_dir: Path,
    sweep_id: str,
    point_index: int,
    point: dict[str, Any],
) -> dict[str, Any]:
    parameter_values = dict(point.get("parameter_values", {}))
    varied_parameter = str(point.get("varied_parameter", "unknown"))
    varied_value = point.get("varied_value")
    point_name = _point_output_name(point_index, varied_parameter, varied_value)
    point_dir = _ensure_dir(sweep_dir / point_name)
    manifest_path = point_dir / f"{point_name}_manifest.csv"

    manifest_result = build_manifest(
        dataset_path=str(plan["dataset_path"]),
        split=str(plan["split"]),
        out_csv=str(manifest_path),
        max_images=int(parameter_values["max_images"]),
        sampling_mode=str(parameter_values["sampling_mode"]),
        seed=int(parameter_values["seed"]),
        max_images_per_class=int(parameter_values["max_images_per_class"]),
    )
    if not manifest_result.get("success") or not manifest_result.get("manifest_path"):
        return {
            "sweep_point_id": point_name,
            "point_index": point_index,
            "varied_parameter": varied_parameter,
            "varied_value": varied_value,
            "parameter_values": parameter_values,
            "status": "failed",
            "manifest": manifest_result,
            "ablation": None,
            "warning": manifest_result.get("warning"),
            "error": manifest_result.get("error") or "Manifest generation failed.",
        }

    ablation_result = run_sample_ablation(
        manifest_path=str(manifest_result["manifest_path"]),
        max_images=int(parameter_values["max_images"]),
        top_k_values=_top_k_values_for_point(parameter_values),
        summary_top_k=int(parameter_values["top_k"]),
        output_dir=str(point_dir),
        append_history=False,
        write_charts=False,
    )
    return {
        "sweep_point_id": point_name,
        "point_index": point_index,
        "varied_parameter": varied_parameter,
        "varied_value": varied_value,
        "parameter_values": parameter_values,
        "status": "completed" if ablation_result.get("success") else "failed",
        "manifest": manifest_result,
        "ablation": ablation_result,
        "warning": ablation_result.get("warning") or manifest_result.get("warning"),
        "error": ablation_result.get("error"),
    }


def _result_row_from_point(point_result: dict[str, Any]) -> dict[str, Any]:
    parameter_values = dict(point_result.get("parameter_values", {}))
    ablation = point_result.get("ablation") or {}
    summary = ablation.get("summary") or {}
    metrics = summary.get("metrics") or {}
    error_bits = [
        str(value).strip()
        for value in (
            point_result.get("error"),
            point_result.get("warning"),
        )
        if str(value or "").strip()
    ]
    return {
        "sweep_id": summary.get("parent_sweep_id") or point_result.get("sweep_id", ""),
        "sweep_point_id": point_result.get("sweep_point_id", ""),
        "varied_parameter": point_result.get("varied_parameter", ""),
        "varied_value": point_result.get("varied_value", ""),
        "top_k": parameter_values.get("top_k"),
        "sampling_mode": parameter_values.get("sampling_mode"),
        "max_images": parameter_values.get("max_images"),
        "max_images_per_class": parameter_values.get("max_images_per_class"),
        "seed": parameter_values.get("seed"),
        "num_images_evaluated": summary.get("num_images_evaluated", ablation.get("num_images_evaluated")),
        "num_classes": summary.get("num_classes"),
        "tflite_top1_accuracy": metrics.get("tflite_top1_accuracy"),
        "tflite_topk_accuracy": metrics.get("tflite_topk_accuracy"),
        "hog_top1_accuracy": metrics.get("hog_top1_accuracy"),
        "hog_topk_accuracy": metrics.get("hog_topk_accuracy"),
        "model_agreement_rate": metrics.get("model_agreement_rate"),
        "status": point_result.get("status", "failed"),
        "errors": " | ".join(error_bits),
    }


def _group_rows_by_parameter(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("varied_parameter", "unknown")), []).append(dict(row))
    return grouped


def _chart_filename(varied_parameter: str) -> str:
    safe_name = str(varied_parameter or "unknown").strip().lower().replace(" ", "_")
    return f"sweep_{safe_name}_metrics.png"


def _baseline_caption(baseline_values: dict[str, Any], varied_parameter: str) -> str:
    held_fixed = [
        f"{key}={value}"
        for key, value in baseline_values.items()
        if key != varied_parameter
    ]
    if not held_fixed:
        return "Inference-only sweep."
    return "Baseline held fixed: " + ", ".join(held_fixed)


def _parameter_value_order(plan: dict[str, Any], varied_parameter: str) -> dict[str, int]:
    ordered_values = list((plan.get("parameter_ranges") or {}).get(varied_parameter, []))
    return {str(value): idx for idx, value in enumerate(ordered_values)}


def _ordered_rows_for_parameter(plan: dict[str, Any], varied_parameter: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order_map = _parameter_value_order(plan, varied_parameter)
    if not order_map:
        return list(rows)
    return sorted(
        rows,
        key=lambda row: order_map.get(str(row.get("varied_value")), len(order_map)),
    )


def _write_parameter_sweep_charts(
    *,
    plan: dict[str, Any],
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    if not rows:
        return []

    charts_dir = _ensure_dir(output_dir / "charts")
    chart_entries: list[dict[str, Any]] = []
    grouped_rows = _group_rows_by_parameter(rows)
    selected_metrics = _selected_metrics_from_plan(plan)
    baseline_values = dict(summary.get("baseline_values", {}))

    for varied_parameter, parameter_rows in grouped_rows.items():
        ordered_rows = _ordered_rows_for_parameter(plan, varied_parameter, parameter_rows)
        x_labels = [str(row.get("varied_value")) for row in ordered_rows]
        x_positions = list(range(len(ordered_rows)))
        chart_warnings: list[str] = []

        fig, ax = plt.subplots(figsize=(9.5, 5.25), dpi=110)
        plotted_metrics: list[str] = []

        for metric_name in selected_metrics:
            y_values = [_safe_metric_value(row.get(metric_name)) for row in ordered_rows]
            if all(math.isnan(value) for value in y_values):
                chart_warnings.append(
                    f"{_metric_label(metric_name)} was omitted because all values were missing for {varied_parameter}."
                )
                continue
            if any(math.isnan(value) for value in y_values):
                chart_warnings.append(
                    f"{_metric_label(metric_name)} had missing values for {varied_parameter}; missing points were skipped."
                )
            ax.plot(
                x_positions,
                y_values,
                marker="o",
                linewidth=2.0,
                label=_metric_label(metric_name),
            )
            plotted_metrics.append(metric_name)

        if not plotted_metrics:
            plt.close(fig)
            chart_entries.append(
                {
                    "path": None,
                    "filename": _chart_filename(varied_parameter),
                    "varied_parameter": varied_parameter,
                    "metrics": [],
                    "baseline_values": {
                        key: value
                        for key, value in baseline_values.items()
                        if key != varied_parameter
                    },
                    "warnings": chart_warnings or [f"No plottable metric values were available for {varied_parameter}."],
                }
            )
            continue

        ax.set_title(f"Parameter Sweep: {varied_parameter}")
        ax.set_xlabel(varied_parameter)
        ax.set_ylabel("Accuracy / rate")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=25, ha="right")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="best")
        fig.text(0.5, 0.94, _baseline_caption(baseline_values, varied_parameter), ha="center", va="top", fontsize=9)
        fig.text(0.5, 0.02, "Inference-only parameter sweep; no model retraining was performed.", ha="center", va="bottom", fontsize=8)
        fig.tight_layout(rect=(0, 0.05, 1, 0.9))

        chart_path = charts_dir / _chart_filename(varied_parameter)
        fig.savefig(chart_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        chart_entries.append(
            {
                "path": str(chart_path),
                "filename": chart_path.name,
                "varied_parameter": varied_parameter,
                "metrics": plotted_metrics,
                "baseline_values": {
                    key: value
                    for key, value in baseline_values.items()
                    if key != varied_parameter
                },
                "warnings": chart_warnings,
            }
        )

    return chart_entries


def _build_sweep_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Parameter Sweep Summary",
        "",
        "This is an inference-only one-factor-at-a-time parameter sweep. No model retraining was performed.",
        "",
        "## Sweep Metadata",
        f"Sweep ID: `{summary.get('sweep_id')}`",
        f"Output directory: `{summary.get('output_dir')}`",
        f"Dataset path: `{summary.get('dataset_path')}`",
        f"Split: `{summary.get('split')}`",
        "",
        "## Baseline Values",
    ]
    for key, value in dict(summary.get("baseline_values", {})).items():
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(
        [
            "",
            "## Parameters Swept",
            ", ".join(summary.get("parameters_swept", [])) or "None",
            "",
            "## Selected Metrics",
            ", ".join(summary.get("selected_metrics", [])) or "None",
            "",
            "## Execution Status",
            f"Planned runs: {summary.get('total_planned_runs', 0)}",
            f"Completed runs: {summary.get('total_completed_runs', 0)}",
            f"Failed runs: {summary.get('total_failed_runs', 0)}",
        ]
    )
    if summary.get("warnings"):
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {warning}" for warning in summary["warnings"])
    if summary.get("errors"):
        lines.extend(["", "## Errors"])
        lines.extend(f"- {error}" for error in summary["errors"])

    charts = list(summary.get("charts", []))
    if charts:
        lines.extend(["", "## Generated Charts"])
        for chart in charts:
            filename = chart.get("filename") or chart.get("path") or "unknown"
            varied_parameter = chart.get("varied_parameter", "unknown")
            metrics = ", ".join(_metric_label(metric) for metric in chart.get("metrics", [])) or "None"
            baseline_values = ", ".join(
                f"{key}={value}" for key, value in dict(chart.get("baseline_values", {})).items()
            ) or "None"
            lines.append(f"- `{filename}` | varied parameter: `{varied_parameter}` | metrics: {metrics} | baseline held fixed: {baseline_values}")
            for warning in chart.get("warnings", []):
                lines.append(f"  warning: {warning}")

    lines.extend(
        [
            "",
            "## Notes",
            "- Full per-point artifacts are stored in per-point subdirectories under the sweep output directory.",
            "- Interactive chart viewers are planned for a later step.",
        ]
    )
    return "\n".join(lines)


def run_parameter_sweep(plan: dict[str, Any], *, allow_unsafe: bool = False, write_charts: bool = True) -> dict[str, Any]:
    validation = _validate_runnable_plan(plan, allow_unsafe=allow_unsafe)
    if not validation["success"]:
        return {
            "success": False,
            "status": "invalid_plan",
            "sweep_id": None,
            "output_dir": None,
            "plan": plan,
            "results": [],
            "summary": {},
            "artifacts": {},
            "errors": validation["errors"],
            "warnings": validation["warnings"],
        }

    sweep_id = f"parameter_sweep_{_timestamp_slug()}"
    sweep_dir = _ensure_dir(_parameter_sweep_output_root() / sweep_id)

    plan_copy = json.loads(json.dumps(plan))
    plan_copy["study_type"] = "parameter_sweep"
    plan_copy["execution_started_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    plan_copy["output_dir"] = str(sweep_dir)

    results: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    completed = 0
    failed = 0

    for index, point in enumerate(plan.get("generated_sweep_points", []), start=1):
        try:
            point_result = _run_sweep_point(
                plan=plan_copy,
                sweep_dir=sweep_dir,
                sweep_id=sweep_id,
                point_index=index,
                point=point,
            )
        except Exception as exc:
            point_result = {
                "sweep_point_id": _point_output_name(
                    index,
                    str(point.get("varied_parameter", "unknown")),
                    point.get("varied_value"),
                ),
                "point_index": index,
                "varied_parameter": point.get("varied_parameter"),
                "varied_value": point.get("varied_value"),
                "parameter_values": dict(point.get("parameter_values", {})),
                "status": "failed",
                "manifest": None,
                "ablation": None,
                "warning": None,
                "error": str(exc),
            }
        point_result["sweep_id"] = sweep_id
        ablation_summary = ((point_result.get("ablation") or {}).get("summary") or {})
        if ablation_summary:
            ablation_summary["parent_sweep_id"] = sweep_id
        results.append(point_result)
        rows.append(_result_row_from_point(point_result))
        if point_result.get("status") == "completed":
            completed += 1
        else:
            failed += 1

    artifacts = {
        "plan_json": str(sweep_dir / "parameter_sweep_plan.json"),
        "results_csv": str(sweep_dir / "parameter_sweep_results.csv"),
        "summary_json": str(sweep_dir / "parameter_sweep_summary.json"),
        "summary_markdown": str(sweep_dir / "parameter_sweep_summary.md"),
    }

    chart_entries: list[dict[str, Any]] = []
    if write_charts and completed > 0:
        chart_entries = _write_parameter_sweep_charts(
            plan=plan_copy,
            summary={
                "baseline_values": dict(plan_copy.get("baseline_values", {})),
            },
            rows=rows,
            output_dir=sweep_dir,
        )
        if any(chart.get("path") for chart in chart_entries):
            artifacts["charts_dir"] = str(sweep_dir / "charts")

    summary = {
        "success": failed == 0 and completed > 0,
        "status": "completed" if failed == 0 else ("partial_success" if completed > 0 else "failed"),
        "study_type": "parameter_sweep",
        "sweep_id": sweep_id,
        "output_dir": str(sweep_dir),
        "dataset_path": plan_copy.get("dataset_path"),
        "split": plan_copy.get("split"),
        "total_planned_runs": int(plan_copy.get("total_planned_runs", 0)),
        "total_completed_runs": completed,
        "total_failed_runs": failed,
        "baseline_values": dict(plan_copy.get("baseline_values", {})),
        "baseline_sources": dict(plan_copy.get("baseline_sources", {})),
        "selected_metrics": _selected_metrics_from_plan(plan_copy),
        "parameters_swept": list(plan_copy.get("parameter_ranges", {}).keys()),
        "artifacts": dict(artifacts),
        "per_parameter_results": _group_rows_by_parameter(rows),
        "charts": chart_entries,
        "warnings": validation["warnings"],
        "errors": [row["errors"] for row in rows if row.get("errors")],
    }

    _save_json(Path(artifacts["plan_json"]), plan_copy)
    _save_csv(
        Path(artifacts["results_csv"]),
        rows,
        [
            "sweep_id",
            "sweep_point_id",
            "varied_parameter",
            "varied_value",
            "top_k",
            "sampling_mode",
            "max_images",
            "max_images_per_class",
            "seed",
            "num_images_evaluated",
            "num_classes",
            "tflite_top1_accuracy",
            "tflite_topk_accuracy",
            "hog_top1_accuracy",
            "hog_topk_accuracy",
            "model_agreement_rate",
            "status",
            "errors",
        ],
    )
    _save_json(Path(artifacts["summary_json"]), summary)
    Path(artifacts["summary_markdown"]).write_text(_build_sweep_markdown(summary), encoding="utf-8")

    return {
        "success": summary["success"],
        "status": summary["status"],
        "sweep_id": sweep_id,
        "output_dir": str(sweep_dir),
        "plan": plan_copy,
        "results": results,
        "summary": summary,
        "artifacts": artifacts,
        "errors": summary["errors"],
        "warnings": validation["warnings"],
    }
