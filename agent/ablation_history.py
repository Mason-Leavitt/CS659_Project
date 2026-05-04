#!/usr/bin/env python3
"""
Persistent append-only history helpers for no-retraining ablation runs.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any

import matplotlib.pyplot as plt

from agent.model_imports import ensure_model_project_on_path


def _load_app_config() -> Any:
    ensure_model_project_on_path()
    import app_config

    return app_config


def _history_dir() -> Path:
    app_config = _load_app_config()
    return app_config.ensure_dir(app_config.DEFAULT_RESULT_DIR / "ablation_history")


def _history_path() -> Path:
    return _history_dir() / "ablation_history.jsonl"


def _history_table_path() -> Path:
    return _history_dir() / "ablation_history_table.csv"


def _history_summary_json_path() -> Path:
    return _history_dir() / "ablation_history_summary.json"


def _history_summary_md_path() -> Path:
    return _history_dir() / "ablation_history_summary.md"


def _history_metrics_png_path() -> Path:
    return _history_dir() / "ablation_history_metrics.png"


def _parse_ablation_label_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        pass
    match = re.search(r"(20\d{2})(\d{2})(\d{2})[_-]?(\d{2})(\d{2})(\d{2})", text)
    if not match:
        return None
    try:
        return datetime(
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
            int(match.group(4)),
            int(match.group(5)),
            int(match.group(6)),
        )
    except Exception:
        return None


def _fallback_short_text(value: Any, limit: int = 16) -> str:
    text = str(value or "").strip()
    if not text:
        return "run"
    if len(text) <= limit:
        return text
    return text[-limit:]


def _short_ablation_run_label(row: dict[str, Any], *, include_seconds: bool = False) -> str:
    for candidate in (row.get("created_at"), row.get("run_id"), row.get("output_dir")):
        parsed = _parse_ablation_label_timestamp(candidate)
        if parsed is not None:
            return parsed.strftime("%m%d_%H%M%S" if include_seconds else "%m%d_%H%M")
    return _fallback_short_text(row.get("run_id") or row.get("output_dir") or row.get("created_at"))


def make_unique_ablation_run_labels(rows: list[dict[str, Any]]) -> list[str]:
    base_labels = [_short_ablation_run_label(row, include_seconds=False) for row in rows]
    counts = Counter(base_labels)
    detailed_labels = [
        _short_ablation_run_label(row, include_seconds=True) if counts[base_label] > 1 else base_label
        for row, base_label in zip(rows, base_labels)
    ]
    detailed_counts = Counter(detailed_labels)
    used_counts: Counter[str] = Counter()
    final_labels: list[str] = []
    for label in detailed_labels:
        if detailed_counts[label] <= 1:
            final_labels.append(label)
            continue
        used_counts[label] += 1
        final_labels.append(f"{label}_{used_counts[label]}")
    return final_labels


def _coerce_metrics(entry: dict[str, Any]) -> dict[str, Any]:
    metrics = entry.get("metrics", {})
    return metrics if isinstance(metrics, dict) else {}


def _coerce_run_id() -> str:
    return datetime.now(timezone.utc).strftime("ablation_%Y%m%d_%H%M%S_%f_UTC")


def _safe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _range(values: list[float]) -> float | None:
    if not values:
        return None
    return max(values) - min(values)


def _message_rate(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.1f}%"


def _normalize_dataset_path(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "<unknown_dataset>"
    return text.replace("/", "\\")


def _dataset_label(value: Any) -> str:
    normalized = _normalize_dataset_path(value)
    if normalized == "<unknown_dataset>":
        return normalized
    try:
        name = Path(normalized).name.strip()
    except Exception:
        name = ""
    if name:
        return name
    return normalized


def _group_key(entry: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        _normalize_dataset_path(entry.get("dataset_path")),
        str(entry.get("split", "")),
        str(entry.get("sampling_mode", "")),
        str(entry.get("max_images", "")),
        str(entry.get("max_images_per_class", "")),
    )


def _normalize_comparison_value(field: str, value: Any) -> str:
    if field == "dataset_path":
        return _normalize_dataset_path(value)
    text = str(value).strip() if value is not None else ""
    return text or "<unknown>"


def describe_comparison_validity(entries: list[dict[str, Any]]) -> dict[str, Any]:
    core_fields = ("dataset_path", "split", "sampling_mode", "max_images", "max_images_per_class")
    observed_fields = ("num_images_evaluated", "num_classes")
    exploratory_fields = ("seed",)

    if not entries:
        return {
            "status": "not_direct",
            "title": "No runs selected",
            "summary": "No ablation runs were available for comparison.",
            "caveats": ["No comparison can be evaluated because no runs were selected."],
            "differing_fields": [],
            "matched_fields": [],
            "unknown_fields": [],
        }

    if len(entries) == 1:
        return {
            "status": "direct",
            "title": "Single run selected",
            "summary": "Only one ablation run is selected, so no cross-run comparability issues apply yet.",
            "caveats": [],
            "differing_fields": [],
            "matched_fields": list(core_fields),
            "unknown_fields": [],
        }

    differing_fields: list[str] = []
    matched_fields: list[str] = []
    unknown_fields: list[str] = []
    caveats: list[str] = []
    all_fields = core_fields + observed_fields + exploratory_fields

    field_values: dict[str, list[str]] = {}
    for field in all_fields:
        values = [_normalize_comparison_value(field, entry.get(field)) for entry in entries]
        unique_values = sorted(set(values))
        field_values[field] = unique_values
        if len(unique_values) > 1:
            differing_fields.append(field)
        else:
            matched_fields.append(field)
        if any(value.startswith("<unknown") for value in unique_values):
            unknown_fields.append(field)

    major_differences = [field for field in core_fields if field in differing_fields]
    observed_differences = [field for field in observed_fields if field in differing_fields]
    exploratory_differences = [field for field in exploratory_fields if field in differing_fields]
    unknown_core = [field for field in core_fields if field in unknown_fields]

    if major_differences:
        status = "not_direct"
        title = "Not directly comparable"
        caveats.append(
            "These runs are not directly comparable because core comparison fields differ: "
            + ", ".join(major_differences)
            + "."
        )
    elif unknown_core:
        status = "partial"
        title = "Partially comparable"
        caveats.append(
            "These runs may be only partially comparable because core comparison fields are missing or unknown: "
            + ", ".join(unknown_core)
            + "."
        )
    elif observed_differences or exploratory_differences or unknown_fields:
        status = "partial"
        title = "Partially comparable"
    else:
        status = "direct"
        title = "Directly comparable"

    if status == "direct":
        caveats.append(
            "These runs are directly comparable because dataset_path, split, sampling_mode, max_images, "
            "and max_images_per_class match."
        )
    else:
        if observed_differences:
            caveats.append(
                "Observed evaluation coverage differs across runs: " + ", ".join(observed_differences) + "."
            )
        if exploratory_differences:
            caveats.append(
                "Seed differences make this comparison more exploratory even though the core configuration may match."
            )
        unknown_non_core = [field for field in unknown_fields if field not in unknown_core]
        if unknown_non_core:
            caveats.append(
                "Some comparison fields are missing or unknown: " + ", ".join(unknown_non_core) + "."
            )

    summary_parts: list[str] = [caveats[0]] if caveats else []
    if matched_fields:
        matched_core = [field for field in core_fields if field in matched_fields]
        if matched_core and status != "direct":
            summary_parts.append("Matching core fields: " + ", ".join(matched_core) + ".")

    return {
        "status": status,
        "title": title,
        "summary": " ".join(summary_parts).strip(),
        "caveats": caveats,
        "differing_fields": differing_fields,
        "matched_fields": matched_fields,
        "unknown_fields": unknown_fields,
        "field_values": field_values,
    }


def _coverage_ratio(entry: dict[str, Any]) -> float | None:
    classes = entry.get("num_classes")
    images = entry.get("num_images_evaluated")
    if not isinstance(classes, int) or not isinstance(images, int) or images <= 0:
        return None
    return classes / images


def _build_group_summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    sample = entries[0]
    tflite_top1 = [
        value
        for value in (_safe_float(_coerce_metrics(entry).get("tflite_top1_accuracy")) for entry in entries)
        if value is not None
    ]
    tflite_topk = [
        value
        for value in (_safe_float(_coerce_metrics(entry).get("tflite_topk_accuracy")) for entry in entries)
        if value is not None
    ]
    hog_top1 = [
        value
        for value in (_safe_float(_coerce_metrics(entry).get("hog_top1_accuracy")) for entry in entries)
        if value is not None
    ]
    hog_topk = [
        value
        for value in (_safe_float(_coerce_metrics(entry).get("hog_topk_accuracy")) for entry in entries)
        if value is not None
    ]
    agreement = [
        value
        for value in (_safe_float(_coerce_metrics(entry).get("model_agreement_rate")) for entry in entries)
        if value is not None
    ]
    stability = [
        value
        for value in (_safe_float(_coerce_metrics(entry).get("tflite_color_stability_rate")) for entry in entries)
        if value is not None
    ]
    coverage = [value for value in (_coverage_ratio(entry) for entry in entries) if value is not None]

    topk_range = _range(tflite_topk) or 0.0
    agreement_range = _range(agreement) or 0.0
    stability_range = _range(stability) or 0.0
    coverage_range = _range(coverage) or 0.0
    stability_score = len(entries) / (1.0 + topk_range + agreement_range + stability_range + coverage_range)
    if str(sample.get("sampling_mode")) == "balanced":
        stability_score += 0.5
    if sample.get("split") == "test":
        stability_score += 0.25
    if sample.get("max_images") == 200 and sample.get("max_images_per_class") == 1:
        stability_score += 0.5

    image_counts = [value for value in (entry.get("num_images_evaluated") for entry in entries) if isinstance(value, int)]
    class_counts = [value for value in (entry.get("num_classes") for entry in entries) if isinstance(value, int)]

    return {
        "dataset_path": sample.get("dataset_path"),
        "dataset_label": _dataset_label(sample.get("dataset_path")),
        "split": sample.get("split"),
        "sampling_mode": sample.get("sampling_mode"),
        "max_images": sample.get("max_images"),
        "max_images_per_class": sample.get("max_images_per_class"),
        "count": len(entries),
        "seed_values": sorted({entry.get("seed") for entry in entries if entry.get("seed") is not None}),
        "mean_tflite_top1_accuracy": _mean(tflite_top1),
        "mean_tflite_topk_accuracy": _mean(tflite_topk),
        "mean_hog_top1_accuracy": _mean(hog_top1),
        "mean_hog_topk_accuracy": _mean(hog_topk),
        "mean_agreement_rate": _mean(agreement),
        "mean_stability_rate": _mean(stability),
        "mean_class_coverage_ratio": _mean(coverage),
        "best_tflite_top1_accuracy": max(tflite_top1) if tflite_top1 else None,
        "tflite_topk_range": topk_range,
        "best_tflite_topk_accuracy": max(tflite_topk) if tflite_topk else None,
        "best_hog_top1_accuracy": max(hog_top1) if hog_top1 else None,
        "best_hog_topk_accuracy": max(hog_topk) if hog_topk else None,
        "best_agreement_rate": max(agreement) if agreement else None,
        "best_stability_rate": max(stability) if stability else None,
        "agreement_range": agreement_range,
        "stability_range": stability_range,
        "coverage_range": coverage_range,
        "num_images_evaluated_min": min(image_counts) if image_counts else None,
        "num_images_evaluated_max": max(image_counts) if image_counts else None,
        "num_classes_min": min(class_counts) if class_counts else None,
        "num_classes_max": max(class_counts) if class_counts else None,
        "stability_score": stability_score,
        "run_ids": [entry.get("run_id") for entry in entries],
    }


def _evidence_for_group(entries: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for entry in entries[:limit]:
        metrics = _coerce_metrics(entry)
        evidence.append(
            {
                "run_id": entry.get("run_id"),
                "created_at": entry.get("created_at"),
                "dataset_path": entry.get("dataset_path"),
                "dataset_label": _dataset_label(entry.get("dataset_path")),
                "sampling_mode": entry.get("sampling_mode"),
                "max_images": entry.get("max_images"),
                "max_images_per_class": entry.get("max_images_per_class"),
                "num_images_evaluated": entry.get("num_images_evaluated"),
                "num_classes": entry.get("num_classes"),
                "tflite_top1_accuracy": metrics.get("tflite_top1_accuracy"),
                "tflite_topk_accuracy": metrics.get("tflite_topk_accuracy"),
                "hog_top1_accuracy": metrics.get("hog_top1_accuracy"),
                "hog_topk_accuracy": metrics.get("hog_topk_accuracy"),
                "model_agreement_rate": metrics.get("model_agreement_rate"),
                "tflite_color_stability_rate": metrics.get("tflite_color_stability_rate"),
            }
        )
    return evidence


def _find_best_group(
    group_summaries: list[dict[str, Any]],
    *,
    dataset_path: str | None,
    split: str,
    sampling_mode: str,
    max_images: int,
    max_images_per_class: int | None,
) -> dict[str, Any] | None:
    normalized_dataset_path = _normalize_dataset_path(dataset_path)
    exact = [
        group
        for group in group_summaries
        if _normalize_dataset_path(group.get("dataset_path")) == normalized_dataset_path
        and group.get("split") == split
        and group.get("sampling_mode") == sampling_mode
        and group.get("max_images") == max_images
        and group.get("max_images_per_class") == max_images_per_class
    ]
    if exact:
        return sorted(exact, key=lambda item: (-float(item.get("stability_score", 0.0)), -int(item.get("count", 0))))[0]
    fallback = [
        group
        for group in group_summaries
        if _normalize_dataset_path(group.get("dataset_path")) == normalized_dataset_path
        and group.get("split") == split
        and group.get("sampling_mode") == sampling_mode
        and group.get("max_images") == max_images
    ]
    if fallback:
        return sorted(fallback, key=lambda item: (-float(item.get("stability_score", 0.0)), -int(item.get("count", 0))))[0]
    return None


def _history_table_rows(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        metrics = _coerce_metrics(entry)
        rows.append(
            {
                "run_id": entry.get("run_id"),
                "created_at": entry.get("created_at"),
                "split": entry.get("split"),
                "sampling_mode": entry.get("sampling_mode"),
                "max_images": entry.get("max_images"),
                "num_images_evaluated": entry.get("num_images_evaluated"),
                "num_classes": entry.get("num_classes"),
                "model_agreement_rate": metrics.get("model_agreement_rate"),
                "tflite_color_stability_rate": metrics.get("tflite_color_stability_rate"),
                "tflite_top1_accuracy": metrics.get("tflite_top1_accuracy"),
                "tflite_topk_accuracy": metrics.get("tflite_topk_accuracy"),
                "hog_top1_accuracy": metrics.get("hog_top1_accuracy"),
                "hog_topk_accuracy": metrics.get("hog_topk_accuracy"),
                "output_dir": entry.get("output_dir"),
            }
        )
    return rows


def _write_history_table(rows: list[dict[str, Any]]) -> str | None:
    if not rows:
        return None
    target = _history_table_path()
    with target.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return str(target)


def _write_history_metrics_chart(rows: list[dict[str, Any]]) -> str | None:
    if len(rows) < 2:
        return None
    target = _history_metrics_png_path()
    x_labels = make_unique_ablation_run_labels(rows)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=110)
    axes_flat = list(axes.flatten())
    metric_defs = [
        (("tflite_top1_accuracy", "hog_top1_accuracy"), "Top-1 accuracy by run"),
        (("tflite_topk_accuracy", "hog_topk_accuracy"), "Top-k accuracy by run"),
        (("model_agreement_rate",), "Model agreement by run"),
        (("tflite_color_stability_rate",), "TFLite color stability by run"),
    ]
    metric_labels = {
        "tflite_top1_accuracy": "TFLite top-1",
        "hog_top1_accuracy": "HOG top-1",
        "tflite_topk_accuracy": "TFLite top-k",
        "hog_topk_accuracy": "HOG top-k",
        "model_agreement_rate": "Agreement rate",
        "tflite_color_stability_rate": "TFLite color stability",
    }
    colors = ["#2E8B57", "#C46A1A"]
    for ax, (metric_keys, title) in zip(axes_flat, metric_defs):
        plotted_any = False
        for idx, metric_key in enumerate(metric_keys):
            values = [row.get(metric_key) for row in rows]
            plotted = [float(v) if isinstance(v, (int, float)) else float("nan") for v in values]
            if all(str(v) == "nan" for v in plotted):
                continue
            ax.plot(
                range(len(plotted)),
                plotted,
                marker="o",
                color=colors[idx % len(colors)],
                label=metric_labels.get(metric_key, metric_key),
            )
            plotted_any = True
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=25, ha="right", fontsize=9)
        ax.set_ylim(bottom=0.0)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Run label (MMDD_HHMM)", fontsize=10)
        ax.set_ylabel("Rate", fontsize=10)
        ax.tick_params(axis="y", labelsize=9)
        if plotted_any:
            ax.legend(fontsize=8)
        if not plotted_any:
            ax.axis("off")
    fig.suptitle("Ablation History Metrics", fontsize=14, y=0.98)
    fig.text(
        0.5,
        0.945,
        "Interpret direct metric comparisons only within matching dataset, split, sampling mode, "
        f"max_images, and per-class-cap groups. Total runs shown: {len(rows)}.",
        ha="center",
        va="top",
        fontsize=9,
    )
    fig.text(
        0.5,
        0.02,
        "Run labels use MMDD_HHMM from created_at/run_id/output_dir when available; full run IDs remain in tables and exports.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    fig.savefig(target, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(target)


def _history_recommendation_readiness(
    entries: list[dict[str, Any]],
    comparable_groups: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Summarize how much repeated directly comparable evidence exists.

    Thresholds are intentionally conservative:
    - 0 or 1 comparable runs in every group => insufficient history
    - 2 runs in the same comparable group => limited evidence
    - 3+ runs in the same comparable group => supported evidence
    """

    rationale: list[str] = []
    if comparable_groups is None:
        grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
        for entry in entries:
            grouped.setdefault(_group_key(entry), []).append(entry)
        comparable_groups = [
            _build_group_summary(group_entries)
            for group_entries in grouped.values()
            if group_entries
        ]

    repeated_groups = [
        group for group in comparable_groups if int(group.get("count", 0) or 0) >= 2
    ]
    strongest_repeated_group = None
    if repeated_groups:
        strongest_repeated_group = sorted(
            repeated_groups,
            key=lambda group: (-int(group.get("count", 0)), -float(group.get("stability_score", 0.0))),
        )[0]

    if not entries:
        rationale.append(
            "No completed ablation runs are present in recorded history, so recommendations must rely on safe defaults."
        )
        return {
            "enough_history": False,
            "recommendation_status": "insufficient_history",
            "rationale": rationale,
            "repeated_groups": repeated_groups,
            "strongest_repeated_group": strongest_repeated_group,
        }

    if not repeated_groups:
        rationale.append(
            "Recorded history includes runs, but none repeat within the same dataset, split, sampling mode, max_images, and max_images_per_class group."
        )
        rationale.append(
            "Recommendations should remain conservative because the available runs are one-off or not directly comparable."
        )
        return {
            "enough_history": False,
            "recommendation_status": "insufficient_history",
            "rationale": rationale,
            "repeated_groups": repeated_groups,
            "strongest_repeated_group": strongest_repeated_group,
        }

    strongest_count = int(strongest_repeated_group.get("count", 0) or 0) if strongest_repeated_group else 0
    strongest_dataset_unknown = (
        _normalize_dataset_path(strongest_repeated_group.get("dataset_path")) == "<unknown_dataset>"
        if strongest_repeated_group
        else False
    )

    if strongest_count >= 3 and not strongest_dataset_unknown:
        recommendation_status = "evidence_supported"
        rationale.append(
            "At least one directly comparable group has three or more recorded runs, which is enough for cautious evidence-supported recommendations."
        )
    else:
        recommendation_status = "evidence_limited"
        rationale.append(
            "At least one directly comparable group has repeated runs, but the evidence remains limited because support is only two runs or key grouping fields are missing."
        )

    rationale.append(
        "Recommendation confidence is based on repeated directly comparable groups rather than total global run count."
    )

    return {
        "enough_history": recommendation_status != "insufficient_history",
        "recommendation_status": recommendation_status,
        "rationale": rationale,
        "repeated_groups": repeated_groups,
        "strongest_repeated_group": strongest_repeated_group,
    }


def _format_rate(value: Any) -> str:
    numeric = _safe_float(value)
    return _message_rate(numeric)


def _format_cell(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    text = str(value).strip()
    return text or "n/a"


def _format_range(min_value: Any, max_value: Any) -> str:
    if min_value is None and max_value is None:
        return "n/a"
    if min_value == max_value:
        return _format_cell(min_value)
    return f"{_format_cell(min_value)} to {_format_cell(max_value)}"


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    if not rows:
        return ["No rows available."]
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(_format_cell(value) for value in row) + " |" for row in rows]
    return [header_line, separator, *body]


def _best_metric_rows(entries: list[dict[str, Any]]) -> list[list[Any]]:
    metric_defs = [
        ("TFLite top-1 accuracy", "tflite_top1_accuracy"),
        ("TFLite top-k accuracy", "tflite_topk_accuracy"),
        ("HOG top-1 accuracy", "hog_top1_accuracy"),
        ("HOG top-k accuracy", "hog_topk_accuracy"),
        ("Model agreement rate", "model_agreement_rate"),
        ("TFLite color stability", "tflite_color_stability_rate"),
    ]
    rows: list[list[Any]] = []
    for metric_label, metric_key in metric_defs:
        best_entry: dict[str, Any] | None = None
        best_value: float | None = None
        for entry in entries:
            value = _safe_float(_coerce_metrics(entry).get(metric_key))
            if value is None:
                continue
            if best_value is None or value > best_value:
                best_value = value
                best_entry = entry
        if best_entry is None or best_value is None:
            continue
        rows.append(
            [
                metric_label,
                _format_rate(best_value),
                best_entry.get("run_id"),
                _dataset_label(best_entry.get("dataset_path")),
                best_entry.get("split"),
                best_entry.get("sampling_mode"),
                best_entry.get("max_images"),
                best_entry.get("max_images_per_class"),
            ]
        )
    return rows


def _coverage_comparability_note(comparable_groups: list[dict[str, Any]]) -> str:
    balanced_coverage = [
        group.get("mean_class_coverage_ratio")
        for group in comparable_groups
        if group.get("sampling_mode") == "balanced"
    ]
    random_coverage = [
        group.get("mean_class_coverage_ratio")
        for group in comparable_groups
        if group.get("sampling_mode") == "random"
    ]
    if balanced_coverage and random_coverage:
        balanced_mean = _mean([float(v) for v in balanced_coverage if isinstance(v, (int, float))])
        random_mean = _mean([float(v) for v in random_coverage if isinstance(v, (int, float))])
        diff = abs(float(balanced_mean or 0.0) - float(random_mean or 0.0))
        if diff >= 0.15:
            return (
                "Balanced and random runs are not directly comparable here because their observed class coverage "
                "differs substantially."
            )
    return "Balanced and random runs only support direct comparison when dataset, split, sampling configuration, and class coverage align."


def _group_supporting_metrics(group: dict[str, Any] | None) -> dict[str, Any]:
    if not group:
        return {}
    return {
        "dataset_path": group.get("dataset_path"),
        "dataset_label": group.get("dataset_label") or _dataset_label(group.get("dataset_path")),
        "split": group.get("split"),
        "sampling_mode": group.get("sampling_mode"),
        "max_images": group.get("max_images"),
        "max_images_per_class": group.get("max_images_per_class"),
        "num_runs": group.get("count"),
        "seed_values": group.get("seed_values", []),
        "evaluated_image_range": _format_range(
            group.get("num_images_evaluated_min"),
            group.get("num_images_evaluated_max"),
        ),
        "class_count_range": _format_range(
            group.get("num_classes_min"),
            group.get("num_classes_max"),
        ),
        "best_tflite_top1": group.get("best_tflite_top1_accuracy"),
        "best_tflite_topk": group.get("best_tflite_topk_accuracy"),
        "best_hog_top1": group.get("best_hog_top1_accuracy"),
        "best_hog_topk": group.get("best_hog_topk_accuracy"),
        "best_agreement": group.get("best_agreement_rate"),
        "best_stability": group.get("best_stability_rate"),
        "stability_score": group.get("stability_score"),
    }


def _recommended_config(
    profile: str,
    fallback: dict[str, Any],
    group: dict[str, Any] | None,
) -> dict[str, Any]:
    config = dict(fallback)
    config["profile"] = profile
    if group:
        config.update(
            {
                "dataset_path": group.get("dataset_path", config.get("dataset_path")),
                "dataset_label": group.get("dataset_label") or _dataset_label(group.get("dataset_path")),
                "split": group.get("split", config.get("split")),
                "sampling_mode": group.get("sampling_mode", config.get("sampling_mode")),
                "max_images": group.get("max_images", config.get("max_images")),
                "max_images_per_class": group.get("max_images_per_class", config.get("max_images_per_class")),
            }
        )
    else:
        config["dataset_label"] = _dataset_label(config.get("dataset_path"))
    return config


def _recommendation_strength(group: dict[str, Any] | None) -> str:
    if not group:
        return "weak"
    count = int(group.get("count", 0) or 0)
    if count >= 3:
        return "strong"
    if count >= 2:
        return "moderate"
    return "weak"


def _build_history_markdown(
    entries: list[dict[str, Any]],
    comparable_groups: list[dict[str, Any]],
    corrupted_lines: list[dict[str, Any]],
    artifacts: dict[str, str | None],
    generated_at: str,
) -> str:
    if not entries:
        lines = [
            "# Ablation History Summary Report",
            "",
            "## Scope / Data Source",
            f"- Generated at: `{generated_at}`",
            "- Recorded runs: `0`",
            "- Comparable groups: `0`",
            "- Source of truth: `ablation_history.jsonl` remains the canonical append-only record.",
            "",
            "## Executive Summary",
            "No ablation runs have been recorded yet, so there is no evidence base for recommendations or comparisons.",
            "",
            "## Generated Artifacts",
        ]
        for label, path_text in artifacts.items():
            if path_text:
                lines.append(f"- `{Path(path_text).name}`")
        return "\n".join(lines) + "\n"

    readiness = _history_recommendation_readiness(entries, comparable_groups)
    enough_history = bool(readiness.get("enough_history"))
    readiness_rationale = list(readiness.get("rationale", []))
    best_rows = _best_metric_rows(entries)
    comparable_note = _coverage_comparability_note(comparable_groups)
    metrics_available = sum(1 for entry in entries if _coerce_metrics(entry))
    groups_with_repeats = list(readiness.get("repeated_groups", []))
    strongest_groups = sorted(
        groups_with_repeats,
        key=lambda group: (-int(group.get("count", 0)), -float(group.get("stability_score", 0.0))),
    )[:5]

    executive_points: list[str] = [
        f"- Recommendation-ready history: `{'yes' if enough_history else 'no'}`",
        f"- Runs with metric payloads: `{metrics_available}` of `{len(entries)}`",
    ]
    for row in best_rows:
        executive_points.append(f"- Best observed {row[0].lower()}: `{row[1]}` in `{row[2]}`")
    executive_points.append(
        "- Comparability note: results are only directly comparable within matching dataset, split, sampling mode, max_images, and max_images_per_class groups."
    )

    group_rows: list[list[Any]] = []
    for group in sorted(
        comparable_groups,
        key=lambda item: (-int(item.get("count", 0)), -float(item.get("stability_score", 0.0))),
    )[:10]:
        group_rows.append(
            [
                group.get("dataset_label") or _dataset_label(group.get("dataset_path")),
                group.get("split"),
                group.get("sampling_mode"),
                group.get("max_images"),
                group.get("max_images_per_class"),
                group.get("count"),
                _format_range(group.get("num_images_evaluated_min"), group.get("num_images_evaluated_max")),
                _format_range(group.get("num_classes_min"), group.get("num_classes_max")),
                _format_rate(group.get("best_tflite_top1_accuracy")),
                _format_rate(group.get("best_hog_top1_accuracy")),
                _format_rate(group.get("best_agreement_rate")),
                _format_rate(group.get("best_stability_rate")),
            ]
        )

    lines = [
        "# Ablation History Summary Report",
        "",
        "## Scope / Data Source",
        f"- Generated at: `{generated_at}`",
        f"- Recorded history records: `{len(entries)}`",
        f"- Comparable groups: `{len(comparable_groups)}`",
        f"- Corrupted history lines skipped: `{len(corrupted_lines)}`",
        "- Source of truth: `ablation_history.jsonl` remains the canonical append-only record.",
        "",
        "## Executive Summary",
        *executive_points,
        "",
        "## Best Observed Runs",
    ]
    lines.extend(
        _markdown_table(
            [
                "Metric",
                "Value",
                "Run ID",
                "Dataset",
                "Split",
                "Sampling",
                "Max Images",
                "Per-Class Cap",
            ],
            best_rows,
        )
    )
    lines.extend(
        [
            "",
            "## Comparable Group Summary",
            f"The table below shows up to the top `{min(len(comparable_groups), 10)}` comparable groups ranked by repeated evidence and stability. Use the CSV/JSON exports for the full history.",
        ]
    )
    lines.extend(
        _markdown_table(
            [
                "Dataset",
                "Split",
                "Sampling",
                "Max Images",
                "Per-Class Cap",
                "Runs",
                "Evaluated Images Range",
                "Class Count Range",
                "Best TFLite Top-1",
                "Best HOG Top-1",
                "Best Agreement",
                "Best Stability",
            ],
            group_rows,
        )
    )
    lines.extend(
        [
            "",
            "## Recommendation Readiness / Evidence",
            f"- Sufficient history for evidence-based recommendations: `{'yes' if enough_history else 'no'}`",
        ]
    )
    for item in readiness_rationale:
        lines.append(f"- {item}")
    if strongest_groups:
        lines.append("- Strongest repeated groups in recorded history:")
        for group in strongest_groups:
            lines.append(
                "  - "
                f"{group.get('dataset_label')} | split={_format_cell(group.get('split'))} | "
                f"sampling={_format_cell(group.get('sampling_mode'))} | max_images={_format_cell(group.get('max_images'))} | "
                f"per_class={_format_cell(group.get('max_images_per_class'))} | runs={_format_cell(group.get('count'))}"
            )
    else:
        lines.append("- No group currently has repeated evidence strong enough to stand out beyond single observations.")
    lines.extend(
        [
            "",
            "## Caveats and Interpretation Notes",
            "- Ablations are inference-only; no model retraining is performed.",
            "- Cross-dataset or cross-sampling comparisons are exploratory rather than directly comparable.",
            "- Accuracy metrics are only meaningful when manifest labels are available for the evaluated images.",
            "- Sample-based runs are not full-dataset evaluations unless a full test split was explicitly used.",
            f"- {comparable_note}",
        ]
    )
    if any(_normalize_dataset_path(entry.get("dataset_path")) == "<unknown_dataset>" for entry in entries):
        lines.append("- Some older history records are missing dataset_path values, which reduces comparability confidence.")
    if corrupted_lines:
        lines.append(f"- `{len(corrupted_lines)}` corrupted JSONL line(s) were skipped during export.")
    lines.extend(["", "## Generated Artifacts"])
    for label, path_text in artifacts.items():
        if path_text:
            lines.append(f"- `{Path(path_text).name}`")
    return "\n".join(lines) + "\n"


def _write_history_summary(entries: list[dict[str, Any]], comparable_groups: list[dict[str, Any]], corrupted_lines: list[dict[str, Any]]) -> dict[str, str | None]:
    summary_json_path = _history_summary_json_path()
    summary_md_path = _history_summary_md_path()
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = {
        "generated_at": generated_at,
        "count": len(entries),
        "comparable_groups": comparable_groups,
        "corrupted_lines": corrupted_lines,
    }
    summary_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    artifacts = {
        "history_jsonl": str(_history_path()),
        "table_csv": str(_history_table_path()) if _history_table_path().is_file() else None,
        "summary_json": str(summary_json_path),
        "summary_markdown": str(summary_md_path),
        "metrics_png": str(_history_metrics_png_path()) if _history_metrics_png_path().is_file() else None,
    }
    summary_md_path.write_text(
        _build_history_markdown(entries, comparable_groups, corrupted_lines, artifacts, generated_at),
        encoding="utf-8",
    )
    return {
        "summary_json": str(summary_json_path),
        "summary_markdown": str(summary_md_path),
    }


def _history_artifact_targets() -> dict[str, str]:
    return {
        "history_jsonl": str(_history_path()),
        "table_csv": str(_history_table_path()),
        "summary_json": str(_history_summary_json_path()),
        "summary_markdown": str(_history_summary_md_path()),
        "metrics_png": str(_history_metrics_png_path()),
    }


def _safe_read_jsonl_line(line: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        data = json.loads(line)
        if not isinstance(data, dict):
            return None, "Line is not a JSON object."
        return data, None
    except Exception as exc:
        return None, str(exc)


def append_ablation_history(summary: dict[str, Any]) -> dict[str, Any]:
    history_path = _history_path()
    entry = dict(summary)
    entry.setdefault("run_id", _coerce_run_id())
    entry.setdefault("created_at", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    with history_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=True) + "\n")
    export_result = export_ablation_history_artifacts()
    return {
        "success": True,
        "history_path": str(history_path),
        "run_id": entry["run_id"],
        "artifacts": export_result.get("artifacts", {}),
    }


def load_ablation_history(limit: int | None = None) -> dict[str, Any]:
    history_path = _history_path()
    if not history_path.is_file():
        return {
            "success": True,
            "history_path": str(history_path),
            "entries": [],
            "count": 0,
            "corrupted_lines": [],
            "error": None,
        }

    entries: list[dict[str, Any]] = []
    corrupted_lines: list[dict[str, Any]] = []
    with history_path.open("r", encoding="utf-8") as fh:
        for idx, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            data, error = _safe_read_jsonl_line(line)
            if data is None:
                corrupted_lines.append({"line_number": idx, "error": error})
                continue
            entries.append(data)
    if limit is not None:
        entries = entries[-max(int(limit), 0):]
    return {
        "success": True,
        "history_path": str(history_path),
        "entries": entries,
        "count": len(entries),
        "corrupted_lines": corrupted_lines,
        "error": None,
    }


def _build_ablation_history_summary_payload(
    entries: list[dict[str, Any]],
    corrupted_lines: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for entry in entries:
        grouped.setdefault(_group_key(entry), []).append(entry)
    comparable_groups = [
        _build_group_summary(group_entries)
        for _, group_entries in sorted(grouped.items(), key=lambda item: item[0])
        if group_entries
    ]
    artifact_targets = _history_artifact_targets()

    if not entries:
        return {
            "success": True,
            "count": 0,
            "datasets": [],
            "sampling_modes": {},
            "latest_runs": [],
            "comparable_groups": comparable_groups,
            "corrupted_lines": corrupted_lines,
            "history_files": artifact_targets,
            "error": None,
        }

    datasets = Counter(str(entry.get("dataset_path", "")) for entry in entries if entry.get("dataset_path"))
    sampling_modes = Counter(str(entry.get("sampling_mode", "")) for entry in entries if entry.get("sampling_mode"))
    metrics_available = sum(1 for entry in entries if _coerce_metrics(entry))
    latest_runs = entries[-10:]
    return {
        "success": True,
        "count": len(entries),
        "datasets": [{"dataset_path": key, "count": value} for key, value in datasets.most_common()],
        "sampling_modes": dict(sampling_modes),
        "metrics_available_runs": metrics_available,
        "latest_runs": latest_runs,
        "comparable_groups": comparable_groups,
        "corrupted_lines": corrupted_lines,
        "history_files": artifact_targets,
        "error": None,
    }


def summarize_ablation_history() -> dict[str, Any]:
    loaded = load_ablation_history(limit=None)
    entries = loaded.get("entries", [])
    corrupted_lines = loaded.get("corrupted_lines", [])
    return _build_ablation_history_summary_payload(entries, corrupted_lines)


def export_ablation_history_artifacts() -> dict[str, Any]:
    loaded = load_ablation_history(limit=None)
    entries = loaded.get("entries", [])
    corrupted_lines = loaded.get("corrupted_lines", [])
    summary_payload = _build_ablation_history_summary_payload(entries, corrupted_lines)
    comparable_groups = summary_payload.get("comparable_groups", [])
    rows = _history_table_rows(entries)
    artifacts = {
        "history_jsonl": str(_history_path()),
        "table_csv": _write_history_table(rows),
        "summary_json": None,
        "summary_markdown": None,
        "metrics_png": _write_history_metrics_chart(rows),
    }
    summary_artifacts = _write_history_summary(entries, comparable_groups, corrupted_lines)
    artifacts.update(summary_artifacts)
    return {
        "success": True,
        "artifacts": artifacts,
        "count": len(entries),
        "corrupted_lines": corrupted_lines,
        "error": None,
    }


def get_ablation_run(run_id: str | None = None, latest: bool = False) -> dict[str, Any]:
    loaded = load_ablation_history(limit=None)
    entries = loaded.get("entries", [])
    if not entries:
        return {
            "success": False,
            "run": None,
            "error": "No ablation history exists yet.",
        }
    if latest or not run_id:
        entry = entries[-1]
    else:
        entry = next((item for item in entries if str(item.get("run_id")) == str(run_id)), None)
        if entry is None:
            return {
                "success": False,
                "run": None,
                "error": f"Run ID not found: {run_id}",
            }
    return {
        "success": True,
        "run": entry,
        "error": None,
    }


def compare_ablation_runs(run_ids: list[str] | None = None) -> dict[str, Any]:
    loaded = load_ablation_history(limit=None)
    entries = loaded.get("entries", [])
    if not entries:
        return {
            "success": False,
            "runs": [],
            "comparison_rows": [],
            "selected_run_ids": [],
            "requested_run_ids": [],
            "matched_run_ids": [],
            "missing_run_ids": [],
            "comparison_warnings": [],
            "explanation": "No ablation history exists yet.",
            "error": "No ablation history exists yet.",
        }

    selected_entries: list[dict[str, Any]]
    explanation = ""
    requested_run_ids: list[str] = []
    matched_run_ids: list[str] = []
    missing_run_ids: list[str] = []
    comparison_warnings: list[str] = []
    if run_ids:
        requested_run_ids = [str(run_id) for run_id in run_ids]
        entries_by_run_id = {str(entry.get("run_id")): entry for entry in entries}
        seen_requested: set[str] = set()
        deduplicated_requested_run_ids: list[str] = []
        duplicate_run_ids: list[str] = []
        for run_id in requested_run_ids:
            if run_id in seen_requested:
                duplicate_run_ids.append(run_id)
                continue
            seen_requested.add(run_id)
            deduplicated_requested_run_ids.append(run_id)

        selected_entries = []
        for run_id in deduplicated_requested_run_ids:
            entry = entries_by_run_id.get(run_id)
            if entry is None:
                missing_run_ids.append(run_id)
                continue
            selected_entries.append(entry)
            matched_run_ids.append(run_id)

        if duplicate_run_ids:
            duplicate_text = ", ".join(duplicate_run_ids)
            comparison_warnings.append(
                f"Duplicate requested run IDs were ignored: {duplicate_text}."
            )

        if not selected_entries:
            return {
                "success": False,
                "runs": [],
                "comparison_rows": [],
                "selected_run_ids": [],
                "requested_run_ids": requested_run_ids,
                "matched_run_ids": matched_run_ids,
                "missing_run_ids": missing_run_ids,
                "comparison_warnings": comparison_warnings,
                "explanation": "None of the requested run IDs were found in ablation history.",
                "error": "Requested run IDs not found.",
            }
        if len(selected_entries) < 2:
            if missing_run_ids:
                comparison_warnings.append(
                    "Some requested run IDs were not found and were omitted from comparison: "
                    + ", ".join(missing_run_ids)
                    + "."
                )
            return {
                "success": False,
                "runs": selected_entries,
                "comparison_rows": [],
                "selected_run_ids": matched_run_ids,
                "requested_run_ids": requested_run_ids,
                "matched_run_ids": matched_run_ids,
                "missing_run_ids": missing_run_ids,
                "comparison_warnings": comparison_warnings,
                "explanation": "At least two valid requested runs are required for an explicit comparison.",
                "error": "At least two valid requested runs are required.",
            }
        explanation = "Compared the explicitly requested ablation runs."
        if missing_run_ids:
            comparison_warnings.append(
                "Some requested run IDs were not found and were omitted from comparison: "
                + ", ".join(missing_run_ids)
                + "."
            )
    else:
        grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
        for entry in entries:
            grouped.setdefault(_group_key(entry), []).append(entry)
        comparable_groups = [group_entries for group_entries in grouped.values() if len(group_entries) >= 2]
        if comparable_groups:
            best_group = sorted(
                comparable_groups,
                key=lambda group_entries: (-len(group_entries), str(group_entries[-1].get("created_at", ""))),
            )[0]
            selected_entries = best_group[-min(3, len(best_group)) :]
            sample_entry = best_group[-1] if best_group else {}
            explanation = (
                "Compared the most recent comparable runs from the same dataset, split, sampling mode, "
                f"max_images, and max_images_per_class group ({_dataset_label(sample_entry.get('dataset_path'))})."
            )
        else:
            selected_entries = entries[-min(3, len(entries)) :]
            explanation = "No multi-run comparable group existed, so the most recent runs were compared instead."
        matched_run_ids = [str(entry.get("run_id")) for entry in selected_entries]

    comparison_rows: list[dict[str, Any]] = []
    for entry in selected_entries:
        metrics = _coerce_metrics(entry)
        comparison_rows.append(
            {
                "run_id": entry.get("run_id"),
                "created_at": entry.get("created_at"),
                "split": entry.get("split"),
                "sampling_mode": entry.get("sampling_mode"),
                "max_images": entry.get("max_images"),
                "max_images_per_class": entry.get("max_images_per_class"),
                "num_images_evaluated": entry.get("num_images_evaluated"),
                "num_classes": entry.get("num_classes"),
                "model_agreement_rate": metrics.get("model_agreement_rate"),
                "tflite_color_stability_rate": metrics.get("tflite_color_stability_rate"),
                "tflite_top1_accuracy": metrics.get("tflite_top1_accuracy"),
                "tflite_topk_accuracy": metrics.get("tflite_topk_accuracy"),
                "hog_top1_accuracy": metrics.get("hog_top1_accuracy"),
                "hog_topk_accuracy": metrics.get("hog_topk_accuracy"),
                "output_dir": entry.get("output_dir"),
            }
        )
    validity = describe_comparison_validity(selected_entries)
    return {
        "success": True,
        "runs": selected_entries,
        "comparison_rows": comparison_rows,
        "selected_run_ids": [str(entry.get("run_id")) for entry in selected_entries],
        "requested_run_ids": requested_run_ids,
        "matched_run_ids": matched_run_ids or [str(entry.get("run_id")) for entry in selected_entries],
        "missing_run_ids": missing_run_ids,
        "comparison_warnings": comparison_warnings,
        "explanation": explanation,
        "comparison_validity": validity,
        "caveats": validity.get("caveats", []) + comparison_warnings,
        "error": None,
    }


def get_recommendations_from_history() -> dict[str, Any]:
    loaded = load_ablation_history(limit=None)
    entries = loaded.get("entries", [])
    constraints = [
        "No retraining is supported here; only inference-time and post-training ablations are allowed.",
        "HOG decision scores are not probabilities.",
        "Full-dataset runs may be slow.",
        "Accuracy is only meaningful when labels are available.",
        "Balanced sampling is preferred for small samples.",
    ]
    safe_defaults = {
        "split": "test",
        "seed": 42,
        "max_images_quick": 50,
        "max_images_stronger": 200,
        "max_images_per_class": 1,
        "tflite_color_correct_values": ["none", "gray_world", "max_rgb"],
        "tflite_top_k_values": [1, 3, 5],
        "hog_top_k_values": [1, 3, 5],
    }
    history_summary = summarize_ablation_history()
    comparable_groups = history_summary.get("comparable_groups", []) if isinstance(history_summary, dict) else []
    readiness = _history_recommendation_readiness(entries, comparable_groups)
    enough_history = bool(readiness.get("enough_history"))
    recommendation_status = str(readiness.get("recommendation_status") or "insufficient_history")
    rationale: list[str] = list(readiness.get("rationale", []))

    recommendations: dict[str, Any] = {}
    caveats: list[str] = []
    evidence: dict[str, Any] = {}
    next_best_actions: list[str] = [
        "Run a balanced 50-image quick check on the test split when you need a fast sanity check.",
        "Run a balanced 200-image test ablation with one image per class for a stronger representative study.",
        "Repeat the same balanced configuration with a different seed to probe stability before drawing stronger conclusions.",
        "Avoid treating random and balanced runs as direct equivalents unless class coverage is closely aligned.",
    ]

    if enough_history:
        rationale.append("Recommendations are based on the most stable comparable run groups rather than the latest run.")
    else:
        rationale.append(
            "There is not enough repeated directly comparable history yet for performance-based recommendations, so only constraints and safe defaults are shown."
        )

    if entries:
        dataset_counts = Counter(str(entry.get("dataset_path", "")) for entry in entries if entry.get("dataset_path"))
        preferred_dataset = dataset_counts.most_common(1)[0][0] if dataset_counts else None
    else:
        preferred_dataset = None

    quick_group = _find_best_group(
        comparable_groups,
        dataset_path=preferred_dataset,
        split="test",
        sampling_mode="balanced",
        max_images=50,
        max_images_per_class=1,
    )
    robust_group = _find_best_group(
        comparable_groups,
        dataset_path=preferred_dataset,
        split="test",
        sampling_mode="balanced",
        max_images=200,
        max_images_per_class=1,
    )
    exploratory_group = _find_best_group(
        comparable_groups,
        dataset_path=preferred_dataset,
        split="test",
        sampling_mode="random",
        max_images=200,
        max_images_per_class=None,
    )

    quick_default = {
        "dataset_path": preferred_dataset,
        "split": "test",
        "sampling_mode": "balanced",
        "seed": 42,
        "max_images": 50,
        "max_images_per_class": 1,
    }
    robust_default = {
        "dataset_path": preferred_dataset,
        "split": "test",
        "sampling_mode": "balanced",
        "seed": 42,
        "max_images": 200,
        "max_images_per_class": 1,
    }
    exploratory_default = {
        "dataset_path": preferred_dataset,
        "split": "test",
        "sampling_mode": "random",
        "seed": 42,
        "max_images": 200,
        "max_images_per_class": None,
    }
    recommendations["recommended_quick_run"] = _recommended_config("quick", quick_default, quick_group)
    recommendations["recommended_robust_run"] = _recommended_config("robust", robust_default, robust_group)
    recommendations["exploratory_run_option"] = _recommended_config("exploratory", exploratory_default, exploratory_group)
    recommendations["tflite_color_correct_values"] = safe_defaults["tflite_color_correct_values"]
    recommendations["tflite_top_k_values"] = safe_defaults["tflite_top_k_values"]
    recommendations["hog_top_k_values"] = safe_defaults["hog_top_k_values"]

    grouped_entries: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for entry in entries:
        grouped_entries.setdefault(_group_key(entry), []).append(entry)

    if quick_group:
        evidence["recommended_quick_run"] = {
            "group_summary": quick_group,
            "example_runs": _evidence_for_group(grouped_entries[_group_key(quick_group)]),
            "supporting_metrics": _group_supporting_metrics(quick_group),
        }
        rationale.append("A balanced 50-image test run remains the preferred quick-check configuration.")
    else:
        rationale.append("No prior balanced 50-image test group exists yet, so the quick recommendation remains a safe default.")

    if robust_group:
        evidence["recommended_robust_run"] = {
            "group_summary": robust_group,
            "example_runs": _evidence_for_group(grouped_entries[_group_key(robust_group)]),
            "supporting_metrics": _group_supporting_metrics(robust_group),
        }
        rationale.append("A balanced 200-image test run with one image per class is the preferred representative study when enough time is available.")
    else:
        rationale.append("No prior balanced 200-image test group exists yet, so the robust recommendation remains a safe default.")

    if exploratory_group:
        evidence["exploratory_run_option"] = {
            "group_summary": exploratory_group,
            "example_runs": _evidence_for_group(grouped_entries[_group_key(exploratory_group)]),
            "supporting_metrics": _group_supporting_metrics(exploratory_group),
        }
        caveats.append("Random sampling can be useful for exploratory performance estimates, but it should be interpreted separately from balanced small-sample studies.")

    if quick_group and robust_group and float(robust_group.get("stability_score", 0.0)) >= float(quick_group.get("stability_score", 0.0)):
        rationale.append("Among balanced groups, the 200-image test configuration is currently the most stable comparable basis for representative recommendations.")

    balanced_200 = robust_group
    random_200 = exploratory_group
    balanced_coverage = _safe_float(balanced_200.get("mean_class_coverage_ratio")) if balanced_200 else None
    random_coverage = _safe_float(random_200.get("mean_class_coverage_ratio")) if random_200 else None
    if balanced_coverage is not None and random_coverage is not None and abs(balanced_coverage - random_coverage) >= 0.15:
        caveats.append(
            "Balanced and random 200-image runs showed substantially different class coverage, so they are not directly comparable as like-for-like representative studies."
        )

    dataset_count = len(
        {
            _normalize_dataset_path(entry.get("dataset_path"))
            for entry in entries
            if _normalize_dataset_path(entry.get("dataset_path")) != "<unknown_dataset>"
        }
    )
    if dataset_count > 1:
        caveats.append("History contains multiple datasets, so only dataset-matched groups were considered directly comparable.")
    if any(_normalize_dataset_path(entry.get("dataset_path")) == "<unknown_dataset>" for entry in entries):
        caveats.append("Some older history records are missing dataset_path values, which reduces comparability confidence.")

    caveats.extend(
        [
            "Accuracy metrics are only meaningful when manifest labels are available.",
            "Ablations are inference-only; no retraining is performed.",
        ]
    )

    strongest_group = readiness.get("strongest_repeated_group")

    strongest_group_summary = _group_supporting_metrics(strongest_group)
    strongest_group_count = int(strongest_group.get("count", 0) or 0) if strongest_group else 0
    if not enough_history:
        if entries:
            evidence_summary = (
                "Recorded runs exist, but none repeat within the same comparable configuration yet, so recommendations remain conservative and default-driven."
            )
        else:
            evidence_summary = "No recorded history exists yet, so recommendations remain conservative and default-driven."
    elif strongest_group:
        evidence_prefix = "Strongest repeated comparable group"
        if recommendation_status == "evidence_limited":
            evidence_prefix = "Limited repeated comparable evidence"
        evidence_summary = (
            f"{evidence_prefix} has {strongest_group_count} run(s) on dataset "
            f"{_dataset_label(strongest_group.get('dataset_path'))} with split={strongest_group.get('split')}, "
            f"sampling={strongest_group.get('sampling_mode')}, max_images={strongest_group.get('max_images')}, "
            f"and max_images_per_class={strongest_group.get('max_images_per_class')}."
        )
    else:
        evidence_summary = "History contains some repeated evidence, but no repeated comparable group stands out strongly enough to summarize cleanly yet."

    recommended_profile = "robust"
    recommended_config = recommendations["recommended_robust_run"]
    supporting_metrics = _group_supporting_metrics(robust_group)
    if not robust_group and quick_group:
        recommended_profile = "quick"
        recommended_config = recommendations["recommended_quick_run"]
        supporting_metrics = _group_supporting_metrics(quick_group)
    elif not robust_group and not quick_group and exploratory_group:
        recommended_profile = "exploratory"
        recommended_config = recommendations["exploratory_run_option"]
        supporting_metrics = _group_supporting_metrics(exploratory_group)
    elif not robust_group and not quick_group:
        supporting_metrics = {}

    return {
        "success": True,
        "enough_history": enough_history,
        "recommendation_status": recommendation_status,
        "evidence_summary": evidence_summary,
        "recommended_profile": recommended_profile,
        "recommended_config": recommended_config,
        "supporting_metrics": supporting_metrics,
        "strongest_comparable_group": strongest_group_summary,
        "evidence_strength": _recommendation_strength(strongest_group),
        "next_best_actions": next_best_actions,
        "recommendations": recommendations,
        "constraints": constraints,
        "caveats": caveats,
        "evidence": evidence,
        "rationale": rationale,
        "history_summary": history_summary,
        "error": None,
    }
