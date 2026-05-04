#!/usr/bin/env python3
"""
Pure planning/configuration helpers for inference-only one-factor parameter sweeps.
"""
from __future__ import annotations

import re
from typing import Any

from agent.ablation_planner import SAFE_DEFAULTS

SUPPORTED_SWEEP_PARAMETERS = (
    "top_k",
    "sampling_mode",
    "max_images",
    "max_images_per_class",
    "seed",
)

SUPPORTED_SWEEP_METRICS = (
    "tflite_top1_accuracy",
    "tflite_topk_accuracy",
    "hog_top1_accuracy",
    "hog_topk_accuracy",
    "model_agreement_rate",
)

DEFAULT_SELECTED_METRICS = (
    "tflite_top1_accuracy",
    "hog_top1_accuracy",
)

DEFAULT_BASELINE_VALUES = {
    "top_k": 5,
    "sampling_mode": SAFE_DEFAULTS["sampling_mode"],
    "max_images": SAFE_DEFAULTS["max_images_robust"],
    "max_images_per_class": SAFE_DEFAULTS["max_images_per_class"],
    "seed": SAFE_DEFAULTS["seed"],
}

DEFAULT_SWEEP_LIMITS = {
    "max_supported_parameters": 5,
    "max_values_per_parameter": 5,
    "max_total_runs": 25,
    "max_top_k": 10,
    "max_images": 1000,
    "max_images_per_class": 50,
}

_PARAMETER_ALIASES = {
    "top_k": ("top_k", "top k", "top-k", "k"),
    "sampling_mode": (
        "sampling_mode",
        "sampling mode",
        "sampleng_mode",
        "sampleng mode",
        "sampling",
        "sampling method",
        "balanced sampling",
        "random sampling",
        "sorted sampling",
    ),
    "max_images": ("max_images", "max images", "maximum images", "image count", "sample size"),
    "max_images_per_class": (
        "max_images_per_class",
        "max images per class",
        "maximum images per class",
        "images per class",
        "per class",
    ),
    "seed": ("seed", "seeds", "random seed", "random seeds"),
}

_METRIC_ALIASES = {
    "tflite_top1_accuracy": (
        "tflite top 1 accuracy",
        "tflite top-1 accuracy",
        "tflite top1 accuracy",
        "tflite top 1",
        "tflite top-1",
        "tflite top1",
    ),
    "hog_top1_accuracy": (
        "hog svm top 1 accuracy",
        "hog+svm top-1 accuracy",
        "hog svm top-1 accuracy",
        "hog svm top1 accuracy",
        "hog top 1 accuracy",
        "hog top-1 accuracy",
        "hog top1 accuracy",
        "hog top 1",
        "hog top-1",
        "hog top1",
    ),
    "tflite_topk_accuracy": (
        "tflite top k accuracy",
        "tflite top-k accuracy",
        "tflite top k",
        "tflite top-k",
    ),
    "hog_topk_accuracy": (
        "hog svm top k accuracy",
        "hog+svm top-k accuracy",
        "hog svm top-k accuracy",
        "hog top k accuracy",
        "hog top-k accuracy",
        "hog top k",
        "hog top-k",
    ),
    "model_agreement_rate": (
        "model agreement rate",
        "agreement rate",
        "agrrement rate",
        "model agreement",
        "agreement",
    ),
}

_SECTION_ALIASES = {
    "baseline": (
        "baseline parameters",
        "base parameters",
        "base paramateres",
        "baseline",
        "base",
        "static values",
        "fixed values",
        "default values",
    ),
    "sweep": (
        "values to sweep",
        "parameter ranges",
        "sweeps",
        "sweep",
        "ranges",
        "vary",
    ),
    "metrics": (
        "metrics to plot are",
        "metrics to plot",
        "plot metrics",
        "selected metrics",
        "graph metrics",
        "metrics",
    ),
}

_TEXT_NORMALIZATION_REPLACEMENTS = (
    (r"\bparamateres\b", "parameters"),
    (r"\bsampleng_mode\b", "sampling_mode"),
    (r"\bsampleng mode\b", "sampling_mode"),
    (r"\bagrrement\b", "agreement"),
    (r"\baccurracy\b", "accuracy"),
)


def _normalize_parameter_name(name: Any) -> str:
    text = _normalize_request_text(str(name or "").strip().lower())
    normalized = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    for canonical_name, aliases in _PARAMETER_ALIASES.items():
        if normalized == canonical_name:
            return canonical_name
        normalized_aliases = {
            re.sub(r"[^a-z0-9]+", "_", alias).strip("_")
            for alias in aliases
        }
        if normalized in normalized_aliases:
            return canonical_name
    return normalized


def _listify_values(raw_values: Any) -> list[Any]:
    if raw_values is None:
        return []
    if isinstance(raw_values, (list, tuple)):
        return list(raw_values)
    if isinstance(raw_values, set):
        return list(raw_values)
    if isinstance(raw_values, str):
        text = raw_values.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        parts = [part.strip() for part in text.split(",")]
        return [part for part in parts if part]
    return [raw_values]


def _dedupe_preserve_order(values: list[Any]) -> list[Any]:
    seen: set[tuple[str, str]] = set()
    deduped: list[Any] = []
    for value in values:
        key = (type(value).__name__, str(value))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def _normalize_request_text(text: str) -> str:
    normalized = str(text or "").strip().lower()
    normalized = normalized.replace("+", " ")
    normalized = normalized.replace("/", " ")
    normalized = normalized.replace("-", " ")
    for pattern, replacement in _TEXT_NORMALIZATION_REPLACEMENTS:
        normalized = re.sub(pattern, replacement, normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _metric_alias_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for canonical_name, aliases in _METRIC_ALIASES.items():
        for alias in aliases:
            lookup[_normalize_request_text(alias)] = canonical_name
    return lookup


def _parameter_alias_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for canonical_name, aliases in _PARAMETER_ALIASES.items():
        for alias in aliases:
            lookup[_normalize_request_text(alias)] = canonical_name
    return lookup


def _sorted_aliases(values: dict[str, str]) -> list[str]:
    return sorted(values.keys(), key=len, reverse=True)


def _section_span(text: str, section_name: str) -> tuple[int, int] | None:
    candidates = sorted(_SECTION_ALIASES.get(section_name, ()), key=len, reverse=True)
    for alias in candidates:
        pattern = rf"\b{re.escape(_normalize_request_text(alias))}\b\s*[:\-]?"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.start(), match.end()
    return None


def _slice_sections(text: str) -> dict[str, str]:
    spans: list[tuple[str, int, int]] = []
    for section_name in _SECTION_ALIASES:
        span = _section_span(text, section_name)
        if span is not None:
            spans.append((section_name, span[0], span[1]))
    spans.sort(key=lambda item: item[1])

    sections: dict[str, str] = {}
    for index, (section_name, _start, content_start) in enumerate(spans):
        content_end = spans[index + 1][1] if index + 1 < len(spans) else len(text)
        sections[section_name] = text[content_start:content_end].strip(" .,:;-")

    if "sweep" not in sections:
        metric_section_start = next((start for name, start, _end in spans if name == "metrics"), len(text))
        baseline_section_start = next((start for name, start, _end in spans if name == "baseline"), metric_section_start)
        sections["sweep"] = text[: min(metric_section_start, baseline_section_start)].strip(" .,:;-")
    return sections


def _parameter_match_pattern() -> re.Pattern[str]:
    alias_lookup = _parameter_alias_lookup()
    aliases = _sorted_aliases(alias_lookup)
    joined = "|".join(re.escape(alias) for alias in aliases)
    return re.compile(
        rf"(?P<label>\b(?:{joined})\b)\s*(?:[:=]|is\b|are\b|-)?\s*",
        flags=re.IGNORECASE,
    )


def _split_parameter_spans(block_text: str) -> list[tuple[str, str]]:
    normalized = _normalize_request_text(block_text)
    if not normalized:
        return []
    pattern = _parameter_match_pattern()
    matches = list(pattern.finditer(normalized))
    if not matches:
        return []
    alias_lookup = _parameter_alias_lookup()
    spans: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        label = _normalize_request_text(match.group("label"))
        canonical_name = alias_lookup.get(label)
        if not canonical_name:
            continue
        value_start = match.end()
        value_end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
        value_text = normalized[value_start:value_end].strip(" ,.;:-")
        if value_text:
            spans.append((canonical_name, value_text))
    return spans


def _parse_value_list(value_text: str) -> list[str]:
    text = str(value_text or "").strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    parts = [part.strip(" ,.;:") for part in text.split(",")]
    cleaned = [part for part in parts if part]
    normalized_values: list[str] = []
    for value in cleaned:
        normalized_value = _normalize_request_text(value)
        normalized_value = re.sub(r"^(and|use|using|with)\b", "", normalized_value).strip()
        normalized_value = re.sub(r"\b(and|use|using|with|values|value|parameter|parameters)$", "", normalized_value).strip()
        if " " in normalized_value:
            normalized_value = normalized_value.split()[0].strip()
        normalized_value = normalized_value.strip(" .,:;-")
        if normalized_value == "balance":
            normalized_value = "balanced"
        if normalized_value:
            normalized_values.append(normalized_value)
    return normalized_values


def _parse_baseline_values(block_text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for parameter_name, value_text in _split_parameter_spans(block_text):
        parsed_values = _parse_value_list(value_text)
        if parsed_values:
            values[parameter_name] = parsed_values[0]
    return values


def _parse_parameter_ranges(block_text: str) -> dict[str, list[str]]:
    ranges: dict[str, list[str]] = {}
    for parameter_name, value_text in _split_parameter_spans(block_text):
        parsed_values = _parse_value_list(value_text)
        if parsed_values:
            ranges[parameter_name] = parsed_values
    return ranges


def _extract_metric_section_text(sections: dict[str, str], normalized_text: str) -> str:
    metrics_text = sections.get("metrics", "")
    if metrics_text:
        return metrics_text
    metric_span = _section_span(normalized_text, "metrics")
    if metric_span is None:
        return ""
    return normalized_text[metric_span[1]:].strip(" .,:;-")


def _normalize_metric_token(token: str) -> str | None:
    normalized = _normalize_request_text(token)
    normalized = re.sub(
        r"^(metrics?\s+to\s+plot\s+are|metrics?\s+to\s+plot|plot\s+metrics|selected\s+metrics|graph\s+metrics)\b",
        "",
        normalized,
    ).strip(" ,.;:")
    if not normalized:
        return None
    lookup = _metric_alias_lookup()
    if normalized in lookup:
        return lookup[normalized]
    for alias, canonical_name in sorted(lookup.items(), key=lambda item: len(item[0]), reverse=True):
        if normalized == alias:
            return canonical_name
    return None


def _parse_selected_metrics(block_text: str) -> list[str]:
    if not str(block_text or "").strip():
        return []
    metrics: list[str] = []
    for token in re.split(r",|;", block_text):
        canonical_name = _normalize_metric_token(token)
        if canonical_name:
            metrics.append(canonical_name)
    if metrics:
        return _dedupe_preserve_order(metrics)
    normalized_block = _normalize_request_text(block_text)
    lookup = _metric_alias_lookup()
    matches: list[tuple[int, str]] = []
    for alias, canonical_name in lookup.items():
        for match in re.finditer(rf"(?<!\w){re.escape(alias)}(?!\w)", normalized_block):
            matches.append((match.start(), canonical_name))
    if matches:
        matches.sort(key=lambda item: item[0])
        metrics.extend(canonical_name for _, canonical_name in matches)
    return _dedupe_preserve_order(metrics)


def _extract_dataset_path(raw_text: str) -> str | None:
    match = re.search(r"(?im)^\s*dataset\s+path\s*:\s*(.+?)\s*$", str(raw_text or ""))
    if not match:
        return None
    dataset_path = match.group(1).strip().strip("\"'")
    return dataset_path or None


def _extract_split(raw_text: str) -> str | None:
    match = re.search(r"(?im)^\s*split\s*:\s*([A-Za-z_]+)\s*$", str(raw_text or ""))
    if not match:
        return None
    split = match.group(1).strip().lower()
    return split or None


def _normalize_sampling_mode(value: Any) -> str | None:
    text = _normalize_request_text(str(value or "").strip().lower())
    if not text:
        return None
    if text == "balance":
        text = "balanced"
    if text in {"balanced", "random", "sorted"}:
        return text
    return None


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    try:
        return int(str(value).strip())
    except Exception:
        return None


def _validate_single_parameter_value(parameter: str, value: Any, *, limits: dict[str, int]) -> tuple[Any | None, str | None]:
    if parameter == "sampling_mode":
        normalized = _normalize_sampling_mode(value)
        if normalized is None:
            return None, f"Unsupported sampling_mode value: {value!r}. Expected one of balanced, random, or sorted."
        return normalized, None

    coerced = _coerce_int(value)
    if coerced is None:
        return None, f"{parameter} must be an integer value."

    if parameter == "top_k":
        if coerced < 1 or coerced > int(limits["max_top_k"]):
            return None, f"top_k must be between 1 and {limits['max_top_k']}."
        return coerced, None
    if parameter == "max_images":
        if coerced < 1 or coerced > int(limits["max_images"]):
            return None, f"max_images must be between 1 and {limits['max_images']}."
        return coerced, None
    if parameter == "max_images_per_class":
        if coerced < 1 or coerced > int(limits["max_images_per_class"]):
            return None, f"max_images_per_class must be between 1 and {limits['max_images_per_class']}."
        return coerced, None
    if parameter == "seed":
        if coerced < 0:
            return None, "seed must be a non-negative integer."
        return coerced, None
    return None, f"Unsupported sweep parameter: {parameter}"


def _validate_metric_names(selected_metrics: list[Any]) -> tuple[list[str], list[str]]:
    normalized_metrics: list[str] = []
    invalid_metrics: list[str] = []
    lookup = _metric_alias_lookup()
    for metric in _dedupe_preserve_order([str(item).strip() for item in selected_metrics if str(item).strip()]):
        canonical_metric = lookup.get(_normalize_request_text(metric), metric)
        if canonical_metric in SUPPORTED_SWEEP_METRICS:
            normalized_metrics.append(canonical_metric)
        else:
            invalid_metrics.append(metric)
    return normalized_metrics, invalid_metrics


def _validate_parameter_ranges(
    parameter_ranges: dict[str, Any],
    *,
    limits: dict[str, int],
) -> tuple[dict[str, list[Any]], list[str], list[str], bool]:
    normalized_ranges: dict[str, list[Any]] = {}
    errors: list[str] = []
    warnings: list[str] = []
    require_confirmation = False

    for raw_name, raw_values in parameter_ranges.items():
        parameter = _normalize_parameter_name(raw_name)
        if parameter not in SUPPORTED_SWEEP_PARAMETERS:
            errors.append(
                f"Unsupported inference-only sweep parameter: {raw_name!r}. Supported parameters are: "
                + ", ".join(SUPPORTED_SWEEP_PARAMETERS)
                + "."
            )
            continue
        values = _dedupe_preserve_order(_listify_values(raw_values))
        if not values:
            warnings.append(f"No sweep values were provided for {parameter}, so it will not be swept.")
            continue
        if len(values) > int(limits["max_values_per_parameter"]):
            errors.append(
                f"{parameter} includes {len(values)} values, which exceeds the safety limit of {limits['max_values_per_parameter']}."
            )
            require_confirmation = True
            continue
        normalized_values: list[Any] = []
        for value in values:
            normalized_value, error = _validate_single_parameter_value(parameter, value, limits=limits)
            if error is not None:
                errors.append(error)
                continue
            normalized_values.append(normalized_value)
        if normalized_values:
            normalized_ranges[parameter] = normalized_values
    if len(normalized_ranges) > int(limits["max_supported_parameters"]):
        errors.append(
            f"Requested {len(normalized_ranges)} swept parameters, which exceeds the safety limit of {limits['max_supported_parameters']}."
        )
        require_confirmation = True
    return normalized_ranges, errors, warnings, require_confirmation


def _build_baseline_values(
    baseline_values: dict[str, Any] | None,
    *,
    limits: dict[str, int],
) -> tuple[dict[str, Any], dict[str, str], list[str]]:
    baseline_values = dict(baseline_values or {})
    normalized_baseline: dict[str, Any] = {}
    baseline_sources: dict[str, str] = {}
    errors: list[str] = []

    unsupported_keys = [
        raw_name
        for raw_name in baseline_values
        if _normalize_parameter_name(raw_name) not in SUPPORTED_SWEEP_PARAMETERS
    ]
    for raw_name in unsupported_keys:
        errors.append(
            f"Unsupported baseline parameter: {raw_name!r}. Supported parameters are: "
            + ", ".join(SUPPORTED_SWEEP_PARAMETERS)
            + "."
        )

    for parameter in SUPPORTED_SWEEP_PARAMETERS:
        matched_user_value = None
        matched_by_user = False
        for raw_name, raw_value in baseline_values.items():
            if _normalize_parameter_name(raw_name) == parameter:
                matched_user_value = raw_value
                matched_by_user = True
                break
        candidate_value = matched_user_value if matched_by_user else DEFAULT_BASELINE_VALUES[parameter]
        normalized_value, error = _validate_single_parameter_value(parameter, candidate_value, limits=limits)
        if error is not None:
            errors.append(f"Invalid baseline value for {parameter}: {error}")
            continue
        normalized_baseline[parameter] = normalized_value
        baseline_sources[parameter] = "user" if matched_by_user else "default"

    if (
        "max_images" in normalized_baseline
        and "max_images_per_class" in normalized_baseline
        and normalized_baseline["max_images_per_class"] > normalized_baseline["max_images"]
    ):
        errors.append("Baseline max_images_per_class cannot exceed baseline max_images.")

    return normalized_baseline, baseline_sources, errors


def _validate_sweep_points_against_baseline(
    parameter_ranges: dict[str, list[Any]],
    baseline_values: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    baseline_max_images = baseline_values.get("max_images")
    baseline_max_images_per_class = baseline_values.get("max_images_per_class")

    if "max_images_per_class" in parameter_ranges and isinstance(baseline_max_images, int):
        for value in parameter_ranges["max_images_per_class"]:
            if int(value) > baseline_max_images:
                errors.append(
                    f"max_images_per_class value {value} exceeds baseline max_images {baseline_max_images}."
                )

    if "max_images" in parameter_ranges and isinstance(baseline_max_images_per_class, int):
        for value in parameter_ranges["max_images"]:
            if int(baseline_max_images_per_class) > int(value):
                errors.append(
                    f"max_images value {value} is smaller than baseline max_images_per_class {baseline_max_images_per_class}."
                )

    return errors


def _generate_sweep_points(
    parameter_ranges: dict[str, list[Any]],
    baseline_values: dict[str, Any],
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for parameter, values in parameter_ranges.items():
        for value in values:
            point_values = dict(baseline_values)
            point_values[parameter] = value
            points.append(
                {
                    "varied_parameter": parameter,
                    "varied_value": value,
                    "parameter_values": point_values,
                }
            )
    return points


def build_parameter_sweep_plan(
    *,
    dataset_path: str | None = None,
    split: str | None = None,
    parameter_ranges: dict[str, Any] | None = None,
    baseline_values: dict[str, Any] | None = None,
    selected_metrics: list[str] | None = None,
    max_total_runs: int | None = None,
    max_values_per_parameter: int | None = None,
) -> dict[str, Any]:
    limits = dict(DEFAULT_SWEEP_LIMITS)
    if max_total_runs is not None:
        limits["max_total_runs"] = int(max_total_runs)
    if max_values_per_parameter is not None:
        limits["max_values_per_parameter"] = int(max_values_per_parameter)

    errors: list[str] = []
    warnings: list[str] = []
    require_confirmation = False

    if parameter_ranges is None:
        parameter_ranges = {}
    if not isinstance(parameter_ranges, dict):
        return {
            "success": False,
            "study_type": "parameter_sweep",
            "errors": ["parameter_ranges must be a dictionary keyed by supported parameter name."],
            "warnings": [],
            "dataset_path": dataset_path,
            "split": split or SAFE_DEFAULTS["split"],
            "supported_parameters": list(SUPPORTED_SWEEP_PARAMETERS),
            "parameter_ranges": {},
            "baseline_values": {},
            "baseline_sources": {},
            "generated_sweep_points": [],
            "total_planned_runs": 0,
            "safety_limits": limits,
            "selected_metrics": list(DEFAULT_SELECTED_METRICS),
            "require_confirmation": False,
        }

    normalized_ranges, range_errors, range_warnings, range_require_confirmation = _validate_parameter_ranges(
        parameter_ranges,
        limits=limits,
    )
    errors.extend(range_errors)
    warnings.extend(range_warnings)
    require_confirmation = require_confirmation or range_require_confirmation

    if not normalized_ranges:
        errors.append("At least one non-empty supported parameter range is required to build a parameter sweep plan.")

    normalized_baseline, baseline_sources, baseline_errors = _build_baseline_values(
        baseline_values,
        limits=limits,
    )
    errors.extend(baseline_errors)

    if split is None:
        normalized_split = SAFE_DEFAULTS["split"]
        split_source = "default"
    else:
        normalized_split = str(split).strip().lower()
        split_source = "user"
    if normalized_split not in {"test", "train", "val", "validation"}:
        errors.append("split must be one of: test, train, val, validation.")

    normalized_metrics, invalid_metrics = _validate_metric_names(
        selected_metrics if selected_metrics is not None else list(DEFAULT_SELECTED_METRICS)
    )
    if invalid_metrics:
        errors.append(
            "Unsupported selected_metrics values: "
            + ", ".join(invalid_metrics)
            + ". Supported metrics are: "
            + ", ".join(SUPPORTED_SWEEP_METRICS)
            + "."
        )

    errors.extend(_validate_sweep_points_against_baseline(normalized_ranges, normalized_baseline))

    generated_sweep_points = _generate_sweep_points(normalized_ranges, normalized_baseline) if not errors else []
    total_planned_runs = len(generated_sweep_points)
    if total_planned_runs > int(limits["max_total_runs"]):
        errors.append(
            f"Total planned sweep points ({total_planned_runs}) exceed the safety limit of {limits['max_total_runs']}."
        )
        require_confirmation = True

    if not dataset_path:
        warnings.append("dataset_path was not provided; the future sweep runner will require a dataset path before execution.")

    return {
        "success": not errors,
        "study_type": "parameter_sweep",
        "errors": errors,
        "warnings": warnings,
        "require_confirmation": require_confirmation,
        "dataset_path": dataset_path,
        "split": normalized_split,
        "split_source": split_source,
        "supported_parameters": list(SUPPORTED_SWEEP_PARAMETERS),
        "parameter_ranges": normalized_ranges,
        "baseline_values": normalized_baseline,
        "baseline_sources": baseline_sources,
        "generated_sweep_points": generated_sweep_points,
        "total_planned_runs": total_planned_runs,
        "safety_limits": limits,
        "selected_metrics": normalized_metrics,
    }


def parse_parameter_sweep_request(text: str) -> dict[str, Any]:
    raw_text = str(text or "").strip()
    normalized_text = _normalize_request_text(raw_text)
    warnings: list[str] = []
    sections = _slice_sections(normalized_text)
    dataset_path = _extract_dataset_path(raw_text)
    split = _extract_split(raw_text)
    baseline_values = _parse_baseline_values(sections.get("baseline", ""))
    parameter_ranges = _parse_parameter_ranges(sections.get("sweep", ""))
    selected_metrics = _parse_selected_metrics(_extract_metric_section_text(sections, normalized_text))

    if not parameter_ranges and "parameter sweep" in normalized_text:
        warnings.append(
            "No obvious parameter ranges were parsed. Provide explicit values like `top_k 1,3,5` or `max_images: [50,100,200]`."
        )

    return {
        "success": True,
        "text": raw_text,
        "dataset_path": dataset_path,
        "split": split,
        "parameter_ranges": parameter_ranges,
        "baseline_values": baseline_values,
        "selected_metrics": selected_metrics,
        "warnings": warnings,
    }
