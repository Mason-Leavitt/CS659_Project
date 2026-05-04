#!/usr/bin/env python3
"""
Conversational helpers for planning no-retraining ablation studies.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from agent.model_imports import ensure_model_project_on_path

DEFAULT_DATASET_PATH = Path(r"E:\school\659\plantnet_300K\plantnet_300K")
SAFE_DEFAULTS = {
    "split": "test",
    "sampling_mode": "balanced",
    "seed": 42,
    "max_images_quick": 50,
    "max_images_robust": 200,
    "max_images_per_class": 1,
}


def _message_lower(message: str) -> str:
    return str(message or "").strip().lower()


def _extract_path_like_token(message: str) -> str | None:
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', message)
    for pair in quoted:
        value = next((part for part in pair if part), "")
        if value and (":" in value or "\\" in value or "/" in value):
            return value

    parts = re.split(r"\s+", str(message or "").strip())
    for part in parts:
        cleaned = part.strip(" ,.;:()[]{}")
        if ":" in cleaned or "\\" in cleaned or cleaned.lower() == "dataset_sample":
            return cleaned
    return None


def _extract_int(pattern: str, text: str) -> int | None:
    match = re.search(pattern, text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _resolve_default_dataset(use_default_dataset: bool) -> str | None:
    if use_default_dataset and DEFAULT_DATASET_PATH.exists():
        return str(DEFAULT_DATASET_PATH)
    return None


def parse_ablation_request(message: str) -> dict[str, Any]:
    text = _message_lower(message)
    dataset_path = _extract_path_like_token(message)
    use_default_dataset = "use default dataset" in text or "default dataset" in text
    if dataset_path is None:
        dataset_path = _resolve_default_dataset(use_default_dataset)

    run_profile = None
    max_images = None
    if "full test split" in text or "full dataset run" in text or re.search(r"\bfull\b", text):
        run_profile = "full"
        max_images = None
    elif "larger ablation" in text or "larger run" in text or re.search(r"\blarger\b", text):
        run_profile = "large"
        max_images = 500
    elif "robust ablation" in text or "robust run" in text or "stronger run" in text or re.search(r"\brobust\b", text):
        run_profile = "robust"
        max_images = 200
    elif (
        "quick ablation" in text
        or "small ablation" in text
        or "quick run" in text
        or re.search(r"\bquick\b", text)
        or re.search(r"\bsmall\b", text)
    ):
        run_profile = "quick"
        max_images = 50

    explicit_max = (
        _extract_int(r"\b(\d+)\s*image\b", text)
        or _extract_int(r"\b(\d+)\s*images\b", text)
        or _extract_int(r"\bmax[_ -]?images\s*=?\s*(\d+)\b", text)
    )
    if explicit_max is not None:
        max_images = explicit_max

    split = None
    for candidate in ("validation", "val", "train", "test"):
        if re.search(rf"\b{candidate}\b", text):
            split = candidate
            break

    sampling_mode = None
    for candidate in ("balanced", "random", "sorted"):
        if re.search(rf"\b{candidate}\b", text):
            sampling_mode = candidate
            break

    seed = _extract_int(r"\bseed\s*=?\s*(\d+)\b", text)
    max_images_per_class = (
        _extract_int(r"\bmax[_ -]?images[_ -]?per[_ -]?class\s*=?\s*(\d+)\b", text)
        or _extract_int(r"\b(\d+)\s*(?:images?|samples?)\s*per\s*class\b", text)
    )

    asks_for_run = any(
        token in text
        for token in ("run ablation", "run an ablation", "start ablation study", "run a balanced", "ablation study", "run ")
    )
    asks_for_recommendation = any(
        token in text
        for token in ("recommend", "what ablation should i run", "settings do you recommend")
    )
    asks_for_history = any(
        token in text
        for token in ("previous ablation", "prior runs", "compare ablation runs", "ablation history", "observations from prior runs")
    )

    return {
        "success": True,
        "message": message,
        "dataset_path": dataset_path,
        "split": split,
        "max_images": max_images,
        "sampling_mode": sampling_mode,
        "seed": seed,
        "max_images_per_class": max_images_per_class,
        "run_profile": run_profile,
        "use_default_dataset": use_default_dataset,
        "asks_for_run": asks_for_run,
        "asks_for_recommendation": asks_for_recommendation,
        "asks_for_history": asks_for_history,
        "asks_for_full_run": max_images is None and run_profile == "full",
    }


def get_ablation_parameter_prompt(current_context: dict[str, Any] | None = None) -> str:
    context = current_context or {}
    missing = list(context.get("missing_fields", []))
    if not missing:
        missing = ["dataset_path", "split", "max_images"]

    prompts = {
        "dataset_path": "Please provide the dataset path first. I can use `dataset_sample/` or an absolute PlantNet-style path.",
        "split": "Which split should I use: `test`, `train`, `val`, or `validation`?",
        "max_images": "How large should the run be? Safe defaults are `50` for a quick run or `200` for a stronger run.",
        "sampling_mode": "Which sampling mode do you want: `balanced`, `random`, or `sorted`? `balanced` is safest for small samples.",
        "seed": "If you want deterministic sampling, provide a seed. Otherwise I will use `42`.",
        "max_images_per_class": "Do you want to cap images per class? If not, I can leave that unset or use `1` for balanced small-sample studies.",
    }
    return " ".join(prompts[field] for field in missing if field in prompts)


def plan_ablation_from_request(message: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    parsed = parse_ablation_request(message)
    merged = dict(context or {})
    for key in ("dataset_path", "split", "max_images", "sampling_mode", "seed", "max_images_per_class", "run_profile"):
        if parsed.get(key) is not None:
            merged[key] = parsed[key]

    if merged.get("split") is None:
        merged["split"] = SAFE_DEFAULTS["split"]
    if merged.get("sampling_mode") is None:
        merged["sampling_mode"] = SAFE_DEFAULTS["sampling_mode"]
    if merged.get("seed") is None:
        merged["seed"] = SAFE_DEFAULTS["seed"]
    if merged.get("max_images") is None and parsed.get("run_profile") in {"quick", "robust", "large"}:
        mapping = {"quick": 50, "robust": 200, "large": 500}
        merged["max_images"] = mapping[parsed["run_profile"]]

    missing_fields: list[str] = []
    if not merged.get("dataset_path"):
        missing_fields.append("dataset_path")
    if merged.get("split") is None:
        missing_fields.append("split")
    if merged.get("max_images") is None and not parsed.get("asks_for_full_run"):
        missing_fields.append("max_images")

    warnings: list[str] = []
    if parsed.get("asks_for_full_run"):
        warnings.append("A full test-split run may take noticeably longer than the 50-image or 200-image presets.")
    if merged.get("sampling_mode") == "balanced" and merged.get("max_images_per_class") is None:
        warnings.append("Balanced sampling is preferred for small samples because it avoids over-representing the first sorted classes.")
    warnings.append("This planner only supports no-retraining, inference-time/post-training ablations.")

    needs_more_info = bool(missing_fields)
    plan = {
        "dataset_path": merged.get("dataset_path"),
        "split": merged.get("split", SAFE_DEFAULTS["split"]),
        "max_images": merged.get("max_images"),
        "sampling_mode": merged.get("sampling_mode", SAFE_DEFAULTS["sampling_mode"]),
        "seed": int(merged.get("seed", SAFE_DEFAULTS["seed"])),
        "max_images_per_class": merged.get("max_images_per_class"),
        "run_profile": merged.get("run_profile") or parsed.get("run_profile"),
        "asks_for_full_run": parsed.get("asks_for_full_run", False),
    }
    return {
        "success": True,
        "needs_more_info": needs_more_info,
        "missing_fields": missing_fields,
        "prompt": get_ablation_parameter_prompt({"missing_fields": missing_fields}) if needs_more_info else None,
        "plan": plan,
        "warnings": warnings,
        "parsed": parsed,
    }
