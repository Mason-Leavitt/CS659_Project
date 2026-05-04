#!/usr/bin/env python3
"""
Streamlit UI for the plant classification assistant.
"""
from __future__ import annotations

import json
import hashlib
from datetime import datetime
from io import StringIO
from io import BytesIO
import zipfile
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent

for candidate in (str(REPO_ROOT), str(CURRENT_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import streamlit as st

from agent.model_imports import ensure_model_project_on_path
from agent.ablation_history import describe_comparison_validity, make_unique_ablation_run_labels
from agent.agent_tools import (
    build_dataset_manifest,
    classify_with_hog_svm,
    classify_with_tflite,
    get_ablation_history,
    get_ablation_result_by_id,
    get_ablation_results,
    get_ablation_recommendations,
    get_latest_ablation_result,
    get_ablation_feasibility,
    get_artifact_status,
    get_metrics_summary,
    plan_ablation_study,
    compare_ablation_results,
    export_ablation_results,
    run_planned_ablation,
    run_sample_ablation,
)
from agent.plant_chat_agent import get_agent_mode_status, run_agent_message, run_agent_turn
from agent.parameter_sweep_planner import (
    DEFAULT_BASELINE_VALUES as SWEEP_DEFAULT_BASELINE_VALUES,
    DEFAULT_SELECTED_METRICS as SWEEP_DEFAULT_SELECTED_METRICS,
    build_parameter_sweep_plan,
)
from agent.parameter_sweep_runner import run_parameter_sweep

ensure_model_project_on_path()
import app_config


st.set_page_config(page_title="Plant Classification Agent", layout="wide")

_LOADING_OVERLAY_CSS = """
<style>
#global-loading-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.28);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 999999;
}
#global-loading-overlay .spinner-ring {
    width: 72px;
    height: 72px;
    border: 6px solid rgba(255, 255, 255, 0.9);
    border-top-color: transparent;
    border-radius: 50%;
    animation: streamlit-spinner-rotate 0.9s linear infinite;
    box-shadow: 0 0 18px rgba(0, 0, 0, 0.18);
}
@keyframes streamlit-spinner-rotate {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}
</style>
"""

st.markdown(_LOADING_OVERLAY_CSS, unsafe_allow_html=True)
_LOADING_OVERLAY_SLOT = st.empty()
_CHAT_HISTORY_HEIGHT = 420


def _parameter_sweep_sample_prompt() -> str:
    dataset_sample_path = (REPO_ROOT / "dataset_sample").resolve()
    return (
        "Run inference-only parameter sweep using the following details.\n\n"
        f"Dataset path: {dataset_sample_path}\n"
        "Split: test\n\n"
        "Baseline parameters:\n"
        "top_k: 5\n"
        "sampling_mode: balanced\n"
        "max_images: 100\n"
        "max_images_per_class: 5\n"
        "seed: 50\n\n"
        "Sweep ranges:\n"
        "top_k: 1,3,5,8,10\n"
        "sampling_mode: balanced,random,sorted\n"
        "max_images: 10,25,50,100,200\n"
        "max_images_per_class: 1,2,5,7,10\n"
        "seed: 7,25,42,123,200\n\n"
        "Metrics to plot:\n"
        "TFLite top-1 accuracy\n"
        "HOG+SVM top-1 accuracy\n"
        "TFLite top-k accuracy\n"
        "HOG+SVM top-k accuracy\n"
        "model agreement rate"
    )


def _show_loading_overlay() -> None:
    _LOADING_OVERLAY_SLOT.markdown(
        """
        <div id="global-loading-overlay">
          <div class="spinner-ring"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _hide_loading_overlay() -> None:
    _LOADING_OVERLAY_SLOT.empty()


def _run_with_loading(fn, *args, **kwargs):
    _show_loading_overlay()
    try:
        return fn(*args, **kwargs)
    finally:
        _hide_loading_overlay()


if "app_initialized" not in st.session_state:
    _show_loading_overlay()
    st.session_state.app_initialized = True
    st.rerun()


def _init_session_state() -> None:
    if "current_image_path" not in st.session_state:
        st.session_state.current_image_path = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_input_prefill" not in st.session_state:
        st.session_state.chat_input_prefill = ""
    if "chat_input_version" not in st.session_state:
        st.session_state.chat_input_version = 0
    if "uploaded_file_token" not in st.session_state:
        st.session_state.uploaded_file_token = None
    if "latest_tflite_result" not in st.session_state:
        st.session_state.latest_tflite_result = None
    if "latest_hog_result" not in st.session_state:
        st.session_state.latest_hog_result = None
    if "latest_model_used" not in st.session_state:
        st.session_state.latest_model_used = None
    if "image_display_width" not in st.session_state:
        st.session_state.image_display_width = 600
    if "chart_display_width_px" not in st.session_state:
        st.session_state.chart_display_width_px = 700
    if "chart_display_height_px" not in st.session_state:
        st.session_state.chart_display_height_px = 350
    if "chart_font_size" not in st.session_state:
        st.session_state.chart_font_size = 12
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    if "openai_model_dropdown" not in st.session_state:
        st.session_state.openai_model_dropdown = "gpt-5-nano"
    if "openai_model_custom" not in st.session_state:
        st.session_state.openai_model_custom = ""
    if "openai_model" not in st.session_state:
        st.session_state.openai_model = "gpt-5-nano"
    if "ablation_dataset_path" not in st.session_state:
        st.session_state.ablation_dataset_path = "dataset_sample"
    if "ablation_split" not in st.session_state:
        st.session_state.ablation_split = "test"
    if "ablation_max_images" not in st.session_state:
        st.session_state.ablation_max_images = 50
    if "ablation_sampling_mode" not in st.session_state:
        st.session_state.ablation_sampling_mode = "balanced"
    if "ablation_sampling_seed" not in st.session_state:
        st.session_state.ablation_sampling_seed = 42
    if "ablation_max_images_per_class" not in st.session_state:
        st.session_state.ablation_max_images_per_class = 0
    if "latest_manifest_result" not in st.session_state:
        st.session_state.latest_manifest_result = None
    if "latest_ablation_result" not in st.session_state:
        st.session_state.latest_ablation_result = None
    if "ablation_plan" not in st.session_state:
        st.session_state.ablation_plan = None
    if "ablation_history_summary" not in st.session_state:
        st.session_state.ablation_history_summary = None
    if "ablation_planner_message" not in st.session_state:
        st.session_state.ablation_planner_message = ""
    if "latest_ablation_history_export" not in st.session_state:
        st.session_state.latest_ablation_history_export = None
    if "latest_ablation_history_rows" not in st.session_state:
        st.session_state.latest_ablation_history_rows = None
    if "latest_ablation_run_view" not in st.session_state:
        st.session_state.latest_ablation_run_view = None
    if "latest_ablation_comparison" not in st.session_state:
        st.session_state.latest_ablation_comparison = None
    if "selected_ablation_run_id" not in st.session_state:
        st.session_state.selected_ablation_run_id = None
    if "selected_ablation_compare_run_ids" not in st.session_state:
        st.session_state.selected_ablation_compare_run_ids = []
    if "parameter_sweep_baseline_top_k" not in st.session_state:
        st.session_state.parameter_sweep_baseline_top_k = int(SWEEP_DEFAULT_BASELINE_VALUES["top_k"])
    if "parameter_sweep_baseline_sampling_mode" not in st.session_state:
        st.session_state.parameter_sweep_baseline_sampling_mode = str(SWEEP_DEFAULT_BASELINE_VALUES["sampling_mode"])
    if "parameter_sweep_baseline_max_images" not in st.session_state:
        st.session_state.parameter_sweep_baseline_max_images = int(SWEEP_DEFAULT_BASELINE_VALUES["max_images"])
    if "parameter_sweep_baseline_max_images_per_class" not in st.session_state:
        st.session_state.parameter_sweep_baseline_max_images_per_class = int(SWEEP_DEFAULT_BASELINE_VALUES["max_images_per_class"])
    if "parameter_sweep_baseline_seed" not in st.session_state:
        st.session_state.parameter_sweep_baseline_seed = int(SWEEP_DEFAULT_BASELINE_VALUES["seed"])
    if "parameter_sweep_enable_top_k" not in st.session_state:
        st.session_state.parameter_sweep_enable_top_k = False
    if "parameter_sweep_enable_sampling_mode" not in st.session_state:
        st.session_state.parameter_sweep_enable_sampling_mode = False
    if "parameter_sweep_enable_max_images" not in st.session_state:
        st.session_state.parameter_sweep_enable_max_images = False
    if "parameter_sweep_enable_max_images_per_class" not in st.session_state:
        st.session_state.parameter_sweep_enable_max_images_per_class = False
    if "parameter_sweep_enable_seed" not in st.session_state:
        st.session_state.parameter_sweep_enable_seed = False
    if "parameter_sweep_values_top_k" not in st.session_state:
        st.session_state.parameter_sweep_values_top_k = "1,3,5"
    if "parameter_sweep_values_sampling_mode" not in st.session_state:
        st.session_state.parameter_sweep_values_sampling_mode = "balanced,random,sorted"
    if "parameter_sweep_values_max_images" not in st.session_state:
        st.session_state.parameter_sweep_values_max_images = "50,100,200"
    if "parameter_sweep_values_max_images_per_class" not in st.session_state:
        st.session_state.parameter_sweep_values_max_images_per_class = "1,2,5"
    if "parameter_sweep_values_seed" not in st.session_state:
        st.session_state.parameter_sweep_values_seed = "7,42,123"
    if "parameter_sweep_selected_metrics" not in st.session_state:
        st.session_state.parameter_sweep_selected_metrics = list(SWEEP_DEFAULT_SELECTED_METRICS)
    if "parameter_sweep_plan" not in st.session_state:
        st.session_state.parameter_sweep_plan = None
    if "parameter_sweep_result" not in st.session_state:
        st.session_state.parameter_sweep_result = None
    if "parameter_sweep_chart_index" not in st.session_state:
        st.session_state.parameter_sweep_chart_index = 0
    if "show_parameter_sweep_plan_details" not in st.session_state:
        st.session_state.show_parameter_sweep_plan_details = False
    if "show_parameter_sweep_artifacts" not in st.session_state:
        st.session_state.show_parameter_sweep_artifacts = False
    if "show_parameter_sweep_charts" not in st.session_state:
        st.session_state.show_parameter_sweep_charts = False


def _append_chat(role: str, content: str) -> None:
    st.session_state.chat_history.append({"role": role, "content": content})


def _file_exists(path_text: Any) -> bool:
    if not path_text:
        return False
    try:
        return Path(str(path_text)).is_file()
    except Exception:
        return False


def _has_displayable_artifact(path_or_paths: Any) -> bool:
    if isinstance(path_or_paths, dict):
        return any(_has_displayable_artifact(value) for value in path_or_paths.values())
    if isinstance(path_or_paths, (list, tuple, set)):
        return any(_has_displayable_artifact(value) for value in path_or_paths)
    return _file_exists(path_or_paths)


def _toggle_button(show_label: str, state_key: str, *, hide_label: str | None = None) -> bool:
    if state_key not in st.session_state:
        st.session_state[state_key] = False
    is_visible = bool(st.session_state.get(state_key, False))
    button_label = hide_label or show_label.replace("Show", "Hide", 1)
    if st.button(button_label if is_visible else show_label, key=f"toggle_{state_key}", width="stretch"):
        st.session_state[state_key] = not is_visible
    return bool(st.session_state.get(state_key, False))


def _render_toggleable_section(show_label: str, state_key: str, render_fn, *, hide_label: str | None = None) -> None:
    if _toggle_button(show_label, state_key, hide_label=hide_label):
        render_fn()


def _sanitize_upload_name(filename: str) -> str:
    raw_name = Path(filename).name
    suffix = Path(raw_name).suffix
    stem = Path(raw_name).stem
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    if not safe_stem:
        safe_stem = "upload"
    safe_suffix = re.sub(r"[^A-Za-z0-9.]+", "", suffix)
    return f"{safe_stem}{safe_suffix}"


def _save_uploaded_file(uploaded_file) -> Path:
    uploads_dir = app_config.ensure_dir(app_config.DEFAULT_UPLOADS_DIR)
    safe_name = _sanitize_upload_name(uploaded_file.name)
    target = uploads_dir / safe_name
    suffix = target.suffix
    stem = target.stem
    counter = 1
    while target.exists():
        target = uploads_dir / f"{stem}_{counter}{suffix}"
        counter += 1
    target.write_bytes(uploaded_file.getvalue())
    return target


def _handle_uploaded_file(uploaded_file) -> None:
    file_bytes = uploaded_file.getvalue()
    file_token = f"{uploaded_file.name}:{hashlib.sha256(file_bytes).hexdigest()}"
    if st.session_state.uploaded_file_token == file_token and st.session_state.current_image_path:
        return

    saved_path = _save_uploaded_file(uploaded_file)
    st.session_state.current_image_path = str(saved_path)
    st.session_state.uploaded_file_token = file_token
    st.session_state.latest_tflite_result = None
    st.session_state.latest_hog_result = None
    st.session_state.latest_model_used = None
    for state_key in (
        "show_tflite_chart",
        "show_hog_chart",
        "show_manifest_summary",
        "show_ablation_plan",
        "show_ablation_summary",
        "show_ablation_recommendations",
        "show_ablation_history_summary_table",
        "show_ablation_history_summary_chart",
        "show_ablation_runs_table",
        "show_selected_run_details",
        "show_selected_run_comparison",
        "show_latest_ablation_comparison",
        "show_ablation_export_artifacts",
    ):
        st.session_state[state_key] = False


def _is_tflite_request(message: str) -> bool:
    text = str(message or "").strip().lower()
    return any(
        token in text
        for token in (
            "classify",
            "classification",
            "predict",
            "prediction",
            "identify",
            "species",
            "what plant",
            "compare",
            "comparison",
        )
    )


def _is_hog_request(message: str) -> bool:
    text = str(message or "").strip().lower()
    return any(
        token in text
        for token in (
            "hog",
            "svm",
            "hog+svm",
            "classical",
            "traditional",
            "baseline",
            "shallow",
            "hand-crafted",
            "handcrafted",
            "feature-based",
        )
    )


def _update_visual_results(message: str) -> None:
    if not st.session_state.current_image_path:
        return
    if _is_hog_request(message):
        st.session_state.latest_hog_result = _run_with_loading(
            classify_with_hog_svm,
            image_path=st.session_state.current_image_path,
            top_k=5,
        )
        st.session_state.latest_model_used = "hog"
    elif _is_tflite_request(message):
        st.session_state.latest_tflite_result = _run_with_loading(
            classify_with_tflite,
            image_path=st.session_state.current_image_path,
            top_k=5,
            color_correct="none",
        )
        st.session_state.latest_model_used = "tflite"


def _render_tflite_chart() -> None:
    result = st.session_state.latest_tflite_result
    if not result or not result.get("success"):
        return

    predictions = result.get("predictions", [])
    if not predictions:
        return

    labels = [str(pred.get("label", "")) for pred in predictions]
    probabilities = [100.0 * float(pred.get("probability", 0.0)) for pred in predictions]

    settings = _chart_display_settings()
    dpi = settings["dpi"]
    chart_width_px = settings["width_px"]
    chart_height_px = settings["height_px"]
    chart_font_size = settings["font_size"]
    current_image_path = st.session_state.current_image_path
    image_name = Path(current_image_path).name if current_image_path else ""
    image_stem = Path(current_image_path).stem if current_image_path else "image"
    chart_title = (
        f"TFLite Top-k Predictions of {image_name}"
        if image_name
        else "TFLite Top-k Predictions"
    )
    fig, ax = plt.subplots(
        figsize=(
            chart_width_px / dpi,
            chart_height_px / dpi,
        ),
        dpi=dpi,
    )
    ax.bar(range(len(labels)), probabilities, color="#2E8B57")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=chart_font_size)
    ax.set_ylabel("Probability (%)", fontsize=chart_font_size)
    ax.set_xlabel("Predicted label", fontsize=chart_font_size)
    ax.set_title(chart_title, fontsize=chart_font_size)
    ax.tick_params(axis="x", labelsize=chart_font_size)
    ax.tick_params(axis="y", labelsize=chart_font_size)
    fig.tight_layout()
    chart_buffer = BytesIO()
    fig.savefig(chart_buffer, format="png", dpi=dpi, bbox_inches="tight")
    chart_bytes = chart_buffer.getvalue()
    chart_buffer.close()
    st.pyplot(fig, clear_figure=True, use_container_width=False)
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", image_stem).strip("._-") or "image"
    st.download_button(
        "Download confidence chart (PNG)",
        data=chart_bytes,
        file_name=f"tflite_topk_predictions_{safe_stem}.png",
        mime="image/png",
    )
    plt.close(fig)


def _render_hog_chart(result: dict) -> None:
    if not result or not result.get("success"):
        return

    predictions = result.get("predictions", [])
    score_type = str(result.get("score_type") or "label_only")
    prediction = result.get("prediction", {})
    label = str(prediction.get("display_label") or prediction.get("label") or "")
    confidence = prediction.get("confidence")
    score = prediction.get("score")
    if not label:
        return
    if len(predictions) <= 1 and confidence is None and score is None:
        st.write(f"HOG+SVM prediction: {label}")
        return

    settings = _chart_display_settings()
    dpi = settings["dpi"]
    chart_width_px = settings["width_px"]
    chart_height_px = settings["height_px"]
    chart_font_size = settings["font_size"]
    current_image_path = st.session_state.current_image_path
    image_name = Path(current_image_path).name if current_image_path else ""
    image_stem = Path(current_image_path).stem if current_image_path else "image"
    chart_title = (
        f"HOG+SVM Prediction of {image_name}"
        if image_name
        else "HOG+SVM Prediction"
    )

    fig, ax = plt.subplots(
        figsize=(chart_width_px / dpi, chart_height_px / dpi),
        dpi=dpi,
    )

    if len(predictions) >= 2:
        labels = [str(item.get("display_label") or item.get("label") or "") for item in predictions]
        if score_type == "probability":
            values = [100.0 * float(item.get("confidence", 0.0) or 0.0) for item in predictions]
            metric_name = "Probability (%)"
            color = "#C46A1A"
        else:
            values = [float(item.get("score", 0.0) or 0.0) for item in predictions]
            metric_name = "Decision score"
            color = "#5B8E7D"
        ax.bar(range(len(labels)), values, color=color)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=chart_font_size)
    else:
        metric_name = "Confidence (%)" if confidence is not None else "Decision score"
        metric_value = 100.0 * float(confidence) if confidence is not None else float(score or 0.0)
        color = "#C46A1A" if confidence is not None else "#5B8E7D"
        ax.bar([0], [metric_value], color=color)
        ax.set_xticks([0])
        ax.set_xticklabels([label], rotation=20, ha="right", fontsize=chart_font_size)

    ax.set_ylabel(metric_name, fontsize=chart_font_size)
    ax.set_xlabel("Predicted label", fontsize=chart_font_size)
    ax.set_title(chart_title, fontsize=chart_font_size)
    ax.tick_params(axis="x", labelsize=chart_font_size)
    ax.tick_params(axis="y", labelsize=chart_font_size)
    fig.tight_layout()

    chart_buffer = BytesIO()
    fig.savefig(chart_buffer, format="png", dpi=dpi, bbox_inches="tight")
    chart_bytes = chart_buffer.getvalue()
    chart_buffer.close()
    st.pyplot(fig, clear_figure=True, use_container_width=False)
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", image_stem).strip("._-") or "image"
    st.download_button(
        "Download HOG+SVM chart (PNG)",
        data=chart_bytes,
        file_name=f"hog_svm_prediction_{safe_stem}.png",
        mime="image/png",
    )
    plt.close(fig)


def _render_active_chart() -> None:
    latest_model_used = st.session_state.latest_model_used
    if latest_model_used == "hog":
        _render_hog_chart(st.session_state.latest_hog_result)
        return
    if latest_model_used == "tflite":
        _render_tflite_chart()


def _render_inference_result_sections() -> None:
    tflite_result = st.session_state.latest_tflite_result
    hog_result = st.session_state.latest_hog_result
    if isinstance(tflite_result, dict) and tflite_result.get("success"):
        _render_toggleable_section("Show TFLite prediction chart", "show_tflite_chart", _render_tflite_chart)
    if isinstance(hog_result, dict) and hog_result.get("success"):
        _render_toggleable_section(
            "Show HOG+SVM prediction chart",
            "show_hog_chart",
            lambda: _render_hog_chart(hog_result),
        )


def _parse_sweep_values_text(value_text: str) -> list[str]:
    text = str(value_text or "").strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return [part.strip() for part in text.split(",") if part.strip()]


def _build_parameter_sweep_ranges_from_inputs(raw_inputs: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    ranges: dict[str, list[str]] = {}
    for parameter, config in raw_inputs.items():
        if not bool(config.get("enabled")):
            continue
        values = _parse_sweep_values_text(str(config.get("values", "")))
        if values:
            ranges[parameter] = values
    return ranges


def _parameter_sweep_metric_options() -> list[tuple[str, str]]:
    return [
        ("tflite_top1_accuracy", "TFLite top-1 accuracy"),
        ("tflite_topk_accuracy", "TFLite top-k accuracy"),
        ("hog_top1_accuracy", "HOG+SVM top-1 accuracy"),
        ("hog_topk_accuracy", "HOG+SVM top-k accuracy"),
        ("model_agreement_rate", "Model agreement rate"),
    ]


def _current_parameter_sweep_plan_inputs() -> dict[str, Any]:
    parameter_ranges = _build_parameter_sweep_ranges_from_inputs(
        {
            "top_k": {
                "enabled": st.session_state.parameter_sweep_enable_top_k,
                "values": st.session_state.parameter_sweep_values_top_k,
            },
            "sampling_mode": {
                "enabled": st.session_state.parameter_sweep_enable_sampling_mode,
                "values": st.session_state.parameter_sweep_values_sampling_mode,
            },
            "max_images": {
                "enabled": st.session_state.parameter_sweep_enable_max_images,
                "values": st.session_state.parameter_sweep_values_max_images,
            },
            "max_images_per_class": {
                "enabled": st.session_state.parameter_sweep_enable_max_images_per_class,
                "values": st.session_state.parameter_sweep_values_max_images_per_class,
            },
            "seed": {
                "enabled": st.session_state.parameter_sweep_enable_seed,
                "values": st.session_state.parameter_sweep_values_seed,
            },
        }
    )
    return {
        "dataset_path": str(st.session_state.ablation_dataset_path or ""),
        "split": str(st.session_state.ablation_split or "test"),
        "baseline_values": {
            "top_k": int(st.session_state.parameter_sweep_baseline_top_k),
            "sampling_mode": str(st.session_state.parameter_sweep_baseline_sampling_mode),
            "max_images": int(st.session_state.parameter_sweep_baseline_max_images),
            "max_images_per_class": int(st.session_state.parameter_sweep_baseline_max_images_per_class),
            "seed": int(st.session_state.parameter_sweep_baseline_seed),
        },
        "parameter_ranges": parameter_ranges,
        "selected_metrics": list(st.session_state.parameter_sweep_selected_metrics or []),
    }


def _build_current_parameter_sweep_plan() -> dict[str, Any]:
    inputs = _current_parameter_sweep_plan_inputs()
    return build_parameter_sweep_plan(
        dataset_path=inputs["dataset_path"],
        split=inputs["split"],
        parameter_ranges=inputs["parameter_ranges"],
        baseline_values=inputs["baseline_values"],
        selected_metrics=inputs["selected_metrics"],
    )


def _existing_parameter_sweep_chart_entries(result: dict | None) -> list[dict[str, Any]]:
    if not isinstance(result, dict):
        return []
    summary = result.get("summary", {}) if isinstance(result.get("summary"), dict) else {}
    charts = summary.get("charts", []) if isinstance(summary.get("charts"), list) else []
    entries: list[dict[str, Any]] = []
    for chart in charts:
        if not isinstance(chart, dict):
            continue
        if _file_exists(chart.get("path")):
            entries.append(chart)
    return entries


def _zip_chart_artifacts(chart_entries: list[dict[str, Any]]) -> bytes:
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for chart in chart_entries:
            path_text = chart.get("path")
            if not _file_exists(path_text):
                continue
            chart_path = Path(str(path_text))
            zf.writestr(chart_path.name, chart_path.read_bytes())
    return zip_buffer.getvalue()


def _clamp_chart_index(current_index: int, chart_count: int) -> int:
    if chart_count <= 0:
        return 0
    return max(0, min(int(current_index), chart_count - 1))


def _run_parameter_sweep_with_status() -> None:
    status_slot = st.empty()
    with st.spinner("Running parameter sweep..."):
        status_slot.info("Building parameter sweep plan and executing one-factor inference-only runs.")
        plan = _build_current_parameter_sweep_plan()
        st.session_state.parameter_sweep_plan = plan
        if not plan.get("success") or plan.get("require_confirmation"):
            st.session_state.parameter_sweep_result = None
            st.session_state.parameter_sweep_chart_index = 0
            return
        result = run_parameter_sweep(plan)
        st.session_state.parameter_sweep_result = result
        st.session_state.parameter_sweep_chart_index = 0
        st.session_state.show_parameter_sweep_artifacts = True
        st.session_state.show_parameter_sweep_charts = False
    status_slot.empty()


def _looks_like_openai_api_key(value: str | None) -> bool:
    if not value:
        return False
    text = value.strip()
    return text.startswith("sk-") and len(text) >= 20


def _sync_openai_settings() -> None:
    entered_key = str(st.session_state.openai_api_key or "").strip()
    if entered_key:
        os.environ["OPENAI_API_KEY"] = entered_key
    else:
        os.environ.pop("OPENAI_API_KEY", None)

    custom_model = str(st.session_state.openai_model_custom or "").strip()
    dropdown_model = str(st.session_state.openai_model_dropdown or "gpt-5-nano").strip()
    st.session_state.openai_model = custom_model or dropdown_model


def _render_sidebar() -> None:
    st.sidebar.header("OpenAI Agent Settings")
    st.session_state.openai_api_key = st.sidebar.text_input(
        "OpenAI API key",
        value=str(st.session_state.openai_api_key or ""),
        type="password",
        help="Used only for the current app session. The key is not written to disk.",
    )
    if st.session_state.openai_api_key and not _looks_like_openai_api_key(st.session_state.openai_api_key):
        st.sidebar.warning("The API key format does not look valid. Agent mode will stay in fallback.")

    recommended_models = ["gpt-5-nano", "gpt-4.1-nano", "gpt-4o-mini", "gpt-5.4-nano"]
    st.session_state.openai_model_dropdown = st.sidebar.selectbox(
        "Recommended OpenAI models",
        options=recommended_models,
        index=recommended_models.index(str(st.session_state.openai_model_dropdown or "gpt-5-nano"))
        if str(st.session_state.openai_model_dropdown or "gpt-5-nano") in recommended_models
        else 0,
        help="Choose a recommended model or override it with a custom value below.",
    )
    st.sidebar.caption("The recommended list is ordered from cheapest to most expensive.")
    st.session_state.openai_model_custom = st.sidebar.text_input(
        "Custom model override",
        value=str(st.session_state.openai_model_custom or ""),
        help="Optional. If set, this overrides the recommended model selection.",
    )

    _sync_openai_settings()
    mode_status = get_agent_mode_status(model_name=st.session_state.openai_model)
    st.sidebar.markdown(f"**Mode:** {mode_status['mode']}")
    if mode_status["mode"] == "Agent":
        st.sidebar.info(mode_status["message"])
    else:
        st.sidebar.info(mode_status["message"])

    st.sidebar.header("Artifact Status")
    st.sidebar.subheader("Display Settings")
    st.session_state.image_display_width = st.sidebar.slider(
        "Image display width",
        min_value=300,
        max_value=1200,
        value=int(st.session_state.image_display_width),
        step=25,
    )
    st.session_state.chart_display_width_px = st.sidebar.slider(
        "Chart display width",
        min_value=300,
        max_value=1200,
        value=int(st.session_state.chart_display_width_px),
        step=25,
    )
    st.session_state.chart_display_height_px = st.sidebar.slider(
        "Chart display height",
        min_value=200,
        max_value=700,
        value=int(st.session_state.chart_display_height_px),
        step=25,
    )
    st.session_state.chart_font_size = st.sidebar.slider(
        "Chart font size",
        min_value=8,
        max_value=24,
        value=int(st.session_state.chart_font_size),
        step=1,
    )

    status = get_artifact_status()

    st.sidebar.write(f"TFLite model: {'Available' if status['tflite_model']['exists'] else 'Missing'}")
    st.sidebar.write(f"Default labels: {'Available' if status['default_labels']['exists'] else 'Missing'}")
    st.sidebar.write(f"HOG+SVM model: {'Available' if status['hog_svm_model']['exists'] else 'Missing'}")
    st.sidebar.write(f"HOG+SVM labels: {'Available' if status['hog_svm_labels']['exists'] else 'Missing'}")
    st.sidebar.write(f"Result directory: {'Available' if status['result_dir']['exists'] else 'Missing'}")

    if not status["hog_svm_model"]["exists"] or not status["hog_svm_labels"]["exists"]:
        st.sidebar.warning("HOG+SVM artifacts are missing, so baseline comparison is not currently available.")

    with st.sidebar.expander("Metrics Availability", expanded=False):
        metrics = _run_with_loading(get_metrics_summary)
        if metrics.get("available"):
            st.write("Known metrics files found:")
            for run in metrics.get("runs", []):
                files = run.get("files", [])
                if not files:
                    continue
                st.write(f"Run directory: {run.get('run_dir', 'result/')}")
                st.write(f"Method: {run.get('method', 'unknown')}")
                metrics_dict = run.get("metrics", {}) if isinstance(run.get("metrics"), dict) else {}
                metric_bits = []
                for key in ("accuracy", "validation_accuracy", "f1_macro"):
                    value = metrics_dict.get(key)
                    if isinstance(value, (int, float)):
                        metric_bits.append(f"{key}: {value:.4f}")
                if metric_bits:
                    st.write("Key metrics:")
                    for bit in metric_bits:
                        st.write(f"- {bit}")
                st.write("Files:")
                for file_name in files:
                    st.write(f"- {file_name}")
        else:
            st.warning(metrics.get("error") or "Metrics are not available.")

    with st.sidebar.expander("Ablation Feasibility", expanded=False):
        feasibility = _run_with_loading(get_ablation_feasibility)
        if not feasibility.get("success"):
            st.warning(feasibility.get("error") or "Ablation feasibility is not available.")
        else:
            hardware = feasibility.get("hardware", {})
            hardware_lines = [
                f"Python: {hardware.get('python_version', 'unknown')}",
                f"Platform: {hardware.get('platform', 'unknown')}",
            ]
            if hardware.get("ram_total_gb") is not None:
                hardware_lines.append(f"RAM: {hardware['ram_total_gb']} GB")
            if hardware.get("gpu_name"):
                gpu_line = str(hardware["gpu_name"])
                if hardware.get("gpu_vram_gb") is not None:
                    gpu_line += f" ({hardware['gpu_vram_gb']} GB VRAM)"
                hardware_lines.append(f"GPU: {gpu_line}")
            else:
                hardware_lines.append("GPU: not currently visible")
            hardware_lines.append(f"PyTorch installed: {hardware.get('torch_installed', False)}")
            hardware_lines.append(f"TensorFlow installed: {hardware.get('tensorflow_installed', False)}")

            st.write("Hardware summary")
            for line in hardware_lines:
                st.write(f"- {line}")

            def _render_list(title: str, items: list[str]) -> None:
                st.write(title)
                if items:
                    for item in items:
                        st.write(f"- {item}")
                else:
                    st.write("- None")

            _render_list("Possible now", feasibility.get("possible_now", []))
            _render_list("Requires dataset", feasibility.get("requires_dataset", []))
            _render_list("Requires training", feasibility.get("requires_training", []))
            _render_list("Skipped", feasibility.get("skipped", []))


def _run_and_render(message: str) -> None:
    prior_history = list(st.session_state.chat_history)
    _append_chat("user", message)
    agent_result = _run_with_loading(
        run_agent_turn,
        message=message,
        image_path=st.session_state.current_image_path,
        chat_history=prior_history,
        model_name=st.session_state.openai_model,
    )
    _sync_agent_outputs_to_ui_state(agent_result)
    response_text = str(agent_result.get("message", "")) if isinstance(agent_result, dict) else str(agent_result)
    _append_chat("assistant", response_text)
    _update_visual_results(message)
    st.rerun()


def _format_tflite_assistant_message(result: dict) -> str:
    if not result.get("success"):
        return f"TFLite classification is not available: {result.get('error') or 'unknown error'}"

    predictions = result.get("predictions", [])
    if not predictions:
        return "TFLite classification completed, but no predictions were returned."

    top1 = predictions[0]
    top1_prob = float(top1.get("probability", 0.0))
    if top1_prob >= 0.90:
        confidence_text = "very confident"
    elif top1_prob >= 0.70:
        confidence_text = "confident"
    elif top1_prob >= 0.50:
        confidence_text = "moderately confident"
    else:
        confidence_text = "uncertain"

    parts = [
        f"TFLite predicts '{top1.get('label')}' as the top result with {100.0 * top1_prob:.2f}% confidence, so the model appears {confidence_text}."
    ]
    if len(predictions) > 1:
        preview = []
        for pred in predictions[: min(3, len(predictions))]:
            preview.append(
                f"{pred.get('rank')}. {pred.get('label')} ({100.0 * float(pred.get('probability', 0.0)):.2f}%)"
            )
        parts.append("Top predictions: " + "; ".join(preview) + ".")

        top2 = predictions[1]
        top2_prob = float(top2.get("probability", 0.0))
        if top1_prob - top2_prob < 0.15:
            parts.append(
                f"The top two classes are fairly close, so the model may be uncertain between '{top1.get('label')}' and '{top2.get('label')}'."
            )

    warning = result.get("warning")
    if warning:
        parts.append(str(warning))
    return " ".join(parts)


def _run_direct_tflite_action(message: str) -> None:
    _append_chat("user", message)
    result = _run_with_loading(
        classify_with_tflite,
        image_path=st.session_state.current_image_path,
        top_k=5,
        color_correct="none",
    )
    st.session_state.latest_tflite_result = result
    st.session_state.latest_model_used = "tflite"
    _append_chat("assistant", _format_tflite_assistant_message(result))
    st.rerun()


def _run_direct_hog_action(message: str) -> None:
    _append_chat("user", message)
    result = _run_with_loading(
        classify_with_hog_svm,
        image_path=st.session_state.current_image_path,
        top_k=5,
    )
    st.session_state.latest_hog_result = result
    st.session_state.latest_model_used = "hog"
    response = _run_with_loading(
        run_agent_message,
        message=message,
        image_path=st.session_state.current_image_path,
        chat_history=list(st.session_state.chat_history[:-1]),
        model_name=st.session_state.openai_model,
    )
    _append_chat("assistant", response)
    st.rerun()


def _render_chat_history() -> None:
    with st.container(height=_CHAT_HISTORY_HEIGHT):
        if not st.session_state.chat_history:
            st.caption("Conversation history will appear here.")
            return
        for entry in st.session_state.chat_history:
            with st.chat_message(entry["role"]):
                st.markdown(entry["content"])


def _is_parameter_sweep_plan_dict(payload: Any) -> bool:
    return isinstance(payload, dict) and (
        payload.get("study_type") == "parameter_sweep"
        or (
            isinstance(payload.get("parameter_ranges"), dict)
            and isinstance(payload.get("baseline_values"), dict)
            and "generated_sweep_points" in payload
        )
    )


def _is_parameter_sweep_result_dict(payload: Any) -> bool:
    return isinstance(payload, dict) and (
        payload.get("study_type") == "parameter_sweep"
        or (
            "summary" in payload
            and "artifacts" in payload
            and "status" in payload
            and ("results" in payload or "sweep_id" in payload or "output_dir" in payload)
        )
    )


def _extract_parameter_sweep_payload(agent_result: Any) -> dict[str, Any]:
    extracted = {
        "parameter_sweep_plan": None,
        "parameter_sweep_result": None,
    }
    if not isinstance(agent_result, dict):
        return extracted
    plan_candidate = agent_result.get("parameter_sweep_plan")
    result_candidate = agent_result.get("parameter_sweep_result")
    if _is_parameter_sweep_plan_dict(plan_candidate):
        extracted["parameter_sweep_plan"] = plan_candidate
    elif _is_parameter_sweep_plan_dict(agent_result.get("plan")):
        extracted["parameter_sweep_plan"] = agent_result.get("plan")
    if _is_parameter_sweep_result_dict(result_candidate):
        extracted["parameter_sweep_result"] = result_candidate
    elif _is_parameter_sweep_result_dict(agent_result.get("result")):
        extracted["parameter_sweep_result"] = agent_result.get("result")
    elif _is_parameter_sweep_result_dict(agent_result):
        extracted["parameter_sweep_result"] = agent_result
    if extracted["parameter_sweep_result"] and not extracted["parameter_sweep_plan"]:
        result_plan = extracted["parameter_sweep_result"].get("plan")
        if _is_parameter_sweep_plan_dict(result_plan):
            extracted["parameter_sweep_plan"] = result_plan
    return extracted


def _sync_parameter_sweep_agent_payload(payload: dict[str, Any]) -> None:
    plan = payload.get("parameter_sweep_plan")
    result = payload.get("parameter_sweep_result")
    if _is_parameter_sweep_plan_dict(plan):
        previous_plan = st.session_state.get("parameter_sweep_plan")
        st.session_state.parameter_sweep_plan = plan
        st.session_state.show_parameter_sweep_plan_details = True
        if previous_plan != plan and not _is_parameter_sweep_result_dict(result):
            st.session_state.parameter_sweep_result = None
            st.session_state.parameter_sweep_chart_index = 0
            st.session_state.show_parameter_sweep_artifacts = False
            st.session_state.show_parameter_sweep_charts = False
    if _is_parameter_sweep_result_dict(result):
        st.session_state.parameter_sweep_result = result
        st.session_state.parameter_sweep_chart_index = 0
        st.session_state.show_parameter_sweep_artifacts = _has_displayable_artifact(result.get("artifacts", {}))
        st.session_state.show_parameter_sweep_charts = bool(_existing_parameter_sweep_chart_entries(result))
        result_plan = result.get("plan")
        if _is_parameter_sweep_plan_dict(result_plan):
            st.session_state.parameter_sweep_plan = result_plan
            st.session_state.show_parameter_sweep_plan_details = True


def _sync_agent_outputs_to_ui_state(agent_result: Any) -> None:
    payload = _extract_parameter_sweep_payload(agent_result)
    if payload.get("parameter_sweep_plan") or payload.get("parameter_sweep_result"):
        _sync_parameter_sweep_agent_payload(payload)


def _chat_input_widget_key(version: int | None = None) -> str:
    current_version = int(st.session_state.get("chat_input_version", 0) if version is None else version)
    return f"chat_input_widget_{current_version}"


def _set_chat_prefill(text: str) -> None:
    st.session_state.chat_input_prefill = str(text or "")
    st.session_state.chat_input_version = int(st.session_state.get("chat_input_version", 0)) + 1
    st.rerun()


def _clear_chat_input_after_send() -> None:
    st.session_state.chat_input_prefill = ""
    st.session_state.chat_input_version = int(st.session_state.get("chat_input_version", 0)) + 1


def _render_chat_input_area() -> None:
    control_col, spacer_col = st.columns([1, 4])
    with control_col:
        if st.button("Insert parameter sweep example", width="stretch"):
            _set_chat_prefill(_parameter_sweep_sample_prompt())
    with spacer_col:
        st.caption("Insert a parser-friendly sweep template, edit it, then send when ready.")

    current_version = int(st.session_state.get("chat_input_version", 0))
    widget_key = _chat_input_widget_key(current_version)
    prefill_value = str(st.session_state.get("chat_input_prefill", ""))
    with st.form(key=f"chat_input_form_{current_version}", clear_on_submit=True):
        message = st.text_area(
            "Conversation input",
            key=widget_key,
            value=prefill_value,
            height=180,
            placeholder="Ask about the uploaded image, artifact status, available metrics, or parameter sweeps.",
        )
        submitted = st.form_submit_button("Send message", width="stretch")
    if submitted:
        cleaned_message = str(message or "").strip()
        if cleaned_message:
            _clear_chat_input_after_send()
            _run_and_render(cleaned_message)


def _render_manifest_summary(result: dict | None) -> None:
    if not result:
        return
    if not result.get("success"):
        st.warning(result.get("error") or "Manifest could not be built.")
        return

    st.write(f"Dataset path: {result.get('dataset_path')}")
    st.write(f"Resolved image root: {result.get('resolved_image_root')}")
    st.write(f"Split: {result.get('split')}")
    st.write(f"Images: {result.get('num_images', 0)}")
    st.write(f"Classes: {result.get('num_classes', 0)}")
    st.write(f"Sampling mode: {result.get('sampling_mode', 'n/a')}")
    st.write(f"Seed: {result.get('seed', 'n/a')}")
    st.write(f"Max images per class: {result.get('max_images_per_class') if result.get('max_images_per_class') is not None else 'None'}")
    if result.get("warning"):
        st.info(str(result["warning"]))
    if result.get("manifest_path"):
        st.write(f"Manifest CSV: {result.get('manifest_path')}")
    class_distribution = result.get("class_distribution", {})
    if class_distribution:
        distribution_rows = [
            {"label": label, "count": count}
            for label, count in class_distribution.items()
        ]
        st.write("Class distribution")
        st.dataframe(distribution_rows)
    preview = result.get("preview", [])
    if preview:
        st.write("Preview")
        st.dataframe(preview)


def _ablation_summary_has_displayable_files(result: dict | None) -> bool:
    if not isinstance(result, dict) or not result.get("success"):
        return False
    files = result.get("files", {})
    return _has_displayable_artifact(files)


def _render_ablation_summary(result: dict | None) -> None:
    if not result:
        return
    if not result.get("success"):
        st.warning(result.get("error") or "Sample ablation did not complete.")
        return

    summary = result.get("summary", {})
    metrics = summary.get("metrics", {}) if isinstance(summary.get("metrics"), dict) else {}
    st.write(f"Output directory: {result.get('output_dir')}")
    st.write(f"Images requested: {summary.get('num_images_requested', 0)}")
    st.write(f"Images attempted: {summary.get('num_images_attempted', 0)}")
    st.write(f"Images evaluated: {summary.get('num_images_evaluated', 0)}")
    st.write(f"Failures: {summary.get('num_failures', 0)}")
    if result.get("warning"):
        st.info(str(result["warning"]))

    skipped = result.get("skipped", {})
    if skipped:
        st.write("Skipped components")
        for key, reason in skipped.items():
            st.write(f"- {key}: {reason}")

    notes = metrics.get("metrics_notes", [])
    if notes:
        st.write("Key results")
        for note in notes:
            st.write(f"- {note}")

    files = result.get("files", {})
    if files:
        st.write("Downloads")
        for label, path_text in files.items():
            if not path_text:
                continue
            file_path = Path(str(path_text))
            if not file_path.is_file():
                continue
            mime = "text/plain"
            if file_path.suffix.lower() == ".csv":
                mime = "text/csv"
            elif file_path.suffix.lower() == ".json":
                mime = "application/json"
            elif file_path.suffix.lower() == ".md":
                mime = "text/markdown"
            elif file_path.suffix.lower() == ".png":
                mime = "image/png"
            st.download_button(
                f"Download {label}",
                data=file_path.read_bytes(),
                file_name=file_path.name,
                mime=mime,
            )
            if file_path.suffix.lower() == ".png":
                st.image(str(file_path), caption=file_path.name, width=_chart_image_width_px())


def _render_ablation_recommendations(result: dict | None) -> None:
    if not result:
        return
    if not result.get("success"):
        st.warning(result.get("error") or "Ablation recommendations are unavailable.")
        return
    st.write(f"Enough history: {result.get('enough_history', False)}")
    recommendations = result.get("recommendations", {})
    if recommendations:
        st.write("Recommended settings")
        rows = [{"parameter": key, "value": value} for key, value in recommendations.items()]
        st.dataframe(rows)
    constraints = result.get("constraints", [])
    if constraints:
        st.write("Constraints")
        for item in constraints:
            st.write(f"- {item}")
    rationale = result.get("rationale", [])
    if rationale:
        st.write("Rationale")
        for item in rationale:
            st.write(f"- {item}")


def _render_ablation_history_summary_table(result: dict | None) -> None:
    if not result:
        return
    if not result.get("success"):
        st.warning(result.get("error") or "Ablation history is unavailable.")
        return
    latest_runs = result.get("latest_runs", [])
    if latest_runs:
        rows = []
        for item in latest_runs:
            metrics = item.get("metrics", {}) if isinstance(item.get("metrics"), dict) else {}
            rows.append(
                {
                    "run_id": item.get("run_id"),
                    "created_at": item.get("created_at"),
                    "split": item.get("split"),
                    "sampling_mode": item.get("sampling_mode"),
                    "max_images": item.get("max_images"),
                    "num_images_evaluated": item.get("num_images_evaluated"),
                    "tflite_topk_accuracy": metrics.get("tflite_topk_accuracy"),
                    "hog_topk_accuracy": metrics.get("hog_topk_accuracy"),
                    "agreement_rate": metrics.get("model_agreement_rate"),
                    "stability_rate": metrics.get("tflite_color_stability_rate"),
                }
            )
        st.write("Prior runs")
        st.dataframe(rows)


def _render_ablation_history_summary_chart(result: dict | None) -> None:
    if not result:
        return
    if not result.get("success"):
        st.warning(result.get("error") or "Ablation history is unavailable.")
        return
    history_files = result.get("history_files", {})
    metrics_png = history_files.get("metrics_png") if isinstance(history_files, dict) else None
    if metrics_png and Path(str(metrics_png)).is_file():
        st.image(
            str(metrics_png),
            caption="Ablation history metrics. Run labels use MMDD_HHMM when available; full run IDs remain in tables and exports.",
            width=_chart_image_width_px(),
        )


def _render_ablation_run_view(result: dict | None) -> None:
    if not result:
        return
    if not result.get("success"):
        st.warning(result.get("error") or "Ablation run details are unavailable.")
        return
    run = result.get("run", {})
    metrics = run.get("metrics", {}) if isinstance(run.get("metrics"), dict) else {}
    st.write("Selected run details")
    st.write(f"Run ID: {run.get('run_id')}")
    st.write(f"Created at: {run.get('created_at')}")
    st.write(f"Dataset path: {run.get('dataset_path')}")
    st.write(f"Split: {run.get('split')}")
    st.write(f"Sampling mode: {run.get('sampling_mode')}")
    st.write(f"Seed: {run.get('seed')}")
    st.write(f"Max images: {run.get('max_images')}")
    st.write(
        "Max images per class: "
        f"{run.get('max_images_per_class') if run.get('max_images_per_class') is not None else 'None'}"
    )
    st.write(f"Images evaluated: {run.get('num_images_evaluated')}")
    st.write(f"Classes: {run.get('num_classes')}")
    st.write(f"Output directory: {run.get('output_dir')}")
    metric_rows = [
        {"metric": "model_agreement_rate", "value": metrics.get("model_agreement_rate")},
        {"metric": "tflite_color_stability_rate", "value": metrics.get("tflite_color_stability_rate")},
        {"metric": "tflite_top1_accuracy", "value": metrics.get("tflite_top1_accuracy")},
        {"metric": "tflite_topk_accuracy", "value": metrics.get("tflite_topk_accuracy")},
        {"metric": "hog_top1_accuracy", "value": metrics.get("hog_top1_accuracy")},
        {"metric": "hog_topk_accuracy", "value": metrics.get("hog_topk_accuracy")},
    ]
    st.write("Key metrics")
    st.dataframe(metric_rows)
    caveats = run.get("caveats", [])
    if caveats:
        st.write("Caveats")
        for item in caveats:
            st.write(f"- {item}")
    generated_files = run.get("generated_files", {}) if isinstance(run.get("generated_files"), dict) else {}
    if generated_files:
        st.write("Downloads")
        missing_files = []
        for label, path_text in generated_files.items():
            if not path_text:
                continue
            file_path = Path(str(path_text))
            if not file_path.is_file():
                missing_files.append(f"{label}: {file_path}")
                continue
            mime = "text/plain"
            if file_path.suffix.lower() == ".csv":
                mime = "text/csv"
            elif file_path.suffix.lower() == ".json":
                mime = "application/json"
            elif file_path.suffix.lower() == ".md":
                mime = "text/markdown"
            elif file_path.suffix.lower() == ".png":
                mime = "image/png"
            st.download_button(
                f"Download selected run {label}",
                data=file_path.read_bytes(),
                file_name=file_path.name,
                mime=mime,
            )
            if file_path.suffix.lower() == ".png":
                st.image(str(file_path), caption=file_path.name, width=_chart_image_width_px())
        if missing_files:
            st.warning("Some generated files are missing:")
            for item in missing_files:
                st.write(f"- {item}")


def _format_ablation_run_option(run: dict) -> str:
    created_at = str(run.get("created_at") or "unknown time")
    run_id = str(run.get("run_id") or "unknown-run")
    sampling_mode = str(run.get("sampling_mode") or "unknown")
    max_images = run.get("max_images")
    max_images_text = "full" if max_images is None else str(max_images)
    return f"{created_at} | {run_id} | {sampling_mode} | {max_images_text} images"


def _parse_ablation_created_at(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None


def _ablation_rows_newest_first(rows: list[dict]) -> list[dict]:
    indexed_rows = list(enumerate(rows))

    def _sort_key(item: tuple[int, dict]) -> tuple[int, datetime | int]:
        index, row = item
        parsed = _parse_ablation_created_at(row.get("created_at"))
        if parsed is not None:
            return (1, parsed)
        return (0, index)

    sorted_rows = sorted(indexed_rows, key=_sort_key, reverse=True)
    return [row for _, row in sorted_rows]


def _latest_ablation_row(rows: list[dict]) -> dict | None:
    ordered = _ablation_rows_newest_first(rows)
    if not ordered:
        return None
    return ordered[0]


def _viewed_ablation_run_id(run_view: Any) -> str:
    if not isinstance(run_view, dict):
        return ""
    run = run_view.get("run", {})
    if not isinstance(run, dict):
        return ""
    return str(run.get("run_id") or "")


def _selected_run_view_is_stale(selected_run_id: str) -> bool:
    if not selected_run_id:
        return False
    current_view_run_id = _viewed_ablation_run_id(st.session_state.latest_ablation_run_view)
    return bool(current_view_run_id) and current_view_run_id != str(selected_run_id)


def _run_selected_result_with_status(run_id: str) -> None:
    status_slot = st.empty()
    with st.spinner("Loading selected ablation run..."):
        status_slot.info("Reading the selected ablation run from persistent history.")
        st.session_state.latest_ablation_run_view = get_ablation_result_by_id(run_id)
    status_slot.empty()


def _build_ablation_comparison_rows(runs: list[dict]) -> list[dict]:
    rows = []
    for run in runs:
        metrics = run.get("metrics", {}) if isinstance(run.get("metrics"), dict) else {}
        rows.append(
            {
                "run_id": run.get("run_id"),
                "created_at": run.get("created_at"),
                "split": run.get("split"),
                "sampling_mode": run.get("sampling_mode"),
                "seed": run.get("seed"),
                "max_images": run.get("max_images"),
                "max_images_per_class": run.get("max_images_per_class"),
                "num_images_evaluated": run.get("num_images_evaluated"),
                "num_classes": run.get("num_classes"),
                "model_agreement_rate": metrics.get("model_agreement_rate"),
                "tflite_color_stability_rate": metrics.get("tflite_color_stability_rate"),
                "tflite_top1_accuracy": metrics.get("tflite_top1_accuracy"),
                "tflite_topk_accuracy": metrics.get("tflite_topk_accuracy"),
                "hog_top1_accuracy": metrics.get("hog_top1_accuracy"),
                "hog_topk_accuracy": metrics.get("hog_topk_accuracy"),
                "output_dir": run.get("output_dir"),
            }
        )
    return rows


def _comparison_metric_value(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else math.nan
    text = str(value or "").strip()
    if not text:
        return math.nan
    if text.lower() in {"n/a", "na", "nan", "none", "null", "missing", "unknown"}:
        return math.nan
    try:
        numeric = float(text)
    except Exception:
        return math.nan
    return numeric if math.isfinite(numeric) else math.nan


def _chart_display_settings() -> dict[str, int]:
    return {
        "width_px": int(st.session_state.get("chart_display_width_px", 700)),
        "height_px": int(st.session_state.get("chart_display_height_px", 350)),
        "font_size": int(st.session_state.get("chart_font_size", 12)),
        "dpi": 100,
    }


def _chart_figure_size(settings: dict[str, int]) -> tuple[float, float]:
    dpi = max(int(settings.get("dpi", 100)), 1)
    return (
        float(settings.get("width_px", 700)) / dpi,
        float(settings.get("height_px", 350)) / dpi,
    )


def _chart_image_width_px() -> int:
    return int(_chart_display_settings().get("width_px", 700))


def _render_selected_ablation_comparison_chart(
    comparison_rows: list[dict],
    validity: dict[str, Any] | None = None,
) -> bytes:
    settings = _chart_display_settings()
    dpi = settings["dpi"]
    chart_font_size = settings["font_size"]
    fig, axes = plt.subplots(2, 2, figsize=_chart_figure_size(settings), dpi=dpi)
    row_positions = list(range(len(comparison_rows)))
    run_labels = make_unique_ablation_run_labels(comparison_rows)
    tflite_top1 = [_comparison_metric_value(row.get("tflite_top1_accuracy")) for row in comparison_rows]
    hog_top1 = [_comparison_metric_value(row.get("hog_top1_accuracy")) for row in comparison_rows]
    tflite_topk = [_comparison_metric_value(row.get("tflite_topk_accuracy")) for row in comparison_rows]
    hog_topk = [_comparison_metric_value(row.get("hog_topk_accuracy")) for row in comparison_rows]
    agreement = [_comparison_metric_value(row.get("model_agreement_rate")) for row in comparison_rows]
    stability = [_comparison_metric_value(row.get("tflite_color_stability_rate")) for row in comparison_rows]
    bar_width = 0.38
    validity_info = validity or describe_comparison_validity(comparison_rows)
    validity_status = str(validity_info.get("status") or "unknown")
    validity_title = str(validity_info.get("title") or "Comparison validity")
    validity_summary = str(validity_info.get("summary") or "")
    has_missing_metrics = any(
        math.isnan(value)
        for series in (tflite_top1, hog_top1, tflite_topk, hog_topk, agreement, stability)
        for value in series
    )

    def _bar_series(ax: Any, positions: list[float], values: list[float], *, label: str, color: str) -> None:
        valid_positions = [pos for pos, value in zip(positions, values) if not math.isnan(value)]
        valid_values = [value for value in values if not math.isnan(value)]
        if valid_positions:
            ax.bar(valid_positions, valid_values, width=bar_width, label=label, color=color)

    ax = axes[0][0]
    _bar_series(ax, [x - bar_width / 2 for x in row_positions], tflite_top1, label="TFLite top-1", color="#2E8B57")
    _bar_series(ax, [x + bar_width / 2 for x in row_positions], hog_top1, label="HOG top-1", color="#C46A1A")
    ax.set_title("Top-1 Accuracy by Run", fontsize=chart_font_size)
    ax.set_xticks(row_positions)
    ax.set_xticklabels(run_labels, rotation=20, ha="right", fontsize=chart_font_size)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Selected run label", fontsize=chart_font_size)
    ax.set_ylabel("Accuracy rate", fontsize=chart_font_size)
    ax.tick_params(axis="y", labelsize=chart_font_size)
    ax.legend(fontsize=chart_font_size)

    ax = axes[0][1]
    _bar_series(ax, [x - bar_width / 2 for x in row_positions], tflite_topk, label="TFLite top-k", color="#3A7CA5")
    _bar_series(ax, [x + bar_width / 2 for x in row_positions], hog_topk, label="HOG top-k", color="#B56576")
    ax.set_title("Top-k Accuracy by Run", fontsize=chart_font_size)
    ax.set_xticks(row_positions)
    ax.set_xticklabels(run_labels, rotation=20, ha="right", fontsize=chart_font_size)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Selected run label", fontsize=chart_font_size)
    ax.set_ylabel("Accuracy rate", fontsize=chart_font_size)
    ax.tick_params(axis="y", labelsize=chart_font_size)
    ax.legend(fontsize=chart_font_size)

    ax = axes[1][0]
    ax.plot(row_positions, agreement, marker="o", color="#5B8E7D")
    ax.set_title("Model Agreement by Run", fontsize=chart_font_size)
    ax.set_xticks(row_positions)
    ax.set_xticklabels(run_labels, rotation=20, ha="right", fontsize=chart_font_size)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Selected run label", fontsize=chart_font_size)
    ax.set_ylabel("Agreement rate", fontsize=chart_font_size)
    ax.tick_params(axis="y", labelsize=chart_font_size)

    ax = axes[1][1]
    ax.plot(row_positions, stability, marker="o", color="#6A4C93")
    ax.set_title("TFLite Color Stability by Run", fontsize=chart_font_size)
    ax.set_xticks(row_positions)
    ax.set_xticklabels(run_labels, rotation=20, ha="right", fontsize=chart_font_size)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Selected run label", fontsize=chart_font_size)
    ax.set_ylabel("Stability rate", fontsize=chart_font_size)
    ax.tick_params(axis="y", labelsize=chart_font_size)

    fig.suptitle(f"Selected Ablation Comparison ({validity_status})", fontsize=chart_font_size + 2, y=0.98)
    fig.text(
        0.5,
        0.945,
        f"{validity_title}: {validity_summary}. Run labels use MMDD_HHMM when available.",
        ha="center",
        va="top",
        fontsize=max(chart_font_size - 2, 8),
    )
    if has_missing_metrics:
        fig.text(
            0.5,
            0.02,
            "Some selected runs are missing metric values; missing values are omitted rather than plotted as zero.",
            ha="center",
            va="bottom",
            fontsize=max(chart_font_size - 3, 8),
        )
    fig.tight_layout(rect=(0, 0.08, 1, 0.9))
    chart_buffer = BytesIO()
    fig.savefig(chart_buffer, format="png", dpi=dpi, bbox_inches="tight")
    chart_bytes = chart_buffer.getvalue()
    chart_buffer.close()
    st.pyplot(fig, clear_figure=True, use_container_width=True)
    plt.close(fig)
    return chart_bytes


def _comparison_export_payload(
    comparison_rows: list[dict],
    *,
    validity: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "comparison_rows": comparison_rows,
        "comparison_validity": validity or {},
    }
    if isinstance(result, dict):
        for key in ("requested_run_ids", "matched_run_ids", "missing_run_ids", "comparison_warnings", "explanation"):
            value = result.get(key)
            if value not in (None, "", []):
                payload[key] = value
    warning_messages = _comparison_warning_messages(result)
    if warning_messages:
        payload["comparison_warning_messages"] = warning_messages
    return payload


def _comparison_export_markdown(payload: dict[str, Any]) -> str:
    rows = payload.get("comparison_rows", []) if isinstance(payload, dict) else []
    validity = payload.get("comparison_validity", {}) if isinstance(payload, dict) else {}
    lines = ["# Ablation Comparison Summary", ""]
    explanation = payload.get("explanation") if isinstance(payload, dict) else None
    if explanation:
        lines.extend([str(explanation), ""])
    warning_messages = payload.get("comparison_warning_messages", []) if isinstance(payload, dict) else []
    if warning_messages:
        lines.append("## Comparison Warnings")
        lines.extend(f"- {item}" for item in warning_messages if item)
        lines.append("")
    requested_run_ids = payload.get("requested_run_ids", []) if isinstance(payload, dict) else []
    matched_run_ids = payload.get("matched_run_ids", []) if isinstance(payload, dict) else []
    missing_run_ids = payload.get("missing_run_ids", []) if isinstance(payload, dict) else []
    if requested_run_ids:
        lines.append("Requested run IDs: " + ", ".join(f"`{item}`" for item in requested_run_ids))
    if matched_run_ids:
        lines.append("Matched run IDs: " + ", ".join(f"`{item}`" for item in matched_run_ids))
    if missing_run_ids:
        lines.append("Missing run IDs: " + ", ".join(f"`{item}`" for item in missing_run_ids))
    if requested_run_ids or matched_run_ids or missing_run_ids:
        lines.append("")
    if isinstance(validity, dict) and validity:
        lines.append("## Comparison Validity")
        summary = validity.get("summary") or "Comparison validity is unavailable."
        lines.append(str(summary))
        differing_fields = validity.get("differing_fields", [])
        if differing_fields:
            lines.append("")
            lines.append("Differing fields: " + ", ".join(str(field) for field in differing_fields))
        caveats = validity.get("caveats", [])
        if caveats:
            lines.append("")
            lines.extend(f"- {item}" for item in caveats if item)
        lines.append("")
    lines.append("## Compared Runs")
    if not rows:
        lines.append("No comparison rows were available.")
    else:
        for row in rows:
            lines.append(
                "- "
                + f"{row.get('run_id')}: split={row.get('split')} sampling={row.get('sampling_mode')} "
                + f"max_images={row.get('max_images')} "
                + f"TFLite top-k={row.get('tflite_topk_accuracy')} "
                + f"HOG top-k={row.get('hog_topk_accuracy')}."
            )
    return "\n".join(lines).strip()


def _write_selected_ablation_comparison_artifacts(
    comparison_rows: list[dict],
    chart_bytes: bytes | None,
    *,
    validity: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    stem: str = "selected_ablation_comparison",
) -> dict[str, str]:
    history_dir = app_config.ensure_dir(app_config.DEFAULT_RESULT_DIR / "ablation_history")
    csv_path = history_dir / f"{stem}.csv"
    json_path = history_dir / f"{stem}.json"
    md_path = history_dir / f"{stem}.md"
    png_path = history_dir / f"{stem}.png"

    fieldnames = [
        "run_id",
        "created_at",
        "split",
        "sampling_mode",
        "seed",
        "max_images",
        "max_images_per_class",
        "num_images_evaluated",
        "num_classes",
        "model_agreement_rate",
        "tflite_color_stability_rate",
        "tflite_top1_accuracy",
        "tflite_topk_accuracy",
        "hog_top1_accuracy",
        "hog_topk_accuracy",
        "output_dir",
    ]
    csv_buffer = StringIO()
    import csv

    writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in comparison_rows:
        writer.writerow({key: row.get(key) for key in fieldnames})
    csv_path.write_text(csv_buffer.getvalue(), encoding="utf-8")
    payload = _comparison_export_payload(comparison_rows, validity=validity, result=result)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_comparison_export_markdown(payload), encoding="utf-8")
    if chart_bytes:
        png_path.write_bytes(chart_bytes)

    artifacts = {
        "comparison_csv": str(csv_path),
        "comparison_json": str(json_path),
        "comparison_markdown": str(md_path),
    }
    if chart_bytes:
        artifacts["comparison_png"] = str(png_path)
    return artifacts


def _render_selected_ablation_comparison(selected_runs: list[dict]) -> None:
    st.write("Selected runs comparison")
    if len(selected_runs) < 2:
        st.info("Select at least two runs to compare.")
        return

    comparison_rows = _build_ablation_comparison_rows(selected_runs)
    st.dataframe(comparison_rows)

    validity = describe_comparison_validity(selected_runs)
    caveats = validity.get("caveats", [])
    differing_fields = validity.get("differing_fields", [])
    status = validity.get("status")
    summary = validity.get("summary") or "Comparison validity is unavailable."
    if status == "direct":
        st.success(summary)
    elif status == "partial":
        st.info(summary)
    else:
        st.warning(summary)
    if differing_fields:
        st.caption("Differing fields: " + ", ".join(str(field) for field in differing_fields))
    for caveat in caveats[1:]:
        st.caption(caveat)
    if any(
        math.isnan(_comparison_metric_value(row.get(metric_name)))
        for row in comparison_rows
        for metric_name in (
            "tflite_top1_accuracy",
            "hog_top1_accuracy",
            "tflite_topk_accuracy",
            "hog_topk_accuracy",
            "model_agreement_rate",
            "tflite_color_stability_rate",
        )
    ):
        st.caption("Some selected runs are missing metric values; missing values are omitted rather than plotted as zero.")

    chart_bytes = _render_selected_ablation_comparison_chart(comparison_rows, validity)
    artifacts = _write_selected_ablation_comparison_artifacts(
        comparison_rows,
        chart_bytes,
        validity=validity,
    )

    st.write("Selected comparison downloads")
    for label, path_text in artifacts.items():
        file_path = Path(path_text)
        if not file_path.is_file():
            st.warning(f"Missing comparison artifact: {file_path}")
            continue
        mime = "text/plain"
        if file_path.suffix.lower() == ".csv":
            mime = "text/csv"
        elif file_path.suffix.lower() == ".json":
            mime = "application/json"
        elif file_path.suffix.lower() == ".png":
            mime = "image/png"
        st.download_button(
            f"Download {label}",
            data=file_path.read_bytes(),
            file_name=file_path.name,
            mime=mime,
        )


def _comparison_warning_messages(result: dict[str, Any] | None) -> list[str]:
    if not isinstance(result, dict):
        return []
    explicit_warnings = [
        str(item).strip()
        for item in result.get("comparison_warnings", [])
        if str(item).strip()
    ]
    if explicit_warnings:
        return explicit_warnings
    missing_run_ids = [
        str(item).strip()
        for item in result.get("missing_run_ids", [])
        if str(item).strip()
    ]
    if missing_run_ids:
        return [
            "Requested run IDs were not found and were omitted: "
            + ", ".join(f"`{run_id}`" for run_id in missing_run_ids)
            + "."
        ]
    return []


def _render_ablation_comparison(result: dict | None) -> None:
    if not result:
        return
    warning_messages = _comparison_warning_messages(result if isinstance(result, dict) else None)
    if not result.get("success"):
        st.warning(result.get("error") or "Ablation comparison is unavailable.")
        if warning_messages:
            st.warning("Comparison warnings:\n- " + "\n- ".join(warning_messages))
        requested_run_ids = result.get("requested_run_ids", []) if isinstance(result, dict) else []
        matched_run_ids = result.get("matched_run_ids", []) if isinstance(result, dict) else []
        missing_run_ids = result.get("missing_run_ids", []) if isinstance(result, dict) else []
        if requested_run_ids:
            st.caption("Requested run IDs: " + ", ".join(str(item) for item in requested_run_ids))
        if matched_run_ids:
            st.caption("Matched run IDs: " + ", ".join(str(item) for item in matched_run_ids))
        if missing_run_ids:
            st.caption("Missing run IDs: " + ", ".join(str(item) for item in missing_run_ids))
        return
    rows = result.get("comparison_rows", [])
    if rows:
        st.write("Comparison across runs")
        st.dataframe(rows)
    if warning_messages:
        st.warning("Comparison warnings:\n- " + "\n- ".join(warning_messages))
    validity = result.get("comparison_validity") if isinstance(result, dict) else None
    if isinstance(validity, dict):
        summary = validity.get("summary") or "Comparison validity is unavailable."
        status = validity.get("status")
        if status == "direct":
            st.success(summary)
        elif status == "partial":
            st.info(summary)
        else:
            st.warning(summary)
        differing_fields = validity.get("differing_fields", [])
        if differing_fields:
            st.caption("Differing fields: " + ", ".join(str(field) for field in differing_fields))
        for caveat in validity.get("caveats", [])[1:]:
            st.caption(str(caveat))
    explanation = result.get("explanation")
    if explanation:
        st.markdown(explanation)
    if rows:
        artifacts = _write_selected_ablation_comparison_artifacts(
            rows,
            chart_bytes=None,
            validity=validity if isinstance(validity, dict) else None,
            result=result if isinstance(result, dict) else None,
            stem="latest_ablation_comparison",
        )
        st.write("Comparison downloads")
        for label, path_text in artifacts.items():
            file_path = Path(path_text)
            if not file_path.is_file():
                continue
            mime = "text/plain"
            if file_path.suffix.lower() == ".csv":
                mime = "text/csv"
            elif file_path.suffix.lower() == ".json":
                mime = "application/json"
            elif file_path.suffix.lower() == ".md":
                mime = "text/markdown"
            elif file_path.suffix.lower() == ".png":
                mime = "image/png"
            st.download_button(
                f"Download {label}",
                data=file_path.read_bytes(),
                file_name=file_path.name,
                mime=mime,
            )


def _render_ablation_export(result: dict | None) -> None:
    if not result:
        return
    if not result.get("success"):
        st.warning(result.get("error") or "Ablation export artifacts are unavailable.")
        return
    artifacts = result.get("artifacts", {})
    if not artifacts:
        return
    st.write("Ablation history artifacts")
    for label, path_text in artifacts.items():
        if not path_text:
            continue
        file_path = Path(str(path_text))
        if not file_path.is_file():
            continue
        mime = "text/plain"
        if file_path.suffix.lower() == ".csv":
            mime = "text/csv"
        elif file_path.suffix.lower() == ".json":
            mime = "application/json"
        elif file_path.suffix.lower() == ".md":
            mime = "text/markdown"
        elif file_path.suffix.lower() == ".png":
            mime = "image/png"
        st.download_button(
            f"Download {label}",
            data=file_path.read_bytes(),
            file_name=file_path.name,
            mime=mime,
        )
        if file_path.suffix.lower() == ".png":
            st.image(str(file_path), caption=file_path.name, width=_chart_image_width_px())


def _render_ablation_plan(plan_result: dict | None) -> None:
    if not plan_result:
        return
    if not plan_result.get("success"):
        st.warning(plan_result.get("error") or "Ablation planning failed.")
        return
    if plan_result.get("needs_more_info"):
        st.info(plan_result.get("prompt") or "More information is needed to plan the ablation.")
    plan = plan_result.get("plan", {})
    if plan:
        st.write("Current planned settings")
        st.dataframe([plan])
    warnings = plan_result.get("warnings", [])
    if warnings:
        st.write("Planner notes")
        for item in warnings:
            st.write(f"- {item}")


def _render_parameter_sweep_plan_details(plan: dict[str, Any]) -> None:
    st.write(f"Plan success: {plan.get('success', False)}")
    st.write(f"Total planned runs: {plan.get('total_planned_runs', 0)}")
    st.write("Baseline values")
    st.dataframe([dict(plan.get("baseline_values", {}))])
    st.write("Parameters swept")
    swept = list((plan.get("parameter_ranges") or {}).keys())
    st.write(", ".join(swept) if swept else "None")
    st.write("Selected metrics")
    metric_label_map = dict(_parameter_sweep_metric_options())
    metric_labels = [metric_label_map.get(metric, metric) for metric in plan.get("selected_metrics", [])]
    st.write(", ".join(metric_labels) if metric_labels else "None")
    sample_points = list(plan.get("generated_sweep_points", []))[:5]
    if sample_points:
        st.write("Preview sweep points")
        st.dataframe(sample_points)


def _render_parameter_sweep_artifacts(result: dict[str, Any]) -> None:
    artifacts = result.get("artifacts", {}) if isinstance(result, dict) else {}
    if not isinstance(artifacts, dict):
        return
    st.write("Parameter sweep downloads")
    for label, path_text in artifacts.items():
        if not _file_exists(path_text):
            continue
        file_path = Path(str(path_text))
        mime = "text/plain"
        if file_path.suffix.lower() == ".csv":
            mime = "text/csv"
        elif file_path.suffix.lower() == ".json":
            mime = "application/json"
        elif file_path.suffix.lower() == ".md":
            mime = "text/markdown"
        elif file_path.suffix.lower() == ".png":
            mime = "image/png"
        st.download_button(
            f"Download {label}",
            data=file_path.read_bytes(),
            file_name=file_path.name,
            mime=mime,
        )


def _render_parameter_sweep_chart_viewer(result: dict[str, Any]) -> None:
    chart_entries = _existing_parameter_sweep_chart_entries(result)
    if not chart_entries:
        return

    st.session_state.parameter_sweep_chart_index = _clamp_chart_index(
        st.session_state.parameter_sweep_chart_index,
        len(chart_entries),
    )
    chart_index = int(st.session_state.parameter_sweep_chart_index)
    current_chart = chart_entries[chart_index]
    current_path = Path(str(current_chart["path"]))

    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
    with nav_col1:
        if st.button("Previous chart", key="parameter_sweep_prev_chart", width="stretch"):
            st.session_state.parameter_sweep_chart_index = _clamp_chart_index(chart_index - 1, len(chart_entries))
            st.rerun()
    with nav_col2:
        st.caption(f"Chart {chart_index + 1} of {len(chart_entries)}: {current_chart.get('varied_parameter', 'unknown')}")
    with nav_col3:
        if st.button("Next chart", key="parameter_sweep_next_chart", width="stretch"):
            st.session_state.parameter_sweep_chart_index = _clamp_chart_index(chart_index + 1, len(chart_entries))
            st.rerun()

    st.image(str(current_path), caption=current_chart.get("filename") or current_path.name, width=_chart_image_width_px())

    download_col1, download_col2 = st.columns(2)
    with download_col1:
        st.download_button(
            "Download current chart",
            data=current_path.read_bytes(),
            file_name=current_path.name,
            mime="image/png",
        )
    with download_col2:
        chart_zip = _zip_chart_artifacts(chart_entries)
        if chart_zip:
            st.download_button(
                "Download all charts (ZIP)",
                data=chart_zip,
                file_name="parameter_sweep_charts.zip",
                mime="application/zip",
            )


def _render_parameter_sweep_panel() -> None:
    with st.expander("Inference-only Parameter Sweep", expanded=False):
        st.write(
            "Configure a one-factor-at-a-time inference-only sweep. This varies only inference-time controls; "
            "training-time HOG/SVM hyperparameters are not swept."
        )
        st.caption(
            f"Uses the current dataset path and split from the ablation controls above: "
            f"`{st.session_state.ablation_dataset_path}` / `{st.session_state.ablation_split}`."
        )

        st.write("Baseline / static values")
        base_col1, base_col2, base_col3 = st.columns(3)
        with base_col1:
            st.session_state.parameter_sweep_baseline_top_k = int(
                st.number_input(
                    "Baseline top_k",
                    min_value=1,
                    max_value=10,
                    value=int(st.session_state.parameter_sweep_baseline_top_k),
                    step=1,
                    help="Used whenever top_k is not the varied parameter.",
                )
            )
            st.session_state.parameter_sweep_baseline_sampling_mode = st.selectbox(
                "Baseline sampling_mode",
                options=["balanced", "random", "sorted"],
                index=["balanced", "random", "sorted"].index(str(st.session_state.parameter_sweep_baseline_sampling_mode)),
                help="Used whenever sampling_mode is not the varied parameter.",
            )
        with base_col2:
            st.session_state.parameter_sweep_baseline_max_images = int(
                st.number_input(
                    "Baseline max_images",
                    min_value=1,
                    value=int(st.session_state.parameter_sweep_baseline_max_images),
                    step=1,
                    help="Used whenever max_images is not the varied parameter.",
                )
            )
            st.session_state.parameter_sweep_baseline_max_images_per_class = int(
                st.number_input(
                    "Baseline max_images_per_class",
                    min_value=1,
                    value=int(st.session_state.parameter_sweep_baseline_max_images_per_class),
                    step=1,
                    help="Used whenever max_images_per_class is not the varied parameter.",
                )
            )
        with base_col3:
            st.session_state.parameter_sweep_baseline_seed = int(
                st.number_input(
                    "Baseline seed",
                    min_value=0,
                    value=int(st.session_state.parameter_sweep_baseline_seed),
                    step=1,
                    help="Used whenever seed is not the varied parameter.",
                )
            )

        st.write("Sweep values")
        sweep_rows = [
            ("top_k", "Sweep top_k?", "parameter_sweep_enable_top_k", "parameter_sweep_values_top_k", "Example: 1,3,5"),
            ("sampling_mode", "Sweep sampling_mode?", "parameter_sweep_enable_sampling_mode", "parameter_sweep_values_sampling_mode", "Example: balanced,random,sorted"),
            ("max_images", "Sweep max_images?", "parameter_sweep_enable_max_images", "parameter_sweep_values_max_images", "Example: 50,100,200"),
            ("max_images_per_class", "Sweep max_images_per_class?", "parameter_sweep_enable_max_images_per_class", "parameter_sweep_values_max_images_per_class", "Example: 1,2,5"),
            ("seed", "Sweep seed?", "parameter_sweep_enable_seed", "parameter_sweep_values_seed", "Example: 7,42,123"),
        ]
        for parameter, checkbox_label, enabled_key, values_key, help_text in sweep_rows:
            col_check, col_input = st.columns([1, 3])
            with col_check:
                st.session_state[enabled_key] = st.checkbox(
                    checkbox_label,
                    value=bool(st.session_state.get(enabled_key, False)),
                    key=f"{enabled_key}_checkbox",
                )
            with col_input:
                st.session_state[values_key] = st.text_input(
                    f"{parameter} values",
                    value=str(st.session_state.get(values_key, "")),
                    help=help_text,
                    disabled=not bool(st.session_state.get(enabled_key, False)),
                    key=f"{values_key}_input",
                )

        metric_options = _parameter_sweep_metric_options()
        metric_label_map = {name: label for name, label in metric_options}
        current_metric_labels = [
            metric_label_map.get(metric, metric)
            for metric in st.session_state.parameter_sweep_selected_metrics
            if metric in metric_label_map
        ]
        current_parameter_ranges = _build_parameter_sweep_ranges_from_inputs(
            {
                "top_k": {
                    "enabled": st.session_state.parameter_sweep_enable_top_k,
                    "values": st.session_state.parameter_sweep_values_top_k,
                },
                "sampling_mode": {
                    "enabled": st.session_state.parameter_sweep_enable_sampling_mode,
                    "values": st.session_state.parameter_sweep_values_sampling_mode,
                },
                "max_images": {
                    "enabled": st.session_state.parameter_sweep_enable_max_images,
                    "values": st.session_state.parameter_sweep_values_max_images,
                },
                "max_images_per_class": {
                    "enabled": st.session_state.parameter_sweep_enable_max_images_per_class,
                    "values": st.session_state.parameter_sweep_values_max_images_per_class,
                },
                "seed": {
                    "enabled": st.session_state.parameter_sweep_enable_seed,
                    "values": st.session_state.parameter_sweep_values_seed,
                },
            }
        )
        if not current_parameter_ranges:
            st.info("Select at least one parameter and provide sweep values before previewing or running a parameter sweep.")
        chosen_metric_labels = st.multiselect(
            "Metrics to plot",
            options=[label for _, label in metric_options],
            default=current_metric_labels or [metric_label_map[metric] for metric in SWEEP_DEFAULT_SELECTED_METRICS],
        )
        reverse_metric_map = {label: name for name, label in metric_options}
        st.session_state.parameter_sweep_selected_metrics = [
            reverse_metric_map[label]
            for label in chosen_metric_labels
            if label in reverse_metric_map
        ]

        plan_col, run_col = st.columns(2)
        with plan_col:
            if st.button("Preview parameter sweep plan", width="stretch"):
                st.session_state.parameter_sweep_plan = _build_current_parameter_sweep_plan()
                st.session_state.parameter_sweep_result = None
                st.session_state.parameter_sweep_chart_index = 0
                st.session_state.show_parameter_sweep_plan_details = True
                st.session_state.show_parameter_sweep_artifacts = False
                st.session_state.show_parameter_sweep_charts = False
        with run_col:
            current_preview = _build_current_parameter_sweep_plan()
            can_run = bool(
                current_preview.get("success")
                and current_preview.get("generated_sweep_points")
                and not current_preview.get("require_confirmation")
                and current_preview.get("selected_metrics")
            )
            if st.button("Run parameter sweep", width="stretch", disabled=not can_run):
                _run_parameter_sweep_with_status()

        plan = st.session_state.parameter_sweep_plan
        if isinstance(plan, dict):
            if plan.get("success"):
                st.success(
                    f"Parameter sweep plan is ready with {plan.get('total_planned_runs', 0)} planned run(s)."
                )
            else:
                st.warning("Parameter sweep plan is not runnable yet.")
            for error in plan.get("errors", []):
                st.write(f"- Error: {error}")
            for warning in plan.get("warnings", []):
                st.write(f"- Warning: {warning}")
            _render_toggleable_section(
                "Show sweep plan details",
                "show_parameter_sweep_plan_details",
                lambda: _render_parameter_sweep_plan_details(plan),
            )

        result = st.session_state.parameter_sweep_result
        if isinstance(result, dict):
            status = str(result.get("status") or "unknown")
            if status == "completed":
                st.success("Parameter sweep completed successfully.")
            elif status == "partial_success":
                st.warning("Parameter sweep finished with partial failures.")
            elif status == "failed":
                st.warning("Parameter sweep failed.")
            elif status == "invalid_plan":
                st.warning("Parameter sweep was not run because the plan is invalid.")

            summary = result.get("summary", {}) if isinstance(result.get("summary"), dict) else {}
            if summary:
                st.caption(
                    f"Completed runs: {summary.get('total_completed_runs', 0)} | "
                    f"Failed runs: {summary.get('total_failed_runs', 0)} | "
                    f"Output dir: {result.get('output_dir') or 'n/a'}"
                )
            for error in result.get("errors", []):
                if error:
                    st.write(f"- Error: {error}")
            for warning in result.get("warnings", []):
                if warning:
                    st.write(f"- Warning: {warning}")

            if _has_displayable_artifact(result.get("artifacts", {})):
                _render_toggleable_section(
                    "Show parameter sweep artifacts",
                    "show_parameter_sweep_artifacts",
                    lambda: _render_parameter_sweep_artifacts(result),
                )

            chart_entries = _existing_parameter_sweep_chart_entries(result)
            if chart_entries:
                _render_toggleable_section(
                    "Show parameter sweep charts",
                    "show_parameter_sweep_charts",
                    lambda: _render_parameter_sweep_chart_viewer(result),
                )


def _run_manifest_with_status() -> None:
    status_slot = st.empty()
    with st.spinner("Building dataset manifest..."):
        status_slot.info("Scanning dataset folders and collecting manifest rows.")
        st.session_state.latest_manifest_result = build_dataset_manifest(
            dataset_path=str(st.session_state.ablation_dataset_path),
            split=str(st.session_state.ablation_split),
            max_images=int(st.session_state.ablation_max_images),
            sampling_mode=str(st.session_state.ablation_sampling_mode),
            seed=int(st.session_state.ablation_sampling_seed),
            max_images_per_class=(
                int(st.session_state.ablation_max_images_per_class)
                if int(st.session_state.ablation_max_images_per_class) > 0
                else None
            ),
        )
    status_slot.empty()
    st.session_state.latest_ablation_result = None


def _run_plan_with_status() -> None:
    status_slot = st.empty()
    with st.spinner("Planning ablation study..."):
        status_slot.info("Parsing conversational ablation request.")
        st.session_state.ablation_plan = plan_ablation_study(
            message=str(st.session_state.ablation_planner_message or ""),
            context={
                "dataset_path": str(st.session_state.ablation_dataset_path or ""),
                "split": str(st.session_state.ablation_split or "test"),
                "max_images": int(st.session_state.ablation_max_images),
                "sampling_mode": str(st.session_state.ablation_sampling_mode),
                "seed": int(st.session_state.ablation_sampling_seed),
                "max_images_per_class": (
                    int(st.session_state.ablation_max_images_per_class)
                    if int(st.session_state.ablation_max_images_per_class) > 0
                    else None
                ),
            },
        )
    status_slot.empty()


def _apply_plan_to_state(plan_result: dict | None) -> None:
    if not plan_result or not plan_result.get("success"):
        return
    plan = plan_result.get("plan", {})
    if not isinstance(plan, dict):
        return
    if plan.get("dataset_path"):
        st.session_state.ablation_dataset_path = str(plan["dataset_path"])
    if plan.get("split"):
        st.session_state.ablation_split = str(plan["split"])
    if plan.get("max_images") is not None:
        st.session_state.ablation_max_images = int(plan["max_images"])
    if plan.get("sampling_mode"):
        st.session_state.ablation_sampling_mode = str(plan["sampling_mode"])
    if plan.get("seed") is not None:
        st.session_state.ablation_sampling_seed = int(plan["seed"])
    st.session_state.ablation_max_images_per_class = int(plan.get("max_images_per_class") or 0)


def _run_recommendations_with_status() -> None:
    status_slot = st.empty()
    with st.spinner("Loading ablation recommendations..."):
        status_slot.info("Summarizing prior ablation history and recommendation constraints.")
        st.session_state.ablation_history_summary = get_ablation_recommendations()
    status_slot.empty()


def _run_history_with_status() -> None:
    status_slot = st.empty()
    with st.spinner("Loading ablation history..."):
        status_slot.info("Reading persistent ablation history and generating summary artifacts.")
        st.session_state.ablation_history_summary = get_ablation_recommendations().get("history_summary")
        st.session_state.latest_ablation_history_rows = get_ablation_results(limit=50)
        history_rows = st.session_state.latest_ablation_history_rows
        if isinstance(history_rows, dict) and history_rows.get("success"):
            entries = history_rows.get("entries", [])
            latest_run = _latest_ablation_row(entries)
            if latest_run:
                selected_run_id = st.session_state.selected_ablation_run_id
                available_run_ids = {str(item.get("run_id")) for item in entries if item.get("run_id")}
                if not selected_run_id or str(selected_run_id) not in available_run_ids:
                    st.session_state.selected_ablation_run_id = str(latest_run.get("run_id"))
                    st.session_state.latest_ablation_run_view = None
                elif _selected_run_view_is_stale(str(selected_run_id)):
                    st.session_state.latest_ablation_run_view = None
            else:
                st.session_state.latest_ablation_run_view = None
    status_slot.empty()


def _run_latest_result_with_status() -> None:
    status_slot = st.empty()
    with st.spinner("Loading latest ablation result..."):
        status_slot.info("Reading the latest ablation run from persistent history.")
        st.session_state.latest_ablation_run_view = get_latest_ablation_result()
        run = st.session_state.latest_ablation_run_view.get("run", {}) if isinstance(st.session_state.latest_ablation_run_view, dict) else {}
        if run.get("run_id"):
            st.session_state.selected_ablation_run_id = str(run.get("run_id"))
    status_slot.empty()


def _run_compare_results_with_status() -> None:
    status_slot = st.empty()
    with st.spinner("Comparing recent ablation runs..."):
        status_slot.info("Selecting and comparing recent comparable ablation runs.")
        st.session_state.latest_ablation_comparison = compare_ablation_results()
    status_slot.empty()


def _run_export_history_with_status() -> None:
    status_slot = st.empty()
    with st.spinner("Exporting ablation history artifacts..."):
        status_slot.info("Regenerating the canonical CSV, JSON, Markdown, and chart artifacts from JSONL history.")
        st.session_state.latest_ablation_history_export = export_ablation_results()
        st.session_state.ablation_history_summary = get_ablation_recommendations().get("history_summary")
        st.session_state.latest_ablation_history_rows = get_ablation_results(limit=50)
    status_slot.empty()


def _run_ablation_with_status(manifest_path: str) -> None:
    status_slot = st.empty()
    progress_bar = st.progress(0.0, text="Preparing ablation study...")

    def _progress_callback(update: dict) -> None:
        total = max(int(update.get("total", 0)), 1)
        current = max(int(update.get("current", 0)), 0)
        fraction = min(max(current / total, 0.0), 1.0)
        progress_bar.progress(fraction, text=str(update.get("message", "Running sample ablation...")))
        status_slot.info(f"{update.get('stage', 'running')}: {update.get('message', '')}")

    try:
        st.session_state.latest_ablation_result = run_sample_ablation(
            manifest_path=str(manifest_path),
            max_images=int(st.session_state.ablation_max_images),
            progress_callback=_progress_callback,
        )
    finally:
        progress_bar.empty()
        status_slot.empty()


def _run_planned_ablation_with_status() -> None:
    status_slot = st.empty()
    progress_bar = st.progress(0.0, text="Preparing planned ablation...")
    progress_bar.progress(0.1, text="Building manifest from the current ablation plan...")
    try:
        result = run_planned_ablation(
            dataset_path=str(st.session_state.ablation_dataset_path),
            split=str(st.session_state.ablation_split),
            max_images=int(st.session_state.ablation_max_images),
            sampling_mode=str(st.session_state.ablation_sampling_mode),
            seed=int(st.session_state.ablation_sampling_seed),
            max_images_per_class=(
                int(st.session_state.ablation_max_images_per_class)
                if int(st.session_state.ablation_max_images_per_class) > 0
                else None
            ),
        )
        progress_bar.progress(0.9, text="Ablation run finished. Collecting outputs...")
        st.session_state.latest_manifest_result = result.get("manifest")
        st.session_state.latest_ablation_result = result.get("ablation")
        st.session_state.ablation_history_summary = get_ablation_recommendations().get("history_summary")
    finally:
        progress_bar.progress(1.0, text="Done.")
        progress_bar.empty()
        status_slot.empty()


def _render_ablation_panel() -> None:
    st.subheader("Ablation Study")
    st.write(
        "Build a manifest from `dataset_sample/` or a full PlantNet-style dataset path, then run a "
        "sample-based no-retraining ablation study. Use a small max-images value for quick checks on large datasets."
    )

    st.session_state.ablation_dataset_path = st.text_input(
        "Dataset path",
        value=str(st.session_state.ablation_dataset_path),
        help="Supports repo-root dataset_sample/ or an absolute path to plantnet_300K/.",
    )
    st.session_state.ablation_split = st.selectbox(
        "Dataset split",
        options=["test", "train", "val", "validation"],
        index=["test", "train", "val", "validation"].index(str(st.session_state.ablation_split)),
    )
    st.session_state.ablation_max_images = int(
        st.number_input(
            "Max images",
            min_value=1,
            value=int(st.session_state.ablation_max_images),
            step=1,
            help="Useful for quick runs on large datasets.",
        )
    )
    st.session_state.ablation_sampling_mode = st.selectbox(
        "Sampling mode",
        options=["balanced", "random", "sorted"],
        index=["balanced", "random", "sorted"].index(str(st.session_state.ablation_sampling_mode)),
        help="Balanced spreads samples across classes, random shuffles globally, and sorted preserves folder traversal order.",
    )
    st.session_state.ablation_sampling_seed = int(
        st.number_input(
            "Sampling seed",
            value=int(st.session_state.ablation_sampling_seed),
            step=1,
            help="Used for deterministic random or balanced sampling.",
        )
    )
    st.session_state.ablation_max_images_per_class = int(
        st.number_input(
            "Max images per class (0 = no cap)",
            min_value=0,
            value=int(st.session_state.ablation_max_images_per_class),
            step=1,
            help="Optional per-class cap before global sampling is applied.",
        )
    )
    st.session_state.ablation_planner_message = st.text_input(
        "Ask agent to plan ablation",
        value=str(st.session_state.ablation_planner_message),
        help="Examples: run an ablation study; run a quick balanced ablation on the test split; what ablation settings do you recommend?",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Build Manifest", width="stretch"):
            _run_manifest_with_status()
    with col2:
        if st.button("Run Sample Ablation Study", width="stretch"):
            manifest_result = st.session_state.latest_manifest_result
            manifest_path = None
            if isinstance(manifest_result, dict):
                manifest_path = manifest_result.get("manifest_path")
            if not manifest_path:
                st.warning(
                    "Build a manifest first. If no sample dataset exists yet, provide dataset_sample/ or an absolute path to plantnet_300K/."
                )
            else:
                _run_ablation_with_status(str(manifest_path))

    col3, col4 = st.columns(2)
    with col3:
        if st.button("Get Recommendations", width="stretch"):
            _run_recommendations_with_status()
            st.session_state.show_ablation_recommendations = True
    with col4:
        if st.button("Show Ablation History", width="stretch"):
            _run_history_with_status()
            st.session_state.show_ablation_history_summary_table = True

    col5, col6 = st.columns(2)
    with col5:
        if st.button("Plan Ablation", width="stretch"):
            _run_plan_with_status()
            _apply_plan_to_state(st.session_state.ablation_plan)
            st.session_state.show_ablation_plan = True
    with col6:
        if st.button("Run Planned Ablation", width="stretch"):
            _run_planned_ablation_with_status()
            st.session_state.show_ablation_summary = True

    history_state = st.session_state.ablation_history_summary
    recommendation_result = history_state if isinstance(history_state, dict) and "recommendations" in history_state else None
    history_summary = history_state.get("history_summary") if recommendation_result else history_state

    if st.session_state.latest_manifest_result:
        _render_toggleable_section("Show manifest summary", "show_manifest_summary", lambda: _render_manifest_summary(st.session_state.latest_manifest_result))
    if st.session_state.ablation_plan:
        _render_toggleable_section("Show planned ablation settings", "show_ablation_plan", lambda: _render_ablation_plan(st.session_state.ablation_plan))
    if recommendation_result:
        _render_toggleable_section("Show recommendation summary", "show_ablation_recommendations", lambda: _render_ablation_recommendations(recommendation_result))
    if isinstance(history_summary, dict) and history_summary.get("success"):
        latest_runs = history_summary.get("latest_runs", [])
        history_files = history_summary.get("history_files", {})
        if latest_runs:
            _render_toggleable_section(
                "Show ablation history table",
                "show_ablation_history_summary_table",
                lambda: _render_ablation_history_summary_table(history_summary),
            )
        if _has_displayable_artifact(history_files.get("metrics_png") if isinstance(history_files, dict) else None):
            _render_toggleable_section(
                "Show ablation history chart",
                "show_ablation_history_summary_chart",
                lambda: _render_ablation_history_summary_chart(history_summary),
            )
    if st.session_state.latest_ablation_result:
        _render_toggleable_section("Show latest ablation output", "show_ablation_summary", lambda: _render_ablation_summary(st.session_state.latest_ablation_result))
    _render_parameter_sweep_panel()


def _render_ablation_results_panel() -> None:
    st.subheader("Ablation Results")
    st.write(
        "Retrieve prior ablation results from the persistent history store, compare recent runs, and download the canonical CSV, JSON, Markdown, and PNG artifacts."
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Refresh Ablation History", width="stretch"):
            _run_history_with_status()
    with col2:
        if st.button("Show Latest Run", width="stretch"):
            _run_latest_result_with_status()
            st.session_state.show_selected_run_details = True

    col3, col4 = st.columns(2)
    with col3:
        if st.button("Show All Runs Table", width="stretch"):
            _run_history_with_status()
            st.session_state.show_ablation_runs_table = True
    with col4:
        if st.button("Compare Recent Runs", width="stretch"):
            _run_compare_results_with_status()
            st.session_state.show_latest_ablation_comparison = True

    if st.button("Export Ablation History Artifacts", width="stretch"):
        _run_export_history_with_status()
        st.session_state.show_ablation_export_artifacts = True

    history_rows = st.session_state.latest_ablation_history_rows
    if isinstance(history_rows, dict) and history_rows.get("success"):
        rows = history_rows.get("entries", [])
        if rows:
            latest_row = _latest_ablation_row(rows)
            st.caption(
                f"History loaded: {len(rows)} run(s). Latest run ID: {latest_row.get('run_id') if latest_row else 'unknown'}."
            )
            newest_first_rows = _ablation_rows_newest_first(rows)
            option_map = {
                _format_ablation_run_option(item): str(item.get("run_id"))
                for item in newest_first_rows
                if item.get("run_id")
            }
            option_labels = list(option_map.keys())
            selected_run_id = str(st.session_state.selected_ablation_run_id or "")
            if not selected_run_id:
                latest_row = _latest_ablation_row(rows)
                selected_run_id = str(latest_row.get("run_id")) if latest_row else ""
                st.session_state.selected_ablation_run_id = selected_run_id
                st.session_state.latest_ablation_run_view = None
            selected_label = next(
                (label for label, run_id in option_map.items() if run_id == selected_run_id),
                option_labels[0],
            )
            chosen_label = st.selectbox(
                "Select ablation run",
                options=option_labels,
                index=option_labels.index(selected_label),
            )
            chosen_run_id = option_map[chosen_label]
            if str(st.session_state.selected_ablation_run_id or "") != chosen_run_id:
                st.session_state.selected_ablation_run_id = chosen_run_id
                st.session_state.latest_ablation_run_view = None
                _run_selected_result_with_status(chosen_run_id)
                st.session_state.show_selected_run_details = True
            elif (
                not st.session_state.latest_ablation_run_view
                or _selected_run_view_is_stale(chosen_run_id)
            ):
                if _selected_run_view_is_stale(chosen_run_id):
                    st.session_state.latest_ablation_run_view = None
                _run_selected_result_with_status(chosen_run_id)

            default_compare_count = min(3, len(option_labels))
            default_compare_labels = option_labels[:default_compare_count]
            existing_compare_ids = [
                str(run_id)
                for run_id in st.session_state.selected_ablation_compare_run_ids
                if any(mapped_id == str(run_id) for mapped_id in option_map.values())
            ]
            if not existing_compare_ids and len(option_labels) >= 2:
                existing_compare_ids = [option_map[label] for label in default_compare_labels]
                st.session_state.selected_ablation_compare_run_ids = existing_compare_ids
            default_compare_selection = [
                label for label, run_id in option_map.items() if run_id in existing_compare_ids
            ]
            chosen_compare_labels = st.multiselect(
                "Select runs to compare",
                options=option_labels,
                default=default_compare_selection,
            )
            chosen_compare_ids = [option_map[label] for label in chosen_compare_labels]
            st.session_state.selected_ablation_compare_run_ids = chosen_compare_ids
            selected_compare_runs = [
                item for item in rows if str(item.get("run_id")) in set(chosen_compare_ids)
            ]

            table_rows = []
            for item in rows:
                metrics = item.get("metrics", {}) if isinstance(item.get("metrics"), dict) else {}
                table_rows.append(
                    {
                        "run_id": item.get("run_id"),
                        "created_at": item.get("created_at"),
                        "split": item.get("split"),
                        "sampling_mode": item.get("sampling_mode"),
                        "max_images": item.get("max_images"),
                        "num_images_evaluated": item.get("num_images_evaluated"),
                        "num_classes": item.get("num_classes"),
                        "model_agreement_rate": metrics.get("model_agreement_rate"),
                        "tflite_color_stability_rate": metrics.get("tflite_color_stability_rate"),
                        "tflite_top1_accuracy": metrics.get("tflite_top1_accuracy"),
                        "tflite_topk_accuracy": metrics.get("tflite_topk_accuracy"),
                        "hog_top1_accuracy": metrics.get("hog_top1_accuracy"),
                        "hog_topk_accuracy": metrics.get("hog_topk_accuracy"),
                        "output_dir": item.get("output_dir"),
                    }
                )
            if table_rows:
                _render_toggleable_section(
                    "Show all runs table",
                    "show_ablation_runs_table",
                    lambda: (st.write("All runs table"), st.dataframe(table_rows)),
                )
            if st.session_state.latest_ablation_run_view:
                _render_toggleable_section(
                    "Show selected run details",
                    "show_selected_run_details",
                    lambda: _render_ablation_run_view(st.session_state.latest_ablation_run_view),
                )
            if len(selected_compare_runs) >= 2:
                _render_toggleable_section(
                    "Show selected runs comparison",
                    "show_selected_run_comparison",
                    lambda: _render_selected_ablation_comparison(selected_compare_runs),
                )
        else:
            st.info("No ablation runs found yet.")
    elif isinstance(history_rows, dict) and not history_rows.get("success"):
        st.warning(history_rows.get("error") or "Ablation history could not be loaded.")
    else:
        st.info("Load ablation history or run an ablation to enable saved-run views and downloads.")

    if st.session_state.latest_ablation_comparison:
        _render_toggleable_section(
            "Show recent-run comparison",
            "show_latest_ablation_comparison",
            lambda: _render_ablation_comparison(st.session_state.latest_ablation_comparison),
        )
    export_state = st.session_state.latest_ablation_history_export
    if isinstance(export_state, dict) and export_state.get("success") and _has_displayable_artifact(export_state.get("artifacts", {})):
        _render_toggleable_section(
            "Show exported history artifacts",
            "show_ablation_export_artifacts",
            lambda: _render_ablation_export(export_state),
        )


def _reset_image_and_chart_state() -> None:
    st.session_state.current_image_path = None
    st.session_state.uploaded_file_token = None
    st.session_state.latest_tflite_result = None
    st.session_state.latest_hog_result = None
    st.session_state.latest_model_used = None
    for state_key in (
        "show_tflite_chart",
        "show_hog_chart",
    ):
        st.session_state[state_key] = False
    st.rerun()


def main() -> None:
    _init_session_state()
    _render_sidebar()

    st.title("Plant Classification Agent")
    st.write(
        "Upload a plant image and ask questions about classification results, artifact availability, "
        "and metrics. The primary classifier is the TFLite MobileNetV2 model, while HOG+SVM comparison "
        "is shown only when its saved artifacts are available."
    )

    uploader = st.file_uploader(
        "Upload a plant image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=False,
    )

    if st.button("Reset image and chart", width="stretch"):
        _reset_image_and_chart_state()

    if uploader is not None:
        _handle_uploaded_file(uploader)
        current_path = Path(st.session_state.current_image_path)
        st.image(
            str(current_path),
            caption=current_path.name,
            width=int(st.session_state.image_display_width),
        )
    elif st.session_state.current_image_path:
        st.image(
            st.session_state.current_image_path,
            caption=Path(st.session_state.current_image_path).name,
            width=int(st.session_state.image_display_width),
        )

    st.write("Chat")
    _render_chat_history()
    _render_chat_input_area()

    latest_model_used = st.session_state.latest_model_used
    if latest_model_used == "tflite":
        latest_result = st.session_state.latest_tflite_result
        if latest_result and not latest_result.get("success"):
            st.warning(latest_result.get("error") or "TFLite classification is not available.")
    elif latest_model_used == "hog":
        latest_result = st.session_state.latest_hog_result
        if latest_result and not latest_result.get("success"):
            st.warning(latest_result.get("error") or "HOG+SVM classification is not available.")
    if st.session_state.current_image_path:
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            if st.button("Classify with TFLite", width="stretch"):
                _run_direct_tflite_action("Classify this plant image with the TFLite model.")
        with col2:
            if st.button("Classify with HOG+SVM", width="stretch"):
                _run_direct_hog_action("Classify this plant image with the HOG+SVM classical model.")
        with col3:
            if st.button("Compare TFLite vs HOG+SVM", width="stretch"):
                _run_and_render("Compare the TFLite classifier against HOG+SVM for this image.")
        with col4:
            if st.button("Show artifact status", width="stretch"):
                _run_and_render("What artifacts are available?")
        with col5:
            if st.button("Show metrics availability", width="stretch"):
                _run_and_render("What metrics or results are available?")
        with col6:
            if st.button("Check ablation feasibility", width="stretch"):
                _run_and_render("What ablation studies or experiments are feasible right now?")
    else:
        st.info("Upload an image to enable classification and comparison actions.")

    _render_inference_result_sections()
    _render_ablation_panel()
    _render_ablation_results_panel()

    _hide_loading_overlay()


if __name__ == "__main__":
    main()
