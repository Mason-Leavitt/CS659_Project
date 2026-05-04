#!/usr/bin/env python3
"""
Safe hardware and ablation-feasibility inspection helpers.

These helpers do not run training and do not fabricate experiment results.
They only report what appears feasible based on the current environment and
the repository's currently available artifacts.
"""
from __future__ import annotations

import platform
import sys
from typing import Any

from agent.model_imports import ensure_model_project_on_path


def get_hardware_report() -> dict[str, Any]:
    """Return a best-effort hardware/software capability report without crashing."""
    report: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "psutil_installed": False,
        "ram_total_bytes": None,
        "ram_total_gb": None,
        "torch_installed": False,
        "cuda_available": False,
        "gpu_name": None,
        "gpu_vram_bytes": None,
        "gpu_vram_gb": None,
        "tensorflow_installed": False,
        "tensorflow_gpu_visible": False,
        "tensorflow_gpus": [],
        "notes": [],
    }

    try:
        import psutil

        vm = psutil.virtual_memory()
        report["psutil_installed"] = True
        report["ram_total_bytes"] = int(vm.total)
        report["ram_total_gb"] = round(float(vm.total) / (1024 ** 3), 2)
    except Exception as exc:
        report["notes"].append(f"psutil unavailable: {exc}")

    try:
        import torch

        report["torch_installed"] = True
        cuda_available = bool(torch.cuda.is_available())
        report["cuda_available"] = cuda_available
        if cuda_available:
            device_index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device_index)
            report["gpu_name"] = str(props.name)
            report["gpu_vram_bytes"] = int(props.total_memory)
            report["gpu_vram_gb"] = round(float(props.total_memory) / (1024 ** 3), 2)
    except Exception as exc:
        report["notes"].append(f"torch unavailable or incomplete: {exc}")

    try:
        import tensorflow as tf

        report["tensorflow_installed"] = True
        gpus = tf.config.list_physical_devices("GPU")
        report["tensorflow_gpu_visible"] = bool(gpus)
        report["tensorflow_gpus"] = [str(gpu) for gpu in gpus]
    except Exception as exc:
        report["notes"].append(f"tensorflow unavailable or incomplete: {exc}")

    return report


def get_ablation_feasibility() -> dict[str, Any]:
    """
    Return a safe feasibility summary for future ablation work.

    This function does not run experiments. It only reports what appears
    possible now versus what still requires dataset availability, retraining,
    or stronger hardware.
    """
    ensure_model_project_on_path()
    import app_config

    hardware = get_hardware_report()
    artifact_status = app_config.get_artifact_status()
    result_dir_exists = bool(artifact_status.get("result_dir", {}).get("exists"))
    tflite_exists = bool(artifact_status.get("tflite_model", {}).get("exists"))
    default_labels_exist = bool(artifact_status.get("default_labels", {}).get("exists"))
    hog_model_exists = bool(artifact_status.get("hog_svm_model", {}).get("exists"))
    hog_labels_exist = bool(artifact_status.get("hog_svm_labels", {}).get("exists"))
    tensorflow_installed = bool(hardware.get("tensorflow_installed"))

    possible_now: list[str] = []
    requires_dataset: list[str] = []
    requires_training: list[str] = []
    skipped: list[str] = []

    if tflite_exists and default_labels_exist and tensorflow_installed:
        possible_now.append("TFLite inference-only top_k variation")
    else:
        skipped.append(
            "TFLite inference-only top_k variation (requires the TFLite model, labels, and a TensorFlow runtime)"
        )

    possible_now.append("artifact/metrics inspection")

    requires_dataset.extend(
        [
            "HOG+SVM hyperparameter ablations",
            "CNN/TFLite validation comparisons",
        ]
    )

    requires_training.extend(
        [
            "CNN color_correct retraining comparison",
            "CNN img_size retraining comparison",
            "k-fold retraining",
            "HOG+SVM SVM C/kernel retraining",
        ]
    )

    if not result_dir_exists:
        skipped.append("Metrics-based ablation review is limited because result/ does not exist yet")
    if not hog_model_exists or not hog_labels_exist:
        skipped.append("HOG+SVM runtime comparison is limited because saved HOG artifacts are missing")

    gpu_available = bool(hardware.get("cuda_available") or hardware.get("tensorflow_gpu_visible"))
    if not gpu_available:
        skipped.append("Large retraining sweeps should be deferred because no GPU is currently visible")

    explanation_parts = [
        "This feasibility report describes what can be inspected or varied safely right now without running training.",
        "Inference-only checks and artifact inspection are possible immediately when the required saved artifacts exist.",
        "Meaningful hyperparameter or preprocessing ablations still require access to the original dataset.",
        "Retraining-heavy CNN and HOG+SVM comparisons should be treated as future work until dataset access and suitable hardware are available.",
    ]

    return {
        "success": True,
        "hardware": hardware,
        "possible_now": possible_now,
        "requires_dataset": requires_dataset,
        "requires_training": requires_training,
        "skipped": skipped,
        "explanation": " ".join(explanation_parts),
    }
