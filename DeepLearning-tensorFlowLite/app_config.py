#!/usr/bin/env python3
"""
Centralized repository-relative path resolution for the plant classification project.

These helpers avoid relying on the current process working directory so the same
code can run from the repository root, from DeepLearning-tensorFlowLite/, or
from future app entry points such as Streamlit.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent

DEFAULT_TFLITE_MODEL_PATH = PROJECT_DIR / "plant_classifier.tflite"
DEFAULT_SCIENTIFIC_LABELS_PATH = PROJECT_DIR / "plant_labels_scientific.txt"
DEFAULT_EXPORT_LABELS_PATH = PROJECT_DIR / "plant_labels_export.txt"
DEFAULT_UPLOADS_DIR = PROJECT_DIR / "uploads"
DEFAULT_RESULT_DIR = PROJECT_DIR / "result"


def resolve_project_path(*parts: str | Path) -> Path:
    """Resolve a path under DeepLearning-tensorFlowLite/."""
    return PROJECT_DIR.joinpath(*parts).resolve()


def resolve_repo_path(*parts: str | Path) -> Path:
    """Resolve a path under the repository root."""
    return REPO_ROOT.joinpath(*parts).resolve()


def ensure_dir(path: Path | str) -> Path:
    """Create a directory if needed and return its absolute Path."""
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_DIR / p
    p = p.resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_default_labels_path() -> Path:
    """Prefer scientific labels and fall back to exported class ids."""
    if DEFAULT_SCIENTIFIC_LABELS_PATH.is_file():
        return DEFAULT_SCIENTIFIC_LABELS_PATH
    return DEFAULT_EXPORT_LABELS_PATH


def _find_hog_artifact(filename: str) -> Path | None:
    direct = PROJECT_DIR / filename
    if direct.is_file():
        return direct
    if DEFAULT_RESULT_DIR.is_dir():
        matches = sorted(DEFAULT_RESULT_DIR.rglob(filename))
        if matches:
            return matches[-1]
    return None


def _artifact_entry(path: Path, *, exists: bool | None = None) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists() if exists is None else exists,
    }


def get_artifact_status() -> dict[str, Any]:
    """Report current artifact presence without failing if optional files are absent."""
    uploads_exists = DEFAULT_UPLOADS_DIR.exists()
    result_exists = DEFAULT_RESULT_DIR.exists()
    default_labels = get_default_labels_path()
    hog_model = _find_hog_artifact("hog_svm_model.joblib")
    hog_labels = _find_hog_artifact("hog_svm_labels.txt")

    return {
        "project_dir": str(PROJECT_DIR),
        "repo_root": str(REPO_ROOT),
        "tflite_model": _artifact_entry(DEFAULT_TFLITE_MODEL_PATH),
        "scientific_labels": _artifact_entry(DEFAULT_SCIENTIFIC_LABELS_PATH),
        "export_labels": _artifact_entry(DEFAULT_EXPORT_LABELS_PATH),
        "default_labels": _artifact_entry(default_labels),
        "uploads_dir": _artifact_entry(DEFAULT_UPLOADS_DIR, exists=uploads_exists),
        "result_dir": _artifact_entry(DEFAULT_RESULT_DIR, exists=result_exists),
        "hog_svm_model": {
            "path": str(hog_model) if hog_model is not None else str(PROJECT_DIR / "hog_svm_model.joblib"),
            "exists": hog_model is not None,
        },
        "hog_svm_labels": {
            "path": str(hog_labels) if hog_labels is not None else str(PROJECT_DIR / "hog_svm_labels.txt"),
            "exists": hog_labels is not None,
        },
    }
