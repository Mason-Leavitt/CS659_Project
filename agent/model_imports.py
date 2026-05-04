#!/usr/bin/env python3
"""
Helpers for importing model-layer modules from DeepLearning-tensorFlowLite/.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PROJECT_DIR = REPO_ROOT / "DeepLearning-tensorFlowLite"


def ensure_model_project_on_path() -> Path:
    """Ensure the model project directory is importable regardless of cwd."""
    model_dir = MODEL_PROJECT_DIR.resolve()
    model_dir_str = str(model_dir)
    if model_dir_str not in sys.path:
        sys.path.insert(0, model_dir_str)
    return model_dir
