#!/usr/bin/env python3
"""
Run saved HOG+SVM inference on a single image.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any
import warnings

import joblib
import numpy as np
import sklearn
from sklearn.exceptions import InconsistentVersionWarning

import app_config
from hog_svm_utils import load_hog_labels, prepare_hog_features

_DEFAULT_ORIENTATIONS = 9
_DEFAULT_PIXELS_PER_CELL = 16
_DEFAULT_CELLS_PER_BLOCK = 2
_DEFAULT_BLOCK_NORM = "L2-Hys"


def load_plantnet_id_name_map() -> dict[str, str]:
    export_path = app_config.DEFAULT_EXPORT_LABELS_PATH
    scientific_path = app_config.DEFAULT_SCIENTIFIC_LABELS_PATH
    if not export_path.is_file() or not scientific_path.is_file():
        return {}

    export_labels = [line.strip() for line in export_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    scientific_labels = [line.strip() for line in scientific_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(export_labels) != len(scientific_labels):
        return {}

    return {
        plantnet_id: scientific_name
        for plantnet_id, scientific_name in zip(export_labels, scientific_labels)
        if plantnet_id and scientific_name
    }


def _resolve_default_hog_model_path() -> Path | None:
    status = app_config.get_artifact_status()
    model_info = status.get("hog_svm_model", {})
    if model_info.get("exists"):
        return Path(str(model_info["path"]))
    return None


def _resolve_default_hog_labels_path() -> Path | None:
    status = app_config.get_artifact_status()
    labels_info = status.get("hog_svm_labels", {})
    if labels_info.get("exists"):
        return Path(str(labels_info["path"]))
    return None


def _prediction_template() -> dict[str, float | str | None]:
    return {
        "rank": None,
        "label": None,
        "display_label": None,
        "confidence": None,
        "score": None,
    }


def _display_label_for(label: str, id_name_map: dict[str, str]) -> str:
    if label.isdigit():
        return id_name_map.get(label, label)
    return label


def _unwrap_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _candidate_estimator_objects(model: Any) -> list[Any]:
    candidates: list[Any] = []

    def _append(value: Any) -> None:
        if value is not None and value not in candidates:
            candidates.append(value)

    _append(model)
    if isinstance(model, dict):
        for key in ("model", "pipeline", "estimator", "classifier", "clf"):
            _append(model.get(key))

    named_steps = getattr(model, "named_steps", None)
    if isinstance(named_steps, dict) and named_steps:
        _append(next(reversed(named_steps.values())))

    steps = getattr(model, "steps", None)
    if isinstance(steps, list) and steps:
        last_step = steps[-1]
        if isinstance(last_step, tuple) and len(last_step) >= 2:
            _append(last_step[1])

    return candidates


def _resolve_prediction_model(model: Any) -> Any:
    for candidate in _candidate_estimator_objects(model):
        if callable(getattr(candidate, "predict", None)):
            return candidate
    return model


def _get_estimator_classes(model: Any) -> list[Any] | None:
    for candidate in _candidate_estimator_objects(model):
        classes = getattr(candidate, "classes_", None)
        if classes is None:
            continue
        array = np.asarray(classes).reshape(-1)
        if array.size > 0:
            return [_unwrap_scalar(value) for value in array.tolist()]
    return None


def _class_value_to_label(class_value: Any, labels: list[str]) -> str:
    raw_value = _unwrap_scalar(class_value)
    text_value = str(raw_value).strip()
    if text_value and text_value in labels:
        return text_value
    # Legacy fallback: some saved estimators may still expose integer-encoded classes
    # whose values line up with the external labels-file order.
    if isinstance(raw_value, (int, np.integer)) and 0 <= int(raw_value) < len(labels):
        return labels[int(raw_value)]
    if text_value:
        return text_value
    return f"class_{raw_value}"


def _scores_from_binary_decision_function(scores: np.ndarray, estimator_classes: list[Any] | None) -> np.ndarray:
    flat = np.asarray(scores, dtype=np.float64).reshape(-1)
    if flat.size == 1 and estimator_classes is not None and len(estimator_classes) == 2:
        positive_score = float(flat[0])
        return np.asarray([-positive_score, positive_score], dtype=np.float64)
    return flat


def _rank_scores_with_classes(
    scores: np.ndarray,
    *,
    labels: list[str],
    estimator_classes: list[Any] | None,
    top_k: int,
    id_name_map: dict[str, str],
    score_type: str,
) -> list[dict[str, Any]]:
    flat_scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if score_type == "decision_function":
        flat_scores = _scores_from_binary_decision_function(flat_scores, estimator_classes)
    if flat_scores.size <= 0:
        return []

    predictions: list[dict[str, Any]] = []
    top_indices = np.argsort(flat_scores)[::-1][: min(max(int(top_k), 1), flat_scores.size)]
    for rank, class_index in enumerate(top_indices, start=1):
        column_index = int(class_index)
        if estimator_classes is not None and 0 <= column_index < len(estimator_classes):
            class_label = _class_value_to_label(estimator_classes[column_index], labels)
        else:
            # Legacy fallback: if classes_ is missing, preserve the older index-based mapping.
            class_label = labels[column_index] if 0 <= column_index < len(labels) else f"class_{column_index}"
        predictions.append(
            {
                "rank": rank,
                "label": class_label,
                "display_label": _display_label_for(str(class_label), id_name_map),
                "confidence": float(flat_scores[column_index]) if score_type == "probability" else None,
                "score": float(flat_scores[column_index]) if score_type == "decision_function" else None,
            }
        )
    return predictions


def run_hog_svm_inference(
    image_path: str | Path,
    model_path: str | Path | None = None,
    labels_path: str | Path | None = None,
    img_size: int = 224,
    top_k: int = 5,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "success": False,
        "image_path": str(Path(image_path)),
        "model_path": None,
        "labels_path": None,
        "prediction": _prediction_template(),
        "predictions": [],
        "score_type": "label_only",
        "preprocessing": {
            "img_size": int(img_size),
            "grayscale": True,
            "hog": {
                "orientations": _DEFAULT_ORIENTATIONS,
                "pixels_per_cell": _DEFAULT_PIXELS_PER_CELL,
                "cells_per_block": _DEFAULT_CELLS_PER_BLOCK,
                "block_norm": _DEFAULT_BLOCK_NORM,
            },
        },
        "warning": None,
        "error": None,
    }

    try:
        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        resolved_model_path = Path(model_path) if model_path is not None else _resolve_default_hog_model_path()
        if resolved_model_path is None:
            raise FileNotFoundError(
                "HOG+SVM model artifact not available. Expected hog_svm_model.joblib in "
                "DeepLearning-tensorFlowLite/ or under result/."
            )
        if not resolved_model_path.is_file():
            raise FileNotFoundError(f"HOG+SVM model not found: {resolved_model_path}")

        resolved_labels_path = Path(labels_path) if labels_path is not None else _resolve_default_hog_labels_path()
        if resolved_labels_path is None:
            raise FileNotFoundError(
                "HOG+SVM labels artifact not available. Expected hog_svm_labels.txt in "
                "DeepLearning-tensorFlowLite/ or under result/."
            )
        if not resolved_labels_path.is_file():
            raise FileNotFoundError(f"HOG+SVM labels not found: {resolved_labels_path}")

        result["model_path"] = str(resolved_model_path)
        result["labels_path"] = str(resolved_labels_path)

        labels = load_hog_labels(resolved_labels_path)
        id_name_map = load_plantnet_id_name_map()
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", InconsistentVersionWarning)
            pipe = joblib.load(resolved_model_path)
        for caught in caught_warnings:
            if issubclass(caught.category, InconsistentVersionWarning):
                result["warning"] = (
                    "HOG+SVM artifact version mismatch: this model was saved with a different "
                    f"scikit-learn version than the current environment ({sklearn.__version__}). "
                    "Inference may still work, but results can be unreliable unless you install "
                    "the matching scikit-learn version or retrain/export the artifact in this environment."
                )
            else:
                warnings.warn(
                    str(caught.message),
                    category=caught.category,
                    stacklevel=2,
                )

        inference_model = _resolve_prediction_model(pipe)
        estimator_classes = _get_estimator_classes(pipe)

        features = prepare_hog_features(
            image_path=image_path,
            img_size=img_size,
            orientations=_DEFAULT_ORIENTATIONS,
            pixels_per_cell=_DEFAULT_PIXELS_PER_CELL,
            cells_per_block=_DEFAULT_CELLS_PER_BLOCK,
            block_norm=_DEFAULT_BLOCK_NORM,
        )
        x = np.expand_dims(features, axis=0)

        pred_raw = inference_model.predict(x)[0]
        pred_label = _class_value_to_label(pred_raw, labels)

        prediction = _prediction_template()
        prediction["rank"] = 1
        prediction["label"] = pred_label
        prediction["display_label"] = _display_label_for(pred_label, id_name_map)

        predictions: list[dict[str, Any]] = []
        max_k = max(int(top_k), 1)

        predict_proba = getattr(inference_model, "predict_proba", None)
        if callable(predict_proba):
            probs = np.asarray(predict_proba(x)[0], dtype=np.float64).reshape(-1)
            if probs.size > 0:
                result["score_type"] = "probability"
                predictions = _rank_scores_with_classes(
                    probs,
                    labels=labels,
                    estimator_classes=estimator_classes,
                    top_k=max_k,
                    id_name_map=id_name_map,
                    score_type="probability",
                )
        else:
            decision_function = getattr(inference_model, "decision_function", None)
            if callable(decision_function):
                scores = np.asarray(decision_function(x), dtype=np.float64)
                if scores.size > 0:
                    result["score_type"] = "decision_function"
                    predictions = _rank_scores_with_classes(
                        scores,
                        labels=labels,
                        estimator_classes=estimator_classes,
                        top_k=max_k,
                        id_name_map=id_name_map,
                        score_type="decision_function",
                    )

        if predictions:
            result["predictions"] = predictions
            result["prediction"] = predictions[0]
        else:
            prediction["display_label"] = _display_label_for(pred_label, id_name_map)
            result["predictions"] = [prediction]
            result["prediction"] = prediction
        result["success"] = True
        return result
    except Exception as exc:
        result["error"] = str(exc)
        return result


def _format_cli_output(result: dict[str, Any]) -> str:
    if not result.get("success"):
        return f"ERROR: {result.get('error') or 'Inference failed.'}"

    pred = result["prediction"]
    lines = []
    if result.get("warning"):
        lines.append(f"WARNING: {result['warning']}")
    lines.extend([
        f"Image: {result['image_path']}",
        f"Model: {Path(str(result['model_path'])).name}",
        f"Labels: {Path(str(result['labels_path'])).name}",
        f"Prediction: {pred.get('display_label') or pred.get('label')}",
    ])
    if pred.get("display_label") and pred.get("label") and pred.get("display_label") != pred.get("label"):
        lines.append(f"Original label/id: {pred.get('label')}")
    if pred.get("confidence") is not None:
        lines.append(f"Confidence: {100.0 * float(pred['confidence']):.2f}%")
    if pred.get("score") is not None:
        lines.append(f"Score: {float(pred['score']):.6f}")
    predictions = result.get("predictions", [])
    if len(predictions) > 1:
        score_type = result.get("score_type", "label_only")
        if score_type == "decision_function":
            lines.append("Scores are SVM decision values, not probabilities.")
        lines.append("Top predictions:")
        for item in predictions:
            label_text = str(item.get("display_label") or item.get("label") or "")
            if item.get("display_label") and item.get("label") and item.get("display_label") != item.get("label"):
                label_text += f" (PlantNet ID: {item.get('label')})"
            if item.get("confidence") is not None:
                value_text = f"{100.0 * float(item['confidence']):.2f}%"
            elif item.get("score") is not None:
                value_text = f"{float(item['score']):.6f}"
            else:
                value_text = "n/a"
            lines.append(f"  {item.get('rank')}. {label_text} - {value_text}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Run HOG+SVM plant classifier on one image")
    p.add_argument("--image", type=Path, required=True, help="JPEG/PNG etc.")
    p.add_argument("--model", type=Path, default=None, help="Path to hog_svm_model.joblib")
    p.add_argument("--labels", type=Path, default=None, help="Path to hog_svm_labels.txt")
    p.add_argument("--img_size", type=int, default=224, help="Resize image before HOG extraction")
    p.add_argument("--top_k", type=int, default=5, help="Number of top HOG+SVM predictions to return")
    args = p.parse_args()

    result = run_hog_svm_inference(
        image_path=args.image,
        model_path=args.model,
        labels_path=args.labels,
        img_size=args.img_size,
        top_k=args.top_k,
    )

    output = _format_cli_output(result)
    if result.get("success"):
        print(output)
        return

    print(output, file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
