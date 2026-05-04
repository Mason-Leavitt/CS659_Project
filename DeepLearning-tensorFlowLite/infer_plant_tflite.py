#!/usr/bin/env python3
"""
Run the exported plant_classifier.tflite on a single image and print top-k species with %.

Preprocessing matches train_export_tflite.py (RGB float32 [0,1], NHWC, optional --color_correct):
  float32 NHWC [1,H,W,3], RGB, values in [0, 1] (divide by 255).

Example:
  python infer_plant_tflite.py \\
    --model plant_classifier.tflite \\
    --labels plant_labels_scientific.txt \\
    --image /path/to/photo.jpg \\
    --top_k 10
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import warnings

# Reduce TensorFlow C++ log noise and oneDNN informational messages during inference.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# Suppress the LiteRT migration warning while keeping tf.lite.Interpreter for compatibility.
warnings.filterwarnings(
    "ignore",
    message=".*Please use the LiteRT interpreter.*",
    category=UserWarning,
)

import numpy as np
try:
    import tensorflow as tf
except ModuleNotFoundError:
    tf = None

try:
    import color_correction
except ModuleNotFoundError:
    color_correction = None

COLOR_METHODS = tuple(color_correction.COLOR_METHODS) if color_correction is not None else ("none", "gray_world", "max_rgb")

# --- Preprocessing and postprocessing helpers (match training script) ---


def load_labels(labels_path: str | Path) -> list[str]:
    path = Path(labels_path)
    lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        t = line.strip()
        if t and not t.startswith("#"):
            lines.append(t)
    return lines


def _softmax(logits: np.ndarray) -> np.ndarray:
    # Numerically stable softmax (subtract max before exp).
    z = logits.astype(np.float64)
    z -= np.max(z)
    e = np.exp(z)
    return (e / np.sum(e)).astype(np.float32)


def _load_image_rgb01(path: Path, height: int, width: int) -> np.ndarray:
    """HWC float32 RGB in [0, 1]. Uses TensorFlow only (no Pillow required)."""
    if tf is None:
        raise RuntimeError("TensorFlow is not installed, so TFLite classification is unavailable in this environment.")
    raw = tf.io.read_file(str(path))
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [height, width])
    return tf.cast(image, tf.float32).numpy() / 255.0


def prepare_image_for_tflite(
    image_path: str | Path,
    input_details: dict,
    color_correct: str = "none",
) -> np.ndarray:
    """Prepare a single image tensor for the current TFLite model input contract."""
    if color_correction is None:
        raise RuntimeError("TensorFlow is not installed, so TFLite classification is unavailable in this environment.")
    in_shape = tuple(input_details["shape"])
    if len(in_shape) != 4:
        raise ValueError(f"Unexpected input rank: {in_shape}")

    if in_shape[1] == 3:
        nchw = True
        height, width = int(in_shape[2]), int(in_shape[3])
    else:
        nchw = False
        height, width = int(in_shape[1]), int(in_shape[2])

    hwc = _load_image_rgb01(Path(image_path), height, width)
    hwc_b = tf.expand_dims(tf.constant(hwc, dtype=tf.float32), 0)
    hwc_b = color_correction.apply_color_rgb01_bhwc(hwc_b, color_correct)
    x = np.expand_dims(hwc_b[0].numpy(), axis=0)

    if nchw:
        x = np.transpose(x, (0, 3, 1, 2))

    if input_details.get("dtype") == np.uint8:
        x = (x * 255.0).astype(np.uint8)

    return x


def run_tflite_inference(
    model_path: str | Path,
    image_path: str | Path,
    labels_path: str | Path,
    top_k: int = 5,
    color_correct: str = "none",
) -> dict:
    """Run TFLite inference and return structured top-k predictions."""
    model_path = Path(model_path)
    image_path = Path(image_path)
    labels_path = Path(labels_path)

    result: dict = {
        "success": False,
        "image_path": str(image_path),
        "model_path": str(model_path),
        "labels_path": str(labels_path),
        "top_k": int(top_k),
        "color_correct": color_correct,
        "input_shape": [],
        "predictions": [],
        "error": None,
        "num_classes": 0,
        "label_count": 0,
        "warning": None,
    }

    try:
        if tf is None:
            raise RuntimeError("TensorFlow is not installed, so TFLite classification is unavailable in this environment.")

        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not labels_path.is_file():
            raise FileNotFoundError(f"Labels not found: {labels_path}")

        labels = load_labels(labels_path)
        if not labels:
            raise ValueError(f"No labels in {labels_path}")

        interpreter = tf.lite.Interpreter(model_path=str(model_path.resolve()))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        result["input_shape"] = [int(v) for v in input_details["shape"]]

        x = prepare_image_for_tflite(
            image_path=image_path,
            input_details=input_details,
            color_correct=color_correct,
        )

        interpreter.set_tensor(input_details["index"], x)
        interpreter.invoke()
        logits = interpreter.get_tensor(output_details["index"])[0]
        probs = _softmax(logits)

        num_classes = int(probs.shape[0])
        result["num_classes"] = num_classes
        result["label_count"] = len(labels)

        if len(labels) != num_classes:
            result["warning"] = (
                f"WARNING: {len(labels)} labels but model outputs {num_classes} classes — "
                "line order must match training export."
            )

        k = min(max(int(top_k), 1), num_classes)
        result["top_k"] = k
        top_idx = np.argsort(-probs)[:k]

        predictions: list[dict] = []
        rank = 1
        for i in top_idx:
            predictions.append(
                {
                    "rank": rank,
                    "label": labels[i] if i < len(labels) else f"class_{i}",
                    "probability": float(probs[i]),
                }
            )
            rank += 1

        result["predictions"] = predictions
        result["success"] = True
        return result
    except Exception as exc:
        result["error"] = str(exc)
        return result


def format_predictions_for_cli(result: dict) -> str:
    """Render structured inference results using the existing CLI output format."""
    if not result.get("success"):
        error = result.get("error") or "Inference failed."
        return f"ERROR: {error}"

    image_path = result["image_path"]
    model_name = Path(result["model_path"]).name
    labels_name = Path(result["labels_path"]).name
    top_k = int(result["top_k"])

    lines: list[str] = []
    warning = result.get("warning")
    if warning:
        lines.append(str(warning))
        lines.append("")

    lines.append(f"Image: {image_path}")
    lines.append(f"Model: {model_name}  |  labels: {labels_name}  |  top-{top_k}")
    lines.append("")
    for pred in result.get("predictions", []):
        rank = int(pred["rank"])
        pct = 100.0 * float(pred["probability"])
        label = str(pred["label"])
        lines.append(f"  {rank:2}. {pct:6.2f}%  {label}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Test TFLite plant classifier on one image")
    p.add_argument("--model", type=Path, required=True, help="plant_classifier.tflite")
    p.add_argument("--labels", type=Path, required=True, help="plant_labels.txt (scientific names OK)")
    p.add_argument("--image", type=Path, required=True, help="JPEG/PNG etc.")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--img_size", type=int, default=224, help="Must match training export")
    p.add_argument(
        "--color_correct",
        type=str,
        default="none",
        choices=COLOR_METHODS,
        help="Must match training (see train_export_tflite.py --color_correct)",
    )
    args = p.parse_args()

    result = run_tflite_inference(
        model_path=args.model,
        image_path=args.image,
        labels_path=args.labels,
        top_k=args.top_k,
        color_correct=args.color_correct,
    )

    output = format_predictions_for_cli(result)
    if result.get("success"):
        print(output)
        return

    print(output, file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
