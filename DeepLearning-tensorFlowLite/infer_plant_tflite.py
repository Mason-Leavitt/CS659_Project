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
from pathlib import Path

import numpy as np
import tensorflow as tf

import color_correction

# --- Preprocessing and postprocessing helpers (match training script) ---


def _read_labels(path: Path) -> list[str]:
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
    raw = tf.io.read_file(str(path))
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [height, width])
    return tf.cast(image, tf.float32).numpy() / 255.0


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
        choices=color_correction.COLOR_METHODS,
        help="Must match training (see train_export_tflite.py --color_correct)",
    )
    args = p.parse_args()

    if not args.model.is_file():
        raise SystemExit(f"Model not found: {args.model}")
    if not args.image.is_file():
        raise SystemExit(f"Image not found: {args.image}")

    labels = _read_labels(args.labels)
    if not labels:
        raise SystemExit(f"No labels in {args.labels}")

    # Load TFLite and read tensor layouts (dtype may require uint8 input).
    interpreter = tf.lite.Interpreter(model_path=str(args.model.resolve()))
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]

    in_shape = tuple(in_det["shape"])
    # Expect rank-4: NHWC [1,H,W,3] or NCHW [1,3,H,W]. Channel at axis 1 => NCHW, else NHWC.
    if len(in_shape) != 4:
        raise SystemExit(f"Unexpected input rank: {in_shape}")

    h, w = args.img_size, args.img_size
    if in_shape[1] == 3:
        nchw = True
        h, w = int(in_shape[2]), int(in_shape[3])
    else:
        nchw = False
        h, w = int(in_shape[1]), int(in_shape[2])

    hwc = _load_image_rgb01(args.image, h, w)
    hwc_b = tf.expand_dims(tf.constant(hwc, dtype=tf.float32), 0)
    hwc_b = color_correction.apply_color_rgb01_bhwc(hwc_b, args.color_correct)
    hwc = hwc_b[0].numpy()
    x = np.expand_dims(hwc, axis=0)

    if nchw:
        x = np.transpose(x, (0, 3, 1, 2))

    if in_det.get("dtype") == np.uint8:
        x = (x * 255.0).astype(np.uint8)

    interpreter.set_tensor(in_det["index"], x)
    interpreter.invoke()
    logits = interpreter.get_tensor(out_det["index"])[0]
    probs = _softmax(logits)

    # Label line i must be class i (same order as train_export_tflite label file).
    n_class = probs.shape[0]
    if len(labels) != n_class:
        print(
            f"WARNING: {len(labels)} labels but model outputs {n_class} classes — "
            "line order must match training export.\n"
        )

    k = min(args.top_k, n_class)
    top_idx = np.argsort(-probs)[:k]

    print(f"Image: {args.image}")
    print(f"Model: {args.model.name}  |  labels: {args.labels.name}  |  top-{k}\n")
    rank = 1
    for i in top_idx:
        pct = 100.0 * float(probs[i])
        name = labels[i] if i < len(labels) else f"class_{i}"
        print(f"  {rank:2}. {pct:6.2f}%  {name}")
        rank += 1


if __name__ == "__main__":
    main()
