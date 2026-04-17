#!/usr/bin/env python3
"""
Train a small image classifier and export TensorFlow Lite for on-device use.

Dataset layout (folder-per-class):
  data/
    fern/
      a.jpg
    pothos/
      b.jpg

The Android app feeds float32 NHWC [1,H,W,3] with RGB in [0, 1] (divide by 255).
This script trains with the same convention — no ImageNet mean/std in the graph.

Writes:
  - plant_classifier.tflite  (copy to app/src/main/assets/ml/)
  - plant_labels_export.txt    (copy lines to app/.../assets/ml/plant_labels.txt)

Class index order = sorted folder names (must match plant_labels.txt line order).
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import tensorflow as tf

# --- Small helpers (logging and timing) ---


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m}m {s}s"


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] {msg}", flush=True)


def _log_tensorflow_devices() -> None:
    """Log TF version, visible GPUs, and enable per-process GPU memory growth."""
    _log(f"TensorFlow {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        _log(
            "No GPU visible to TensorFlow — training will use CPU. "
            "For NVIDIA: install a GPU build of TensorFlow and matching CUDA/cuDNN (see tensorflow.org/install)."
        )
        return
    _log(f"TensorFlow sees {len(gpus)} GPU(s) (training should run on GPU):")
    for i, dev in enumerate(gpus):
        _log(f"  [{i}] {dev.name}")
        try:
            tf.config.experimental.set_memory_growth(dev, True)
        except Exception as ex:
            _log(f"  (could not set memory_growth for {dev.name}: {ex})")
    _log("Set memory_growth=True on each GPU so VRAM is allocated as needed.")


def _dataset_num_batches(ds: tf.data.Dataset) -> Optional[int]:
    # Used for ETA / steps-per-epoch logging; None if TF cannot infer dataset size.
    n = tf.data.experimental.cardinality(ds)
    if n == tf.data.UNKNOWN_CARDINALITY or n == tf.data.INFINITE_CARDINALITY:
        return None
    return int(n.numpy())


class EpochTimingCallback(tf.keras.callbacks.Callback):
    """Logs each epoch duration and ETA for remaining epochs (if training runs to --epochs)."""

    def __init__(self, max_epochs: int) -> None:
        super().__init__()
        self.max_epochs = max_epochs
        self._run_start: Optional[float] = None
        self._epoch_start: Optional[float] = None
        self._epoch_durations: list[float] = []

    def on_train_begin(self, logs=None) -> None:
        self._run_start = time.perf_counter()
        _log(f"Training started (up to {self.max_epochs} epochs; early stopping may end sooner)")

    def on_epoch_begin(self, epoch, logs=None) -> None:
        self._epoch_start = time.perf_counter()
        _log(f"Epoch {epoch + 1}/{self.max_epochs} …")

    def on_epoch_end(self, epoch, logs=None) -> None:
        if self._epoch_start is None:
            return
        epoch_s = time.perf_counter() - self._epoch_start
        self._epoch_durations.append(epoch_s)
        done = epoch + 1
        remaining = self.max_epochs - done
        avg = sum(self._epoch_durations) / len(self._epoch_durations)
        eta = _fmt_duration(avg * remaining) if remaining > 0 else "0s"
        total_elapsed = time.perf_counter() - self._run_start if self._run_start else 0.0
        parts = [
            f"Epoch {done}/{self.max_epochs} finished in {_fmt_duration(epoch_s)}",
            f"(avg epoch {_fmt_duration(avg)})",
        ]
        if remaining > 0:
            parts.append(f"ETA ~{eta} for next {remaining} epoch(s) if pace holds")
        parts.append(f"elapsed total {_fmt_duration(total_elapsed)}")
        if logs:
            loss = logs.get("loss")
            val_loss = logs.get("val_loss")
            acc = logs.get("accuracy")
            val_acc = logs.get("val_accuracy")
            metrics = []
            if loss is not None:
                metrics.append(f"loss={loss:.4f}")
            if acc is not None:
                metrics.append(f"acc={acc:.4f}")
            if val_loss is not None:
                metrics.append(f"val_loss={val_loss:.4f}")
            if val_acc is not None:
                metrics.append(f"val_acc={val_acc:.4f}")
            if metrics:
                parts.append(" | " + ", ".join(metrics))
        _log(" ".join(parts))

    def on_train_end(self, logs=None) -> None:
        if self._run_start is None:
            return
        total = time.perf_counter() - self._run_start
        _log(f"Training finished in {_fmt_duration(total)} ({len(self._epoch_durations)} epoch(s))")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Root with one subfolder per class (folder name = label)",
    )
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument(
        "--out_tflite",
        type=Path,
        default=Path("plant_classifier.tflite"),
    )
    p.add_argument(
        "--out_labels",
        type=Path,
        default=Path("plant_labels_export.txt"),
    )
    p.add_argument("--validation_split", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # --- Resolve paths, discover classes, write label file (order = model class indices) ---
    _log(
        "TFLite export — "
        f"data_dir={args.data_dir!s} img_size={args.img_size} batch_size={args.batch_size} "
        f"epochs={args.epochs} lr={args.learning_rate} validation_split={args.validation_split}"
    )
    _log_tensorflow_devices()

    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"Not a directory: {data_dir}")

    # Same rule as tf.keras image_dataset_from_directory: alphanumeric class order
    class_dirs = sorted(
        d.name for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    if len(class_dirs) < 2:
        raise SystemExit("Need at least 2 class subfolders under --data_dir")

    args.out_labels.write_text(
        "\n".join(class_dirs) + "\n", encoding="utf-8"
    )
    _log(f"Wrote {args.out_labels} ({len(class_dirs)} classes, sorted folder names)")

    # Same seed + split for train/validation so image assignment is consistent across subsets.
    _log("Building training and validation datasets (this can take a while on first run)…")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=args.validation_split,
        subset="training",
        seed=args.seed,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        label_mode="categorical",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=args.validation_split,
        subset="validation",
        seed=args.seed,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        label_mode="categorical",
    )

    # uint8 [0,255] -> float32 [0,1] to match typical on-device TFLite preprocessing
    norm = tf.keras.layers.Rescaling(1.0 / 255.0)
    train_ds = train_ds.map(lambda x, y: (norm(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (norm(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    train_batches = _dataset_num_batches(train_ds)
    val_batches = _dataset_num_batches(val_ds)
    if train_batches is not None:
        _log(
            f"Steps per epoch: train={train_batches} batches, val={val_batches} batches "
            f"(batch_size={args.batch_size})"
        )
    else:
        _log(
            f"Dataset cardinality unknown; steps per epoch follow TensorFlow/Keras progress "
            f"(batch_size={args.batch_size})"
        )

    # --- Model: MobileNetV2 backbone + linear classifier (logits; softmax is inside the loss) ---
    num_classes = len(class_dirs)
    _log(f"Building model: MobileNetV2 + Dense({num_classes}) logits…")
    base = tf.keras.applications.MobileNetV2(
        input_shape=(args.img_size, args.img_size, 3),
        include_top=False,
        weights=None,  # train with [0,1] inputs (matches app); use more data for quality
        pooling="avg",
    )
    base.trainable = True

    inputs = tf.keras.Input(shape=(args.img_size, args.img_size, 3))
    # training=False: batch norm uses inference stats during forward (typical for transfer-style heads)
    x = base(inputs, training=False)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation=None, name="logits")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # Stop when val_accuracy stalls; keep the best weights (not the last epoch).
    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5, restore_best_weights=True
    )
    timing = EpochTimingCallback(max_epochs=args.epochs)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        verbose=1,
        callbacks=[early, timing],
    )

    # Float32 TFLite (input float32 [0,1] NHWC). Skip DEFAULT optimizations to avoid accidental quant without a rep dataset.
    _log("Converting Keras model to TensorFlow Lite (float32)…")
    t0 = time.perf_counter()
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    conv_s = time.perf_counter() - t0
    args.out_tflite.write_bytes(tflite_model)
    _log(f"Wrote {args.out_tflite.resolve()} (convert+write took {_fmt_duration(conv_s)})")
    print()
    _log("Next:")
    print(f"  1) Copy {args.out_tflite.name} -> app/src/main/assets/ml/plant_classifier.tflite")
    print(f"  2) Copy {args.out_labels.name} lines into app/src/main/assets/ml/plant_labels.txt")
    print("  3) Remove plant_classifier.onnx from assets if present (TFLite takes precedence).")


if __name__ == "__main__":
    main()
