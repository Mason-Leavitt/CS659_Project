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

With --k_folds > 1, stratified k-fold cross-validation runs first; then a final
model is trained for export (see --no_final_retrain).

Writes:
  - plant_classifier.tflite  (copy to app/src/main/assets/ml/)
  - plant_labels_export.txt    (copy lines to app/.../assets/ml/plant_labels.txt)
  - result/<timestamp>/   (CSV + sklearn metrics + plots; omit with --no_metric_logs)

Class index order = sorted folder names (must match plant_labels.txt line order).
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf

import color_correction
from metrics_logging import ValidationMetricsCallback

# --- Small helpers (logging and timing) ---

_IMG_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"})


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


def _collect_paths_and_labels(data_dir: Path, class_names: list[str]) -> tuple[list[str], list[int]]:
    paths: list[str] = []
    labels: list[int] = []
    for i, name in enumerate(class_names):
        sub = data_dir / name
        for f in sorted(sub.rglob("*")):
            if f.is_file() and f.suffix.lower() in _IMG_EXTS:
                paths.append(str(f.resolve()))
                labels.append(i)
    if not paths:
        raise SystemExit("No image files found under class subfolders (check extensions).")
    return paths, labels


def _stratified_k_fold_splits(
    paths: list[str],
    labels: list[int],
    k: int,
    num_classes: int,
    seed: int,
) -> list[tuple[list[str], list[str], list[int], list[int]]]:
    """Per fold: train/val path lists and int labels (stratified by class)."""
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[str]] = {c: [] for c in range(num_classes)}
    for p, y in zip(paths, labels):
        by_class[y].append(p)

    folds: list[tuple[list[str], list[str], list[int], list[int]]] = []
    for fold_idx in range(k):
        train_paths: list[str] = []
        val_paths: list[str] = []
        train_labels: list[int] = []
        val_labels: list[int] = []
        for y in range(num_classes):
            plist = list(by_class[y])
            if not plist:
                continue
            rng.shuffle(plist)
            arr = np.array(plist, dtype=object)
            splits = np.array_split(arr, k)
            val_chunk = splits[fold_idx]
            train_chunks = [splits[j] for j in range(k) if j != fold_idx]
            tr = np.concatenate(train_chunks) if train_chunks else np.array([], dtype=object)
            for path in val_chunk:
                val_paths.append(str(path))
                val_labels.append(y)
            for path in tr:
                train_paths.append(str(path))
                train_labels.append(y)
        folds.append((train_paths, val_paths, train_labels, val_labels))
    return folds


def _make_dataset_from_paths(
    paths: list[str],
    labels: list[int],
    *,
    batch_size: int,
    img_size: int,
    num_classes: int,
    shuffle: bool,
    shuffle_seed: int,
) -> tf.data.Dataset:
    path_ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        img = tf.io.read_file(path)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, [img_size, img_size])
        y = tf.one_hot(tf.cast(label, tf.int32), num_classes)
        return img, y

    ds = path_ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle and paths:
        ds = ds.shuffle(buffer_size=min(len(paths), 10_000), seed=shuffle_seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size)


def _attach_preprocess(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    cc: str,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    norm = tf.keras.layers.Rescaling(1.0 / 255.0)

    def _preprocess_batch(x: tf.Tensor, y: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        x = norm(x)
        x = color_correction.apply_color_rgb01_bhwc(x, cc)
        return x, y

    train_ds = train_ds.map(_preprocess_batch, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(_preprocess_batch, num_parallel_calls=tf.data.AUTOTUNE)
    return train_ds.prefetch(tf.data.AUTOTUNE), val_ds.prefetch(tf.data.AUTOTUNE)


def _build_model(num_classes: int, img_size: int) -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights=None,  # train with [0,1] inputs (matches app); use more data for quality
        pooling="avg",
    )
    base.trainable = True
    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    # training=False: batch norm uses inference stats during forward (typical for transfer-style heads)
    x = base(inputs, training=False)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation=None, name="logits")(x)
    return tf.keras.Model(inputs, outputs)


def _export_tflite(model: tf.keras.Model, out_path: Path) -> None:
    _log("Converting Keras model to TensorFlow Lite (float32)…")
    t0 = time.perf_counter()
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    conv_s = time.perf_counter() - t0
    out_path.write_bytes(tflite_model)
    _log(f"Wrote {out_path.resolve()} (convert+write took {_fmt_duration(conv_s)})")


class EpochTimingCallback(tf.keras.callbacks.Callback):
    """Logs each epoch duration and ETA for remaining epochs (if training runs to --epochs)."""

    def __init__(self, max_epochs: int, log_prefix: str = "") -> None:
        super().__init__()
        self.max_epochs = max_epochs
        self.log_prefix = log_prefix
        self._run_start: Optional[float] = None
        self._epoch_start: Optional[float] = None
        self._epoch_durations: list[float] = []

    def on_train_begin(self, logs=None) -> None:
        self._run_start = time.perf_counter()
        p = self.log_prefix
        _log(f"{p}Training started (up to {self.max_epochs} epochs; early stopping may end sooner)")

    def on_epoch_begin(self, epoch, logs=None) -> None:
        self._epoch_start = time.perf_counter()
        p = self.log_prefix
        _log(f"{p}Epoch {epoch + 1}/{self.max_epochs} …")

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
        p = self.log_prefix
        parts = [
            f"{p}Epoch {done}/{self.max_epochs} finished in {_fmt_duration(epoch_s)}",
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
        p = self.log_prefix
        _log(f"{p}Training finished in {_fmt_duration(total)} ({len(self._epoch_durations)} epoch(s))")


def _train_one(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    *,
    num_classes: int,
    img_size: int,
    learning_rate: float,
    epochs: int,
    log_prefix: str,
    extra_callbacks: Optional[list[tf.keras.callbacks.Callback]] = None,
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    _log(f"{log_prefix}Building model: MobileNetV2 + Dense({num_classes}) logits…")
    model = _build_model(num_classes, img_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5, restore_best_weights=True
    )
    timing = EpochTimingCallback(max_epochs=epochs, log_prefix=log_prefix)
    cbs: list[tf.keras.callbacks.Callback] = [early, timing]
    if extra_callbacks:
        cbs.extend(extra_callbacks)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=1,
        callbacks=cbs,
    )
    return model, history


def _log_dataset_steps(train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, batch_size: int) -> None:
    train_batches = _dataset_num_batches(train_ds)
    val_batches = _dataset_num_batches(val_ds)
    if train_batches is not None:
        _log(
            f"Steps per epoch: train={train_batches} batches, val={val_batches} batches "
            f"(batch_size={batch_size})"
        )
    else:
        _log(
            f"Dataset cardinality unknown; steps per epoch follow TensorFlow/Keras progress "
            f"(batch_size={batch_size})"
        )


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
    p.add_argument(
        "--color_correct",
        type=str,
        default="none",
        choices=color_correction.COLOR_METHODS,
        help="Per-image illuminant correction after RGB [0,1] rescaling (use same at inference)",
    )
    p.add_argument(
        "--k_folds",
        type=int,
        default=1,
        help="Stratified k-fold CV: 1 = single train/val split (default); "
        ">=2 runs k folds then a final export train unless --no_final_retrain.",
    )
    p.add_argument(
        "--no_final_retrain",
        action="store_true",
        help="After k-fold CV only: export the best fold’s weights instead of retraining on the full split.",
    )
    p.add_argument(
        "--log_dir",
        type=Path,
        default=None,
        help="Directory for metrics CSV, JSON summary, plots, and classification report. "
        "Default: result/<UTC timestamp> when --no_metric_logs is not set.",
    )
    p.add_argument(
        "--no_metric_logs",
        action="store_true",
        help="Disable sklearn metrics CSV/plots (accuracy/F1/recall/ROC/correlation figures).",
    )
    args = p.parse_args()

    if args.k_folds < 1:
        raise SystemExit("--k_folds must be >= 1")
    if args.k_folds > 1 and args.validation_split <= 0:
        raise SystemExit("--validation_split must be > 0 for final retrain after k-fold")

    # --- Resolve paths, discover classes, write label file (order = model class indices) ---
    _log(
        "TFLite export — "
        f"data_dir={args.data_dir!s} img_size={args.img_size} batch_size={args.batch_size} "
        f"epochs={args.epochs} lr={args.learning_rate} validation_split={args.validation_split} "
        f"color_correct={args.color_correct} k_folds={args.k_folds}"
    )
    _log_tensorflow_devices()

    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"Not a directory: {data_dir}")

    class_dirs = sorted(
        d.name for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    if len(class_dirs) < 2:
        raise SystemExit("Need at least 2 class subfolders under --data_dir")

    num_classes = len(class_dirs)
    args.out_labels.write_text("\n".join(class_dirs) + "\n", encoding="utf-8")
    _log(f"Wrote {args.out_labels} ({num_classes} classes, sorted folder names)")

    base_log_dir: Optional[Path] = None
    if not args.no_metric_logs:
        if args.log_dir is not None:
            base_log_dir = args.log_dir.resolve()
        else:
            base_log_dir = (
                Path("result")
                / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
            ).resolve()
        base_log_dir.mkdir(parents=True, exist_ok=True)
        _log(f"Metric logs (CSV, plots, reports): {base_log_dir}")

    cc = args.color_correct

    def metrics_cb(
        val_for_metrics: tf.data.Dataset,
        subdir: str,
        run_tag: str,
    ) -> Optional[ValidationMetricsCallback]:
        if args.no_metric_logs or base_log_dir is None:
            return None
        sub = base_log_dir / subdir
        return ValidationMetricsCallback(
            val_for_metrics,
            num_classes,
            sub,
            class_names=class_dirs,
            run_tag=run_tag,
        )

    def run_keras_split_pipeline() -> tuple[tf.data.Dataset, tf.data.Dataset]:
        """Single train/val split via image_dataset_from_directory (same as k_folds=1)."""
        _log("Building training and validation datasets (image_dataset_from_directory)…")
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
        train_ds, val_ds = _attach_preprocess(train_ds, val_ds, cc)
        return train_ds, val_ds

    model: Optional[tf.keras.Model] = None

    if args.k_folds == 1:
        train_ds, val_ds = run_keras_split_pipeline()
        if not args.no_metric_logs:
            val_ds = val_ds.cache()
        _log_dataset_steps(train_ds, val_ds, args.batch_size)
        mcb = metrics_cb(val_ds, "single_split", "metrics")
        model, _ = _train_one(
            train_ds,
            val_ds,
            num_classes=num_classes,
            img_size=args.img_size,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            log_prefix="",
            extra_callbacks=[mcb] if mcb else None,
        )
    else:
        _log(f"Stratified {args.k_folds}-fold CV: enumerating image paths…")
        paths, labels = _collect_paths_and_labels(data_dir, class_dirs)
        _log(f"Found {len(paths)} images across {num_classes} classes.")
        fold_splits = _stratified_k_fold_splits(paths, labels, args.k_folds, num_classes, args.seed)

        best_val_acc = -1.0
        best_fold_idx = -1
        best_weights: Optional[list[np.ndarray]] = None
        fold_scores: list[float] = []

        for fold_idx, (tr_p, va_p, tr_y, va_y) in enumerate(fold_splits):
            prefix = f"[Fold {fold_idx + 1}/{args.k_folds}] "
            if not tr_p or not va_p:
                raise SystemExit(
                    f"{prefix}empty train or validation set — use fewer folds or add more images per class."
                )
            tr_classes = set(tr_y)
            if len(tr_classes) < num_classes:
                raise SystemExit(
                    f"{prefix}training set is missing some classes (each class needs at least "
                    f"--k_folds images for stratified folds to work). Reduce --k_folds or add images."
                )
            _log(f"{prefix}train={len(tr_p)} val={len(va_p)}")
            train_ds = _make_dataset_from_paths(
                tr_p,
                tr_y,
                batch_size=args.batch_size,
                img_size=args.img_size,
                num_classes=num_classes,
                shuffle=True,
                shuffle_seed=args.seed + fold_idx,
            )
            val_ds = _make_dataset_from_paths(
                va_p,
                va_y,
                batch_size=args.batch_size,
                img_size=args.img_size,
                num_classes=num_classes,
                shuffle=False,
                shuffle_seed=args.seed,
            )
            train_ds, val_ds = _attach_preprocess(train_ds, val_ds, cc)
            if not args.no_metric_logs:
                val_ds = val_ds.cache()
            _log_dataset_steps(train_ds, val_ds, args.batch_size)

            mcb = metrics_cb(val_ds, f"fold_{fold_idx + 1}", "metrics")
            fmodel, hist = _train_one(
                train_ds,
                val_ds,
                num_classes=num_classes,
                img_size=args.img_size,
                learning_rate=args.learning_rate,
                epochs=args.epochs,
                log_prefix=prefix,
                extra_callbacks=[mcb] if mcb else None,
            )
            vacc = max(hist.history["val_accuracy"])
            fold_scores.append(float(vacc))
            _log(f"{prefix}best val_accuracy={vacc:.4f}")
            if vacc > best_val_acc:
                best_val_acc = vacc
                best_fold_idx = fold_idx
                best_weights = fmodel.get_weights()

        assert best_weights is not None
        mean_acc = float(np.mean(fold_scores))
        std_acc = float(np.std(fold_scores))
        _log(
            f"K-fold summary: val_accuracy mean={mean_acc:.4f} std={std_acc:.4f} "
            f"per-fold={[round(s, 4) for s in fold_scores]} (best fold {best_fold_idx + 1})"
        )

        if args.no_final_retrain:
            _log("Exporting model from best fold (--no_final_retrain).")
            model = _build_model(num_classes, args.img_size)
            model.set_weights(best_weights)
        else:
            _log(
                "Final training for export: single train/val split on all images "
                f"(validation_split={args.validation_split}) — use same preprocessing as folds."
            )
            train_ds, val_ds = run_keras_split_pipeline()
            if not args.no_metric_logs:
                val_ds = val_ds.cache()
            _log_dataset_steps(train_ds, val_ds, args.batch_size)
            mcb = metrics_cb(val_ds, "final_retrain", "metrics")
            model, _ = _train_one(
                train_ds,
                val_ds,
                num_classes=num_classes,
                img_size=args.img_size,
                learning_rate=args.learning_rate,
                epochs=args.epochs,
                log_prefix="[Final] ",
                extra_callbacks=[mcb] if mcb else None,
            )

    assert model is not None
    _export_tflite(model, args.out_tflite)
    print()
    _log("Next:")
    print(f"  1) Copy {args.out_tflite.name} -> app/src/main/assets/ml/plant_classifier.tflite")
    print(f"  2) Copy {args.out_labels.name} lines into app/src/main/assets/ml/plant_labels.txt")
    print("  3) Remove plant_classifier.onnx from assets if present (TFLite takes precedence).")


if __name__ == "__main__":
    main()
