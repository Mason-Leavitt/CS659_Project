#!/usr/bin/env python3
"""
Classical plant-image baseline: HOG features + linear classifier — same folder-per-class layout as train_export_tflite.py.

Default: StandardScaler + sklearn LinearSVC. Use ``--linear-head tf`` for a Keras Dense head on standardized
HOG (TensorFlow; GPU when visible), same TFLite graph layout as the sklearn path.

Stratified train / validation / test split (default 70/15/15) matches train_export_tflite.py when using the
same --seed and split fractions (see model_hyperparameters.json and --config).

Use this to compare traditional CV + shallow classifier vs the CNN (MobileNetV2) pipeline.

Dataset layout — **folder-per-class** (``data/class_a/*.jpg``) or **split folders** with species
subfolders (``data/train/<species>/*.jpg``, same as ``train_export_tflite.py``).

Class index order = sorted species (folder) names (same rule as the deep-learning script).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from skimage.color import rgb2gray

from classification_metrics_sklearn import (
    save_multiclass_roc_plot,
    sklearn_metrics_with_probs,
    softmax_rows,
)
from experiment_config import (
    collect_paths_and_labels_for_classes,
    discover_class_folder_names,
    merge_config_into_argparse_defaults,
    per_class_counts,
    snapshot_full_config,
    split_summary_dict,
    stratified_train_val_test,
    validate_split_fractions,
    write_json,
)
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize

_IMG_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"})


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] {msg}", flush=True)


def _tensorflow_init_gpu_memory() -> None:
    """
    Must run before any TensorFlow module that touches the GPU (e.g. hog_tf).
    Enables memory growth so TF does not reserve all VRAM — avoids cudaSetDevice OOM when
    another framework or process is using the same GPU.
    """
    import tensorflow as tf

    try:
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except (ValueError, RuntimeError):
        # Already configured or no GPU visible
        pass


def _subsample_stratified(
    paths: list[str],
    labels: list[int],
    n_target: int,
    seed: int,
) -> tuple[list[str], list[int]]:
    """
    Reduce to n_target images with stratified sampling (same class proportions as input).
    Requires sklearn-compatible label distribution (each class present with enough support).
    """
    if n_target >= len(paths):
        return paths, labels
    if n_target < 1:
        raise SystemExit("--max_*_images / sample fraction produced empty set; increase limits.")
    pa = np.array(paths, dtype=object)
    ya = np.array(labels, dtype=np.int64)
    tr_p, _, tr_y, _ = train_test_split(
        pa,
        ya,
        train_size=n_target,
        stratify=ya,
        random_state=seed,
    )
    return tr_p.tolist(), tr_y.tolist()


def _load_gray_resized(path: Path, img_size: int) -> np.ndarray:
    img = imread(str(path))
    if img.ndim == 2:
        g = img.astype(np.float64)
        if g.max() > 1.0:
            g /= 255.0
        return resize(g, (img_size, img_size), anti_aliasing=True)
    if img.shape[2] == 4:
        img = img[..., :3]
    rgb = img.astype(np.float64)
    if rgb.max() > 1.0:
        rgb /= 255.0
    g = rgb2gray(rgb)
    return resize(g, (img_size, img_size), anti_aliasing=True)


def _hog_feature(
    gray: np.ndarray,
    *,
    orientations: int,
    pixels_per_cell: tuple[int, int],
    cells_per_block: tuple[int, int],
) -> np.ndarray:
    return hog(
        gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        visualize=False,
        feature_vector=True,
    )


def _hog_progress_interval(n: int) -> int:
    """Log often enough for large jobs (10% steps mean ~21k images before the first line)."""
    if n <= 0:
        return 1
    if n < 500:
        return max(1, n // 10)
    if n < 5000:
        return max(50, n // 20)
    return min(5000, max(500, n // 40))


def _extract_matrix(
    paths: list[str],
    *,
    img_size: int,
    orientations: int,
    pixels_per_cell: tuple[int, int],
    cells_per_block: tuple[int, int],
) -> np.ndarray:
    feats: list[np.ndarray] = []
    n = len(paths)
    t0 = time.perf_counter()
    step = _hog_progress_interval(n)
    _log(f"  HOG extraction: {n} images, progress every {step} images…")
    for i, p in enumerate(paths):
        gray = _load_gray_resized(Path(p), img_size)
        feats.append(_hog_feature(gray, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block))
        done = i + 1
        if done % step == 0 or done == n:
            elapsed = time.perf_counter() - t0
            rate = done / elapsed if elapsed > 0 else 0.0
            remaining = n - done
            eta_s = remaining / rate if rate > 0 else 0.0
            pct = 100.0 * done / n
            _log(
                f"  HOG features: {done}/{n} ({pct:.1f}%) — elapsed {_fmt_duration(elapsed)}, "
                f"~{rate:.1f} img/s, ETA ~{_fmt_duration(eta_s)}"
            )
    return np.stack(feats, axis=0)


def _extract_matrix_tf_gpu(
    paths: list[str],
    *,
    img_size: int,
    orientations: int,
    pixels_per_cell: int,
    cells_per_block: int,
    batch_size: int,
) -> np.ndarray:
    from hog_tf import extract_hog_matrix_from_paths_tf

    return extract_hog_matrix_from_paths_tf(
        paths,
        img_size=img_size,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        batch_size=batch_size,
        log_fn=_log,
    )


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


def _write_confusion_matrix_csv(
    out_path: Path,
    cm: np.ndarray,
    class_names: list[str],
    *,
    normalized: bool,
) -> None:
    """Rows = true class, columns = predicted class (same layout as sklearn.confusion_matrix)."""
    n = cm.shape[0]
    pred_headers = [f"pred_{j}" for j in range(n)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true_idx", "true_name", *pred_headers])
        for i in range(n):
            row = [i, class_names[i]]
            if normalized:
                row.extend(float(x) for x in cm[i].tolist())
            else:
                row.extend(int(x) for x in cm[i].tolist())
            w.writerow(row)


def _write_metrics_per_split_csv(result_dir: Path, rows: list[dict[str, Any]]) -> None:
    """One row per split for merging runs in pandas/plotting tools."""
    path = result_dir / "metrics_per_split.csv"
    fieldnames = [
        "split",
        "accuracy",
        "f1_macro",
        "f1_weighted",
        "precision_macro",
        "recall_macro",
        "roc_auc_ovr_macro",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class HogLinearHeadModel:
    """
    StandardScaler + multiclass linear logits in LinearSVC layout (coef_ row per class).

    Used when ``--linear-head tf`` trains a Keras ``Dense`` head on standardized HOG; TFLite export
    uses the same ``HogLinearTfliteModelFactory`` graph as sklearn ``LinearSVC``.
    """

    def __init__(self, scaler: StandardScaler, coef: np.ndarray, intercept: np.ndarray) -> None:
        self.scaler = scaler
        self.coef_ = np.asarray(coef, dtype=np.float64)
        self.intercept_ = np.asarray(intercept, dtype=np.float64).reshape(-1)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        z = self.scaler.transform(X)
        return z @ self.coef_.T + self.intercept_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.decision_function(X), axis=1)


def _decision_scores(pipe: Any, X: np.ndarray) -> np.ndarray:
    """Multiclass scores (n_samples, n_classes) for LinearSVC / SVC / HogLinearHeadModel."""
    if isinstance(pipe, HogLinearHeadModel):
        return np.asarray(pipe.decision_function(X), dtype=np.float64)
    if hasattr(pipe, "decision_function"):
        return np.asarray(pipe.decision_function(X), dtype=np.float64)
    raise TypeError("Classifier must implement decision_function for ROC (use linear or RBF kernel).")


def _apply_hog_preset(args: argparse.Namespace) -> None:
    """
    Apply built-in HOG geometry presets after argparse (overrides hog_pixels_per_cell / hog_orientations).

    ``coarse``: larger cells + fewer orientations than default — less compute per image and a shorter
    feature vector; accuracy may drop vs default 16px / 9 orientations.
    """
    preset = getattr(args, "hog_preset", "default")
    if preset == "coarse":
        args.hog_pixels_per_cell = 32
        args.hog_orientations = 6


def _train_linear_head_tf(
    *,
    X_train: np.ndarray,
    y_train: list[int],
    X_val: np.ndarray,
    y_val: list[int],
    num_classes: int,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> HogLinearHeadModel:
    """StandardScaler on CPU, then Keras Dense logits (GPU when TensorFlow sees a GPU)."""
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        _log(
            f"TensorFlow linear head: {len(gpus)} GPU device(s) visible — "
            f"Keras will place the Dense layer on GPU (CUDA) when ops run. "
            f"{[d.name for d in gpus]}"
        )
    else:
        _log(
            "TensorFlow linear head: no GPU in tf.config — training runs on CPU. "
            "Use a CUDA build of TensorFlow and drivers if you expected a GPU."
        )

    scaler = StandardScaler()
    z_train = scaler.fit_transform(X_train).astype(np.float32)
    z_val = scaler.transform(X_val).astype(np.float32)
    y_tr = np.asarray(y_train, dtype=np.int32)
    y_va = np.asarray(y_val, dtype=np.int32)

    tf.random.set_seed(seed)
    feat_dim = int(z_train.shape[1])

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(feat_dim,)),
            tf.keras.layers.Dense(
                num_classes,
                activation=None,
                dtype=tf.float32,
                name="linear_logits",
            ),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    callbacks: list[Any] = []
    fit_kwargs: dict[str, Any] = {
        "epochs": epochs,
        "batch_size": max(1, batch_size),
        "verbose": 1,
    }
    if y_va.size > 0:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=min(7, max(2, epochs // 15)),
                restore_best_weights=True,
                verbose=1,
            )
        )
        fit_kwargs["validation_data"] = (z_val, y_va)
        fit_kwargs["callbacks"] = callbacks

    _log("Training linear head (TensorFlow; GPU if visible)…")
    t0 = time.perf_counter()
    model.fit(z_train, y_tr, **fit_kwargs)
    _log(f"TensorFlow linear head finished in {_fmt_duration(time.perf_counter() - t0)}")

    layer = model.get_layer("linear_logits")
    w = layer.kernel.numpy()
    b = layer.bias.numpy()
    coef = np.asarray(w.T, dtype=np.float64)
    intercept = np.asarray(b.reshape(-1), dtype=np.float64)
    return HogLinearHeadModel(scaler, coef, intercept)


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None, help="JSON file (see model_hyperparameters.json)")
    pre_args, _ = pre.parse_known_args()

    file_defaults: dict[str, Any] = {}
    if pre_args.config is not None:
        from experiment_config import load_json_config

        cfg = load_json_config(pre_args.config.resolve())
        file_defaults = merge_config_into_argparse_defaults(cfg, section="traditional")

    p = argparse.ArgumentParser(
        description="Train HOG + SVM on folder-per-class images (classical baseline vs CNN).",
        parents=[pre],
    )
    p.add_argument("--data_dir", type=Path, required=True, help="Root with one subfolder per class")
    p.add_argument("--img_size", type=int, default=file_defaults.get("img_size", 224), help="Resize (HOG on grayscale)")
    p.add_argument(
        "--train_fraction",
        type=float,
        default=file_defaults.get("train_fraction", 0.7),
        help="Fraction of images for training (stratified)",
    )
    p.add_argument(
        "--validation_fraction",
        type=float,
        default=file_defaults.get("validation_fraction", 0.15),
        help="Fraction for validation (hyperparameter / early stopping tuning)",
    )
    p.add_argument(
        "--test_fraction",
        type=float,
        default=file_defaults.get("test_fraction", 0.15),
        help="Fraction for final held-out test (report metrics)",
    )
    p.add_argument("--seed", type=int, default=file_defaults.get("seed", 42))
    p.add_argument(
        "--kernel",
        type=str,
        choices=("linear", "rbf"),
        default=file_defaults.get("kernel", "linear"),
        help="linear: LinearSVC or TF head (see --linear-head); rbf: SVC (can be slow on large datasets)",
    )
    p.add_argument(
        "--linear-head",
        type=str,
        choices=("sklearn", "tf"),
        default=str(file_defaults.get("linear_head", "sklearn")),
        help="For --kernel linear: sklearn LinearSVC (CPU) or TensorFlow Dense on standardized HOG (GPU if visible).",
    )
    p.add_argument(
        "--tf-linear-epochs",
        type=int,
        default=int(file_defaults.get("tf_linear_epochs", 100)),
        help="When --linear-head tf: max epochs (early stopping may finish sooner).",
    )
    p.add_argument(
        "--tf-linear-batch-size",
        type=int,
        default=int(file_defaults.get("tf_linear_batch_size", 4096)),
        help="When --linear-head tf: batch size for the dense head.",
    )
    p.add_argument(
        "--tf-linear-learning-rate",
        type=float,
        default=float(file_defaults.get("tf_linear_learning_rate", 0.001)),
        help="When --linear-head tf: Adam learning rate.",
    )
    p.add_argument("--C", type=float, default=file_defaults.get("C", 1.0), help="SVM regularization strength")
    p.add_argument(
        "--hog-preset",
        type=str,
        choices=("default", "coarse"),
        default=str(file_defaults.get("hog_preset", "default")),
        help="default: use --hog-pixels-per-cell / --hog-orientations (or config). "
        "coarse: fixed faster HOG — 32px cells, 6 orientations (overrides those two flags).",
    )
    p.add_argument("--hog_orientations", type=int, default=file_defaults.get("hog_orientations", 9))
    p.add_argument("--hog_pixels_per_cell", type=int, default=file_defaults.get("hog_pixels_per_cell", 16))
    p.add_argument("--hog_cells_per_block", type=int, default=file_defaults.get("hog_cells_per_block", 2))
    p.add_argument(
        "--out_model",
        type=Path,
        default=Path("hog_svm_model.joblib"),
        help="Filename for saved model inside result_dir (sklearn Pipeline or HogLinearHeadModel)",
    )
    p.add_argument(
        "--out_labels",
        type=Path,
        default=Path("hog_svm_labels.txt"),
        help="Filename for label list inside result_dir",
    )
    p.add_argument(
        "--result_dir",
        type=Path,
        default=None,
        help="Directory for metrics, plots, and copies of model/labels (default: result/hog_svm_<UTC>/)",
    )
    p.add_argument(
        "--export_tflite",
        type=Path,
        default=None,
        help="If set, also export linear HOG+SVM to this .tflite path (requires --kernel linear; needs tensorflow).",
    )
    p.add_argument(
        "--save_split_lists",
        action="store_true",
        help="Write train_paths.txt, val_paths.txt, test_paths.txt under result_dir for reproducibility.",
    )
    p.add_argument(
        "--gpu-hog",
        dest="use_gpu_hog",
        default=file_defaults.get("use_gpu_hog", False),
        action=argparse.BooleanOptionalAction,
        help="Use TensorFlow for HOG (runs on GPU when visible). Matches TFLite preprocessing; "
        "CPU skimage path (--no-gpu-hog) differs slightly (resize/gray order).",
    )
    p.add_argument(
        "--hog_batch_size",
        type=int,
        default=file_defaults.get("hog_batch_size", 64),
        help="Images per batch for TF HOG (--gpu-hog); lower if GPU OOM.",
    )
    p.add_argument(
        "--train_sample_fraction",
        type=float,
        default=file_defaults.get("train_sample_fraction", 1.0),
        help="After train/val/test split, use only this fraction of TRAIN images (stratified). "
        "Use <1.0 for faster/cheaper runs (e.g. 0.05 = 5%%).",
    )
    p.add_argument(
        "--max_train_images",
        type=int,
        default=file_defaults.get("max_train_images") or 0,
        help="Cap TRAIN images after --train_sample_fraction (0 = no cap). Example: 20000.",
    )
    p.add_argument(
        "--max_val_images",
        type=int,
        default=file_defaults.get("max_val_images") or 0,
        help="Cap validation images for HOG+metrics (0 = no cap). Smaller = faster val pass.",
    )
    p.add_argument(
        "--max_test_images",
        type=int,
        default=file_defaults.get("max_test_images") or 0,
        help="Cap test images for HOG+metrics (0 = no cap).",
    )
    args = p.parse_args()
    _apply_hog_preset(args)

    if args.linear_head == "tf" and args.kernel != "linear":
        raise SystemExit("--linear-head tf requires --kernel linear.")

    if args.use_gpu_hog or args.export_tflite is not None or args.linear_head == "tf":
        _tensorflow_init_gpu_memory()

    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"Not a directory: {data_dir}")

    try:
        class_dirs, nested_split_layout = discover_class_folder_names(data_dir)
    except ValueError as e:
        raise SystemExit(str(e))

    num_classes = len(class_dirs)
    if nested_split_layout:
        _log(
            f"Split-style layout (train/val/test → species): {num_classes} classes — "
            "confusion matrices label species, not split folders."
        )

    result_dir = args.result_dir
    if result_dir is None:
        result_dir = (
            Path("result")
            / f"hog_svm_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC')}"
        ).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    validate_split_fractions(args.train_fraction, args.validation_fraction, args.test_fraction)
    if args.img_size % args.hog_pixels_per_cell != 0:
        raise SystemExit(
            f"--img_size ({args.img_size}) must be divisible by hog_pixels_per_cell ({args.hog_pixels_per_cell}) "
            f"for the HOG grid. Example: --img_size 224 with --hog-preset coarse (32px cells), or --img_size 128."
        )
    _log(
        f"HOG+SVM — data_dir={data_dir} img_size={args.img_size} "
        f"split={args.train_fraction:.2f}/{args.validation_fraction:.2f}/{args.test_fraction:.2f} "
        f"kernel={args.kernel} C={args.C} hog_preset={args.hog_preset} hog_cell={args.hog_pixels_per_cell}px "
        f"hog_ori={args.hog_orientations} "
        f"hog_backend={'tensorflow_gpu' if args.use_gpu_hog else 'skimage_cpu'}"
    )
    if args.hog_preset == "coarse":
        _log(
            "HOG preset 'coarse' is active (32px cells, 6 orientations). "
            "For extra throughput, try --img_size 128 (divisible by 32)."
        )
    if args.use_gpu_hog:
        _log(
            f"TF HOG enabled (batch_size={args.hog_batch_size}); "
            f"linear classifier runs on {'TensorFlow (GPU if visible)' if args.linear_head == 'tf' else 'CPU (sklearn LinearSVC)'}. "
            "Install tensorflow[and-cuda] on a GPU machine for HOG / TF-head speedup."
        )
    elif args.linear_head == "tf":
        _log(
            "Linear classifier: TensorFlow Dense (--linear-head tf); GPU used when TensorFlow sees a GPU. "
            "Loss is softmax cross-entropy on logits (not identical to sklearn LinearSVC hinge/OvR)."
        )
    if (
        args.train_sample_fraction < 1.0
        or (args.max_train_images or 0) > 0
        or (args.max_val_images or 0) > 0
        or (args.max_test_images or 0) > 0
    ):
        _log(
            "Budget / speed subsampling is ON — metrics are on subsets; cite subsample settings in reports. "
            "For full-data experiments use --train_sample_fraction 1.0 and --max_train_images 0."
        )
    _log(f"Result directory: {result_dir}")

    try:
        paths, labels = collect_paths_and_labels_for_classes(
            data_dir, class_dirs, nested_split_layout=nested_split_layout, img_exts=_IMG_EXTS
        )
    except ValueError as e:
        raise SystemExit(str(e))
    _log(f"Found {len(paths)} images, {num_classes} classes.")

    tr_paths, va_paths, te_paths, y_train, y_val, y_test = stratified_train_val_test(
        paths,
        labels,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        random_state=args.seed,
    )

    split_meta = split_summary_dict(
        train_labels=y_train,
        val_labels=y_val,
        test_labels=y_test,
        num_classes=num_classes,
        class_names=class_dirs,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    write_json(result_dir / "split_summary.json", split_meta)
    if args.save_split_lists:
        (result_dir / "train_paths.txt").write_text("\n".join(tr_paths) + "\n", encoding="utf-8")
        (result_dir / "val_paths.txt").write_text("\n".join(va_paths) + "\n", encoding="utf-8")
        (result_dir / "test_paths.txt").write_text("\n".join(te_paths) + "\n", encoding="utf-8")

    _log(
        f"Train={len(tr_paths)} | Val={len(va_paths)} | Test={len(te_paths)} "
        f"(per-class train counts: {per_class_counts(y_train, num_classes)})"
    )

    subsample_note: dict[str, Any] = {
        "train_sample_fraction": args.train_sample_fraction,
        "max_train_images": args.max_train_images or None,
        "max_val_images": args.max_val_images or None,
        "max_test_images": args.max_test_images or None,
    }
    if args.train_sample_fraction <= 0 or args.train_sample_fraction > 1.0:
        raise SystemExit("--train_sample_fraction must be in (0, 1.0]")
    n_train_target = int(len(tr_paths) * args.train_sample_fraction)
    if args.max_train_images and args.max_train_images > 0:
        n_train_target = min(n_train_target, args.max_train_images)
    n_train_target = max(n_train_target, num_classes)
    n_train_target = min(n_train_target, len(tr_paths))
    if n_train_target < num_classes:
        raise SystemExit(
            "Increase --train_sample_fraction or --max_train_images — not enough train images for "
            f"{num_classes} classes after subsampling."
        )
    if n_train_target < len(tr_paths):
        _log(
            f"Subsampling TRAIN: {len(tr_paths)} -> {n_train_target} images (stratified, seed={args.seed})"
        )
        tr_paths, y_train = _subsample_stratified(tr_paths, y_train, n_train_target, args.seed)
    if args.max_val_images and args.max_val_images > 0 and len(va_paths) > args.max_val_images:
        _log(f"Subsampling VAL: {len(va_paths)} -> {args.max_val_images} images (stratified)")
        va_paths, y_val = _subsample_stratified(va_paths, y_val, args.max_val_images, args.seed + 1)
    if args.max_test_images and args.max_test_images > 0 and len(te_paths) > args.max_test_images:
        _log(f"Subsampling TEST: {len(te_paths)} -> {args.max_test_images} images (stratified)")
        te_paths, y_test = _subsample_stratified(te_paths, y_test, args.max_test_images, args.seed + 2)

    subsample_note["actual_train_images"] = len(tr_paths)
    subsample_note["actual_val_images"] = len(va_paths)
    subsample_note["actual_test_images"] = len(te_paths)
    write_json(result_dir / "hog_svm_subsample.json", subsample_note)

    ppc = (args.hog_pixels_per_cell, args.hog_pixels_per_cell)
    cpb = (args.hog_cells_per_block, args.hog_cells_per_block)

    if args.use_gpu_hog:
        _log("Extracting HOG (train)…")
        X_train = _extract_matrix_tf_gpu(
            tr_paths,
            img_size=args.img_size,
            orientations=args.hog_orientations,
            pixels_per_cell=args.hog_pixels_per_cell,
            cells_per_block=args.hog_cells_per_block,
            batch_size=args.hog_batch_size,
        )
        _log("Extracting HOG (validation)…")
        X_val = _extract_matrix_tf_gpu(
            va_paths,
            img_size=args.img_size,
            orientations=args.hog_orientations,
            pixels_per_cell=args.hog_pixels_per_cell,
            cells_per_block=args.hog_cells_per_block,
            batch_size=args.hog_batch_size,
        )
        _log("Extracting HOG (test)…")
        X_test = _extract_matrix_tf_gpu(
            te_paths,
            img_size=args.img_size,
            orientations=args.hog_orientations,
            pixels_per_cell=args.hog_pixels_per_cell,
            cells_per_block=args.hog_cells_per_block,
            batch_size=args.hog_batch_size,
        )
    else:
        _log("Extracting HOG (train)…")
        X_train = _extract_matrix(
            tr_paths,
            img_size=args.img_size,
            orientations=args.hog_orientations,
            pixels_per_cell=ppc,
            cells_per_block=cpb,
        )
        _log("Extracting HOG (validation)…")
        X_val = _extract_matrix(
            va_paths,
            img_size=args.img_size,
            orientations=args.hog_orientations,
            pixels_per_cell=ppc,
            cells_per_block=cpb,
        )
        _log("Extracting HOG (test)…")
        X_test = _extract_matrix(
            te_paths,
            img_size=args.img_size,
            orientations=args.hog_orientations,
            pixels_per_cell=ppc,
            cells_per_block=cpb,
        )

    if args.kernel == "linear":
        if args.linear_head == "tf":
            pipe = _train_linear_head_tf(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                num_classes=num_classes,
                seed=args.seed,
                epochs=args.tf_linear_epochs,
                batch_size=args.tf_linear_batch_size,
                learning_rate=args.tf_linear_learning_rate,
            )
        else:
            clf = LinearSVC(C=args.C, dual="auto", max_iter=20_000, random_state=args.seed)
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
            _log("Training SVM (sklearn LinearSVC)…")
            t0 = time.perf_counter()
            pipe.fit(X_train, np.array(y_train))
            _log(f"SVM fit finished in {_fmt_duration(time.perf_counter() - t0)}")
    else:
        clf = SVC(C=args.C, kernel="rbf", gamma="scale", random_state=args.seed)
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        _log("Training SVM…")
        t0 = time.perf_counter()
        pipe.fit(X_train, np.array(y_train))
        _log(f"SVM fit finished in {_fmt_duration(time.perf_counter() - t0)}")

    def _metrics_block(y_true: list[int], X: np.ndarray, split_name: str) -> dict[str, float]:
        y_true_arr = np.asarray(y_true)
        y_pred = pipe.predict(X)
        scores = _decision_scores(pipe, X)
        y_prob = softmax_rows(scores)
        m = sklearn_metrics_with_probs(y_true_arr, y_pred, y_prob, num_classes)
        roc = m.get("roc_auc_ovr_macro", float("nan"))
        roc_txt = f"{roc:.4f}" if roc == roc else "nan"
        _log(
            f"{split_name.capitalize()} — accuracy={m['accuracy']:.4f} f1_macro={m['f1_macro']:.4f} "
            f"precision_macro={m['precision_macro']:.4f} recall_macro={m['recall_macro']:.4f} "
            f"roc_auc_ovr_macro={roc_txt}"
        )
        return m

    m_train = _metrics_block(y_train, X_train, "train")
    m_val = _metrics_block(y_val, X_val, "validation")
    m_test = _metrics_block(y_test, X_test, "test")

    _write_metrics_per_split_csv(
        result_dir,
        [
            {"split": "train", **m_train},
            {"split": "validation", **m_val},
            {"split": "test", **m_test},
        ],
    )

    def _write_report_and_cm(y_true: list[int], y_pred: np.ndarray, prefix: str, title: str) -> None:
        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
            target_names=class_dirs,
            zero_division=0,
        )
        (result_dir / f"{prefix}_classification_report.txt").write_text(report, encoding="utf-8")
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        kplot = min(48, num_classes)
        cm_norm = cm.astype(np.float64) / (cm.sum(axis=1, keepdims=True).astype(np.float64) + 1e-8)
        _write_confusion_matrix_csv(
            result_dir / f"{prefix}_confusion_matrix_counts.csv",
            cm,
            class_dirs,
            normalized=False,
        )
        _write_confusion_matrix_csv(
            result_dir / f"{prefix}_confusion_matrix_normalized.csv",
            cm_norm,
            class_dirs,
            normalized=True,
        )
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(cm_norm[:kplot, :kplot], interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
        ax.set_title(title)
        tick = [class_dirs[i][:14] for i in range(kplot)]
        ax.set_xticks(np.arange(kplot))
        ax.set_yticks(np.arange(kplot))
        ax.set_xticklabels(tick, rotation=90, fontsize=6)
        ax.set_yticklabels(tick, fontsize=6)
        fig.tight_layout()
        fig.savefig(result_dir / f"{prefix}_confusion_matrix_normalized.png", dpi=150)
        plt.close(fig)

    y_pred_train = pipe.predict(X_train)
    y_pred_val = pipe.predict(X_val)
    y_pred_test = pipe.predict(X_test)
    _write_report_and_cm(y_train, y_pred_train, "train", "HOG+SVM normalized confusion (train)")
    _write_report_and_cm(y_val, y_pred_val, "validation", "HOG+SVM normalized confusion (validation)")
    _write_report_and_cm(y_test, y_pred_test, "test", "HOG+SVM normalized confusion (test)")

    for y_t, X_mat, prefix in (
        (y_train, X_train, "train"),
        (y_val, X_val, "validation"),
        (y_test, X_test, "test"),
    ):
        scores = _decision_scores(pipe, X_mat)
        save_multiclass_roc_plot(
            np.asarray(y_t),
            softmax_rows(scores),
            num_classes,
            result_dir / f"{prefix}_roc_curves.png",
        )

    metrics_all: dict[str, Any] = {
        "method": "hog_svm",
        "subsampling": subsample_note,
        "hog_backend": "tensorflow" if args.use_gpu_hog else "skimage",
        "use_gpu_hog": bool(args.use_gpu_hog),
        "hog_batch_size": args.hog_batch_size if args.use_gpu_hog else None,
        "kernel": args.kernel,
        "linear_head": args.linear_head if args.kernel == "linear" else None,
        "tf_linear_epochs": args.tf_linear_epochs if args.kernel == "linear" and args.linear_head == "tf" else None,
        "tf_linear_batch_size": args.tf_linear_batch_size if args.kernel == "linear" and args.linear_head == "tf" else None,
        "tf_linear_learning_rate": args.tf_linear_learning_rate if args.kernel == "linear" and args.linear_head == "tf" else None,
        "linear_head_tf_note": (
            "Adam + sparse_categorical_crossentropy on logits (multiclass softmax); "
            "not the same optimization as sklearn LinearSVC (hinge, OvR)."
            if args.kernel == "linear" and args.linear_head == "tf"
            else None
        ),
        "C": args.C,
        "img_size": args.img_size,
        "hog_preset": args.hog_preset,
        "hog_orientations": args.hog_orientations,
        "hog_pixels_per_cell": args.hog_pixels_per_cell,
        "hog_cells_per_block": args.hog_cells_per_block,
        "feature_dim": int(X_train.shape[1]),
        "n_train": len(tr_paths),
        "n_validation": len(va_paths),
        "n_test": len(te_paths),
        "metrics": {
            "train": {**m_train, "note": "in-sample; optimistic vs val/test"},
            "validation": m_val,
            "test": {**m_test, "note": "held-out; use for report headline if you tune on val only"},
        },
    }
    write_json(result_dir / "metrics_train_val_test.json", metrics_all)
    # Back-compat: keep summary.json as validation-focused snapshot
    summary = {
        **metrics_all,
        "validation_accuracy": m_val["accuracy"],
        "f1_macro": m_val["f1_macro"],
        "precision_macro": m_val["precision_macro"],
        "recall_macro": m_val["recall_macro"],
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    hp_snapshot = snapshot_full_config(
        pre_args.config.resolve() if pre_args.config else None,
        {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
            if k != "config"
        },
    )
    write_json(result_dir / "hyperparameters_snapshot.json", hp_snapshot)

    out_model = result_dir / args.out_model.name
    out_labels = result_dir / args.out_labels.name
    joblib.dump(pipe, out_model)
    out_labels.write_text("\n".join(class_dirs) + "\n", encoding="utf-8")

    _log(f"Saved model: {out_model}")
    _log(f"Saved labels: {out_labels}")

    if args.export_tflite is not None:
        if args.kernel != "linear":
            raise SystemExit("--export_tflite is only supported for --kernel linear (RBF SVM cannot be exported as TFLite).")
        from hog_tf import export_keras_to_tflite_float32, pipeline_weights_for_tflite

        tflite_path = args.export_tflite.expanduser().resolve()
        factory, _fd = pipeline_weights_for_tflite(
            pipe,
            img_size=args.img_size,
            hog_orientations=args.hog_orientations,
            hog_pixels_per_cell=args.hog_pixels_per_cell,
            hog_cells_per_block=args.hog_cells_per_block,
        )
        keras_model = factory.build()
        export_keras_to_tflite_float32(keras_model, str(tflite_path))
        _log(f"Saved TFLite (HOG + linear SVM): {tflite_path}")

    _log("Compare these metrics with CNN runs in result/<cnn_timestamp>/ (same val split only if same --seed and split logic).")


if __name__ == "__main__":
    main()
