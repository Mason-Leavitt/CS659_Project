#!/usr/bin/env python3
"""
Classical plant-image baseline: HOG features + SVM — same folder-per-class layout as train_export_tflite.py.

Use this to compare traditional CV + shallow classifier vs the CNN (MobileNetV2) pipeline.

Dataset layout:
  data/
    class_a/*.jpg
    class_b/*.jpg

Class index order = sorted folder names (same rule as the deep-learning script).
"""
from __future__ import annotations

import argparse
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
from hog_svm_utils import prepare_hog_features
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

_IMG_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"})


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] {msg}", flush=True)


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
    for i, p in enumerate(paths):
        feats.append(
            prepare_hog_features(
                p,
                img_size=img_size,
                orientations=orientations,
                pixels_per_cell=pixels_per_cell[0],
                cells_per_block=cells_per_block[0],
                block_norm="L2-Hys",
            )
        )
        if (i + 1) % max(1, n // 10) == 0 or i + 1 == n:
            _log(f"  HOG features: {i + 1}/{n} images ({_fmt_duration(time.perf_counter() - t0)})")
    return np.stack(feats, axis=0)


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


def main() -> None:
    p = argparse.ArgumentParser(
        description="Train HOG + SVM on folder-per-class images (classical baseline vs CNN)."
    )
    p.add_argument("--data_dir", type=Path, required=True, help="Root with one subfolder per class")
    p.add_argument("--img_size", type=int, default=224, help="Resize shorter pipeline like CNN (HOG on grayscale)")
    p.add_argument("--validation_split", type=float, default=0.15, help="Fraction held out for validation/test metrics")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--kernel",
        type=str,
        choices=("linear", "rbf"),
        default="linear",
        help="linear: LinearSVC (fast, high-dim HOG); rbf: SVC (can be slow on large datasets)",
    )
    p.add_argument("--C", type=float, default=1.0, help="SVM regularization strength")
    p.add_argument("--hog_orientations", type=int, default=9)
    p.add_argument("--hog_pixels_per_cell", type=int, default=16, help="Square cell side in pixels")
    p.add_argument("--hog_cells_per_block", type=int, default=2, help="Square block side in cells")
    p.add_argument(
        "--out_model",
        type=Path,
        default=Path("hog_svm_model.joblib"),
        help="Filename for saved Pipeline inside result_dir (scaler + SVM)",
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
    args = p.parse_args()

    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"Not a directory: {data_dir}")

    class_dirs = sorted(
        d.name for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    if len(class_dirs) < 2:
        raise SystemExit("Need at least 2 class subfolders under --data_dir")

    num_classes = len(class_dirs)

    result_dir = args.result_dir
    if result_dir is None:
        result_dir = (
            Path("result")
            / f"hog_svm_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC')}"
        ).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    _log(
        f"HOG+SVM — data_dir={data_dir} img_size={args.img_size} val_split={args.validation_split} "
        f"kernel={args.kernel} C={args.C} hog_cell={args.hog_pixels_per_cell}px"
    )
    _log(f"Result directory: {result_dir}")

    paths, labels = _collect_paths_and_labels(data_dir, class_dirs)
    _log(f"Found {len(paths)} images, {num_classes} classes.")

    ppc = (args.hog_pixels_per_cell, args.hog_pixels_per_cell)
    cpb = (args.hog_cells_per_block, args.hog_cells_per_block)

    tr_paths, va_paths, y_train, y_val = train_test_split(
        paths,
        labels,
        test_size=args.validation_split,
        stratify=labels,
        random_state=args.seed,
    )

    _log(f"Train images: {len(tr_paths)} | Validation images: {len(va_paths)}")
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

    if args.kernel == "linear":
        clf = LinearSVC(C=args.C, dual="auto", max_iter=20_000, random_state=args.seed)
    else:
        clf = SVC(C=args.C, kernel="rbf", gamma="scale", random_state=args.seed)

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    _log("Training SVM…")
    t0 = time.perf_counter()
    pipe.fit(X_train, np.array(y_train))
    _log(f"SVM fit finished in {_fmt_duration(time.perf_counter() - t0)}")

    y_pred = pipe.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_val, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_val, y_pred, average="macro", zero_division=0)

    _log(f"Validation — accuracy={acc:.4f} f1_macro={f1:.4f} precision_macro={prec:.4f} recall_macro={rec:.4f}")

    report = classification_report(
        y_val,
        y_pred,
        labels=list(range(num_classes)),
        target_names=class_dirs,
        zero_division=0,
    )
    (result_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_val, y_pred, labels=list(range(num_classes)))
    kplot = min(48, num_classes)
    cm_norm = cm.astype(np.float64) / (cm.sum(axis=1, keepdims=True).astype(np.float64) + 1e-8)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(cm_norm[:kplot, :kplot], interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax.set_title("HOG+SVM normalized confusion (validation)")
    tick = [class_dirs[i][:14] for i in range(kplot)]
    ax.set_xticks(np.arange(kplot))
    ax.set_yticks(np.arange(kplot))
    ax.set_xticklabels(tick, rotation=90, fontsize=6)
    ax.set_yticklabels(tick, fontsize=6)
    fig.tight_layout()
    fig.savefig(result_dir / "confusion_matrix_normalized.png", dpi=150)
    plt.close(fig)

    summary: dict[str, Any] = {
        "method": "hog_svm",
        "kernel": args.kernel,
        "C": args.C,
        "img_size": args.img_size,
        "hog_orientations": args.hog_orientations,
        "hog_pixels_per_cell": args.hog_pixels_per_cell,
        "hog_cells_per_block": args.hog_cells_per_block,
        "feature_dim": int(X_train.shape[1]),
        "n_train": len(tr_paths),
        "n_val": len(va_paths),
        "validation_accuracy": float(acc),
        "f1_macro": float(f1),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    out_model = result_dir / args.out_model.name
    out_labels = result_dir / args.out_labels.name
    joblib.dump(pipe, out_model)
    out_labels.write_text("\n".join(class_dirs) + "\n", encoding="utf-8")

    _log(f"Saved model: {out_model}")
    _log(f"Saved labels: {out_labels}")
    _log("Compare these metrics with CNN runs in result/<cnn_timestamp>/ (same val split only if same --seed and split logic).")


if __name__ == "__main__":
    main()
