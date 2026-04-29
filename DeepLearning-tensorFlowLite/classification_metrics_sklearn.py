"""Sklearn-only metrics (no TensorFlow). Lets train_hog_svm configure the GPU before TF imports."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def sklearn_metrics_from_labels(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    _num_classes: int,
) -> dict[str, float]:
    """Accuracy / precision / recall / F1 without probabilities (no ROC AUC)."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    """Stable softmax for multiclass scores (n_samples, n_classes)."""
    m = np.max(logits, axis=1, keepdims=True)
    ex = np.exp(np.clip(logits - m, -60.0, 60.0))
    s = np.sum(ex, axis=1, keepdims=True)
    return ex / (s + 1e-12)


def sklearn_metrics_with_probs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int,
) -> dict[str, float]:
    """Same as ``sklearn_metrics_from_labels`` plus ``roc_auc_ovr_macro`` (OvR, macro)."""
    out = sklearn_metrics_from_labels(y_true, y_pred, num_classes)
    try:
        if num_classes < 2 or np.unique(y_true).size < 2:
            out["roc_auc_ovr_macro"] = float("nan")
        elif num_classes == 2:
            out["roc_auc_ovr_macro"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            Y = label_binarize(y_true, classes=np.arange(num_classes))
            if Y.shape[1] == 1:
                out["roc_auc_ovr_macro"] = float("nan")
            else:
                out["roc_auc_ovr_macro"] = float(
                    roc_auc_score(Y, y_prob, average="macro", multi_class="ovr")
                )
    except ValueError:
        out["roc_auc_ovr_macro"] = float("nan")
    return out


def save_multiclass_roc_plot(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int,
    out_path: Path,
    *,
    max_curves: int = 12,
) -> None:
    """
    Micro-average ROC plus per-class OvR curves for the most frequent classes (same style as CNN run).
    """
    if num_classes < 2:
        return
    Y = label_binarize(y_true, classes=np.arange(num_classes))
    if Y.size == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    try:
        fpr, tpr, _ = roc_curve(Y.ravel(), y_prob.ravel())
        auc_micro = float(auc(fpr, tpr))
        ax.plot(fpr, tpr, label=f"micro-average ROC (area ≈ {auc_micro:.3f})", linewidth=2.5)
    except ValueError:
        pass

    counts = np.bincount(y_true, minlength=num_classes)
    order = np.argsort(-counts)
    plotted = 0
    for c in order:
        if plotted >= max_curves:
            break
        if c >= Y.shape[1]:
            continue
        y_bin = Y[:, c]
        if y_bin.sum() < 1 or (1 - y_bin).sum() < 1:
            continue
        try:
            fpr_c, tpr_c, _ = roc_curve(y_bin, y_prob[:, c])
            auc_c = float(auc(fpr_c, tpr_c))
            ax.plot(fpr_c, tpr_c, alpha=0.35, label=f"class {c} (AUC≈{auc_c:.2f})")
            plotted += 1
        except ValueError:
            continue

    ax.plot([0, 1], [0, 1], "k--", alpha=0.35)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (one-vs-rest); micro-average + top-supported classes")
    ax.legend(loc="lower right", fontsize=6)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
