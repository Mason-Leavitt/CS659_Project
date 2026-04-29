"""
TensorFlow HOG + linear head matching train_hog_svm.py (skimage HOG, grayscale path).

Used only for exporting the classical HOG+SVM pipeline to TFLite. Input images must be
float32 RGB in [0, 1], same convention as train_export_tflite.py / the Android app.
"""
from __future__ import annotations

import math
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional, Tuple

import numpy as np
import tensorflow as tf

# skimage.color.rgb2gray (ITU-R BT.601)
_RGB_TO_GRAY = tf.constant([0.2125, 0.7154, 0.0721], dtype=tf.float32)


def _fmt_duration_hms(seconds: float) -> str:
    """Human-readable duration for ETA / elapsed (non-negative seconds)."""
    if not math.isfinite(seconds) or seconds < 0:
        return "?"
    if seconds < 60:
        return f"{seconds:.0f}s"
    mi, s = divmod(int(seconds), 60)
    if mi < 60:
        return f"{mi}m {s}s"
    h, mi = divmod(mi, 60)
    if h < 24:
        return f"{h}h {mi}m {s}s"
    d, h = divmod(h, 24)
    return f"{d}d {h}h {mi}m"


def _eta_clock_utc(eta_s: float) -> str:
    """Approximate finish time in UTC if ETA is sane."""
    if not math.isfinite(eta_s) or eta_s < 0 or eta_s > 86400 * 14:
        return ""
    t = datetime.now(timezone.utc) + timedelta(seconds=float(eta_s))
    return f" (~{t.strftime('%Y-%m-%d %H:%M:%S')} UTC)"


def hog_feature_length(
    img_size: int,
    orientations: int,
    pixels_per_cell: int,
    cells_per_block: int,
) -> int:
    """Feature dimension for skimage-style HOG (L2-Hys, full image window)."""
    n_cells = img_size // pixels_per_cell
    if n_cells * pixels_per_cell != img_size:
        raise ValueError(
            f"img_size ({img_size}) must be divisible by pixels_per_cell ({pixels_per_cell})"
        )
    n_blocks = (n_cells - cells_per_block) + 1
    return n_blocks * n_blocks * cells_per_block * cells_per_block * orientations


def hog_single_image(gray_2d: tf.Tensor, orientations: int, c_row: int, c_col: int, b_row: int, b_col: int) -> tf.Tensor:
    """
    Grayscale single image [H, W]. Spatial dims must be known at graph build time.
    Matches skimage.feature.hog(..., block_norm='L2-Hys') for grayscale.
    """
    img = tf.convert_to_tensor(gray_2d, tf.float32)
    g_row = tf.pad(img[2:, :] - img[:-2, :], [[1, 1], [0, 0]])
    g_col = tf.pad(img[:, 2:] - img[:, :-2], [[0, 0], [1, 1]])
    s_row = int(img.shape[0])
    s_col = int(img.shape[1])
    magnitude = tf.sqrt(g_col * g_col + g_row * g_row)
    orientation = (tf.math.atan2(g_row, g_col) * (180.0 / math.pi)) % 180.0
    n_cells_row = s_row // c_row
    n_cells_col = s_col // c_col
    ys = tf.range(s_row)[:, None]
    xs = tf.range(s_col)[None, :]
    r_i = ys // c_row
    c_i = xs // c_col
    nop = 180.0 / float(orientations)
    hists: list[tf.Tensor] = []
    for i in range(orientations):
        o_start = nop * (i + 1)
        o_end = nop * i
        m = (orientation >= o_end) & (orientation < o_start)
        contrib = tf.where(m, magnitude, tf.zeros_like(magnitude))
        seg_id = r_i * n_cells_col + c_i
        num_seg = n_cells_row * n_cells_col
        sums = tf.math.unsorted_segment_sum(
            tf.reshape(contrib, [-1]),
            tf.reshape(seg_id, [-1]),
            num_seg,
        )
        h = tf.reshape(sums, [n_cells_row, n_cells_col]) / float(c_row * c_col)
        hists.append(h)
    hist = tf.stack(hists, axis=-1)
    n_blocks_row = (n_cells_row - b_row) + 1
    n_blocks_col = (n_cells_col - b_col) + 1
    eps = 1e-5
    parts: list[tf.Tensor] = []
    for rr in range(n_blocks_row):
        for cc in range(n_blocks_col):
            block = hist[rr : rr + b_row, cc : cc + b_col, :]
            b = block / tf.sqrt(tf.reduce_sum(block**2) + eps**2)
            b = tf.minimum(b, 0.2)
            b = b / tf.sqrt(tf.reduce_sum(b**2) + eps**2)
            parts.append(tf.reshape(b, [-1]))
    return tf.concat(parts, axis=0)


class _RgbToGrayLayer(tf.keras.layers.Layer):
    """BT.601 grayscale; Keras 3 requires TF ops inside a Layer, not on Input KerasTensors."""

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return tf.reduce_sum(x * _RGB_TO_GRAY, axis=-1, keepdims=True)


class _StandardizeFeaturesLayer(tf.keras.layers.Layer):
    """(x - mean) / scale for HOG row vectors."""

    def __init__(self, mean: np.ndarray, scale: np.ndarray, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        m = np.asarray(mean, dtype=np.float32).reshape(1, -1)
        s = np.asarray(scale, dtype=np.float32).reshape(1, -1)
        s = np.where(s < 1e-12, 1.0, s)
        self._mean = tf.constant(m, dtype=tf.float32)
        self._scale = tf.constant(s, dtype=tf.float32)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return (x - self._mean) / self._scale


class HogLinearTfliteModelFactory:
    """Builds a Keras model: RGB [0,1] -> HOG -> standardize -> linear logits."""

    def __init__(
        self,
        *,
        img_size: int,
        orientations: int,
        pixels_per_cell: int,
        cells_per_block: int,
        mean: np.ndarray,
        scale: np.ndarray,
        coef: np.ndarray,
        intercept: np.ndarray,
    ) -> None:
        self.img_size = img_size
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, -1)
        self.scale = np.asarray(scale, dtype=np.float32).reshape(1, -1)
        self.coef = np.asarray(coef, dtype=np.float32)  # (n_classes, n_features)
        self.intercept = np.asarray(intercept, dtype=np.float32).reshape(-1)

        fd = hog_feature_length(img_size, orientations, pixels_per_cell, cells_per_block)
        if self.mean.shape[1] != fd or self.scale.shape[1] != fd:
            raise ValueError(
                f"Scaler dim {self.mean.shape[1]} != expected HOG dim {fd} "
                f"(img_size={img_size}, cell={pixels_per_cell}, block={cells_per_block})"
            )
        if self.coef.shape[1] != fd:
            raise ValueError(f"LinearSVC coef_ second dim {self.coef.shape[1]} != HOG dim {fd}")
        self._feat_dim = fd

    def build(self) -> tf.keras.Model:
        h = self.img_size
        w = self.img_size
        inp = tf.keras.Input(shape=(h, w, 3), dtype=tf.float32, name="image_rgb_01")
        gray = _RgbToGrayLayer(name="rgb_to_gray")(inp)

        def _hog_batch(batch: tf.Tensor) -> tf.Tensor:
            return tf.map_fn(
                lambda g: hog_single_image(
                    g[..., 0],
                    self.orientations,
                    self.pixels_per_cell,
                    self.pixels_per_cell,
                    self.cells_per_block,
                    self.cells_per_block,
                ),
                batch,
                fn_output_signature=tf.TensorSpec(shape=(self._feat_dim,), dtype=tf.float32),
            )

        hog_feat = tf.keras.layers.Lambda(_hog_batch, name="hog")(gray)
        z = _StandardizeFeaturesLayer(self.mean, self.scale, name="standardize_hog")(hog_feat)
        n_classes = int(self.coef.shape[0])
        logits = tf.keras.layers.Dense(
            n_classes,
            activation=None,
            use_bias=True,
            kernel_initializer=tf.keras.initializers.Constant(np.transpose(self.coef)),
            bias_initializer=tf.keras.initializers.Constant(self.intercept),
            trainable=False,
            dtype=tf.float32,
            name="linear_ovr",
        )(z)
        return tf.keras.Model(inp, logits, name="hog_svm_linear")


def extract_hog_matrix_from_paths_tf(
    paths: list[str],
    *,
    img_size: int,
    orientations: int,
    pixels_per_cell: int,
    cells_per_block: int,
    batch_size: int = 64,
    log_fn: Optional[Callable[[str], None]] = None,
) -> np.ndarray:
    """
    Compute HOG feature matrix using TensorFlow (runs on GPU when visible).

    Preprocessing matches this module's TFLite graph: decode image, resize to (img_size, img_size),
    RGB in [0, 1], grayscale via BT.601, then ``hog_single_image`` (same as skimage HOG math used
    elsewhere in hog_tf). This order differs slightly from skimage ``rgb2gray`` then ``resize`` in
    ``train_hog_svm.py`` CPU mode; use CPU path for exact parity with older runs.

    Args:
        paths: Image file paths.
        batch_size: Images per TF batch (tune for VRAM).
        log_fn: Optional ``callable(str)`` for progress (e.g. train_hog_svm._log).
    """
    n = len(paths)
    if n == 0:
        raise ValueError("paths is empty")

    feat_dim = hog_feature_length(img_size, orientations, pixels_per_cell, cells_per_block)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for d in gpus:
                tf.config.experimental.set_memory_growth(d, True)
        except Exception:
            pass
        if log_fn:
            log_fn(f"TF HOG: using GPU ({len(gpus)} device(s)) for feature extraction")
    elif log_fn:
        log_fn("TF HOG: no GPU visible — HOG runs on CPU (still faster than skimage for large jobs)")

    ds = tf.data.Dataset.from_tensor_slices(np.array(paths, dtype=object))

    def _load(path: tf.Tensor) -> tf.Tensor:
        raw = tf.io.read_file(path)
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, [img_size, img_size], antialias=True)
        return tf.cast(img, tf.float32) * (1.0 / 255.0)

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    n_batches = (n + batch_size - 1) // batch_size
    if log_fn:
        log_fn(
            f"  TF HOG: {n} images in ~{n_batches} batches (batch_size={batch_size}). "
            "The first batch can take several minutes (TensorFlow traces/compiles the HOG graph + disk decode)—"
            "this is normal; you will see a line as soon as it finishes."
        )

    c_row, c_col = pixels_per_cell, pixels_per_cell
    b_row, b_col = cells_per_block, cells_per_block
    out_blocks: list[np.ndarray] = []
    done = 0
    t0 = time.perf_counter()
    # Frequent enough updates without spamming: ~every 1000–2500 images on large jobs
    log_step = max(500, min(2500, n // 80)) if n >= 800 else max(1, n // 10)
    last_logged = 0
    batch_idx = 0

    for batch in ds:
        batch_idx += 1
        gray = tf.reduce_sum(batch * _RGB_TO_GRAY, axis=-1, keepdims=True)
        feats = tf.map_fn(
            lambda g: hog_single_image(g[..., 0], orientations, c_row, c_col, b_row, b_col),
            gray,
            fn_output_signature=tf.TensorSpec(shape=(feat_dim,), dtype=tf.float32),
        )
        out_blocks.append(feats.numpy())
        done += int(feats.shape[0])
        elapsed = time.perf_counter() - t0
        rate = done / elapsed if elapsed > 0 else 0.0
        eta_s = (n - done) / rate if rate > 0 else 0.0

        if not log_fn:
            continue

        batches_left = n_batches - batch_idx
        eta_str = _fmt_duration_hms(eta_s)
        elapsed_str = _fmt_duration_hms(elapsed)
        clock = _eta_clock_utc(eta_s)
        bl_txt = f"{batches_left} batches left" if batches_left > 0 else "last batch"

        # Always log when the first batch completes (user may have waited minutes with no other lines)
        if batch_idx == 1:
            log_fn(
                f"  TF HOG: batch 1/{n_batches} done — {done}/{n} images ({100.0 * done / n:.1f}%) "
                f"in {elapsed:.1f}s (~{rate:.1f} img/s). "
                f"ETA ~{eta_str}{clock} remaining ({bl_txt}); elapsed so far {elapsed_str}"
            )
            last_logged = done
            continue
        if done - last_logged >= log_step or done == n:
            last_logged = done
            log_fn(
                f"  TF HOG: batch {batch_idx}/{n_batches} — {done}/{n} ({100.0 * done / n:.1f}%) — "
                f"{rate:.1f} img/s | elapsed {elapsed_str} | ETA ~{eta_str}{clock} | {bl_txt}"
            )

    return np.vstack(out_blocks)


def export_keras_to_tflite_float32(model: tf.keras.Model, out_path: str) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)


def pipeline_weights_for_tflite(
    pipe,
    *,
    img_size: int,
    hog_orientations: int,
    hog_pixels_per_cell: int,
    hog_cells_per_block: int,
) -> Tuple[HogLinearTfliteModelFactory, int]:
    """
    Extract mean, scale, coef, intercept for TFLite export from either:

    - sklearn ``Pipeline(StandardScaler, LinearSVC)`` (CPU-trained), or
    - a duck-typed object with ``.scaler`` (``StandardScaler``), ``.coef_`` (n_classes, n_features),
      ``.intercept_`` (n_classes,) — e.g. ``HogLinearHeadModel`` from ``train_hog_svm.py`` (TF-trained head).

    Returns (factory, feature_dim).
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    fd = hog_feature_length(img_size, hog_orientations, hog_pixels_per_cell, hog_cells_per_block)

    scaler = None
    coef: np.ndarray | None = None
    intercept: np.ndarray | None = None

    if isinstance(pipe, Pipeline):
        clf = pipe.named_steps.get("clf")
        scaler = pipe.named_steps.get("scaler")
        if not isinstance(scaler, StandardScaler) or not isinstance(clf, LinearSVC):
            raise TypeError("Pipeline must be StandardScaler + LinearSVC for TFLite export")
        coef = np.asarray(clf.coef_, dtype=np.float32)
        intercept = np.asarray(clf.intercept_, dtype=np.float32).reshape(-1)
    elif hasattr(pipe, "scaler") and hasattr(pipe, "coef_") and hasattr(pipe, "intercept_"):
        scaler = pipe.scaler
        if not isinstance(scaler, StandardScaler):
            raise TypeError("Expected .scaler to be a StandardScaler for TFLite export")
        coef = np.asarray(pipe.coef_, dtype=np.float32)
        intercept = np.asarray(pipe.intercept_, dtype=np.float32).reshape(-1)
    else:
        raise TypeError(
            "Expected sklearn Pipeline(StandardScaler, LinearSVC) or an object with "
            ".scaler, .coef_, .intercept_ (e.g. HogLinearHeadModel from train_hog_svm.py --linear-head tf)"
        )

    if coef.shape[1] != fd:
        raise ValueError(
            f"Trained model feature size {coef.shape[1]} does not match "
            f"HOG settings (expected {fd} for img_size={img_size}, "
            f"pixels_per_cell={hog_pixels_per_cell}, cells_per_block={hog_cells_per_block}, "
            f"orientations={hog_orientations}). Retrain with matching --img_size and HOG args."
        )

    mean = scaler.mean_.astype(np.float32)
    scale = scaler.scale_.astype(np.float32)
    scale = np.where(scale < 1e-12, 1.0, scale)
    factory = HogLinearTfliteModelFactory(
        img_size=img_size,
        orientations=hog_orientations,
        pixels_per_cell=hog_pixels_per_cell,
        cells_per_block=hog_cells_per_block,
        mean=mean,
        scale=scale,
        coef=coef,
        intercept=intercept,
    )
    return factory, fd
