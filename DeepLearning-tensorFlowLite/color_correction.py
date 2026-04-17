"""
Per-image illuminant / color cast correction in RGB after rescaling to [0, 1].

Why these methods (for reports / design notes)
----------------------------------------------
Outdoor and phone photos vary in white balance and lighting. Simple *color
constancy* steps reduce that variance so the classifier sees more stable leaf
color and shading.

- **Gray world** assumes the average surface color in the scene is neutral gray.
  It scales R, G, B so their spatial means match. Fast, differentiable, easy to
  replicate on-device. Weak when one channel dominates (e.g. large sky or grass).

- **Max RGB** (Finlayson-style scaling) assumes the brightest observed R, G, B
  channel responses come from a white or neutral highlight. It rescales channels
  using per-channel maxima. Good when specular highlights exist; can misbehave
  if one channel saturates.

- We did **not** use histogram equalization on RGB or V alone as the main path:
  it often warps color distributions and hurts species discrimination compared
  to mild white-balance-style fixes.

- We did **not** add a learned color-correction network: it adds parameters and
  train/serve complexity for a class project where classical constancy is enough.

Apply the **same** `method` during training and at inference (and in any mobile
preprocessing) to avoid train/serve skew.

Tensor inputs are float32, NHWC, values in [0, 1]. Output is clipped to [0, 1].
"""
from __future__ import annotations

import tensorflow as tf

COLOR_METHODS = ("none", "gray_world", "max_rgb")


def gray_world_rgb01_bhwc(x: tf.Tensor) -> tf.Tensor:
    """Batched images BHWC float32 in [0,1]. Gray-world white balance."""
    mean_c = tf.reduce_mean(x, axis=[1, 2])  # [B, 3]
    gray = tf.reduce_mean(mean_c, axis=-1, keepdims=True)  # [B, 1]
    scale = gray / (mean_c + 1e-6)
    out = x * tf.reshape(scale, (-1, 1, 1, 3))
    return tf.clip_by_value(out, 0.0, 1.0)


def max_rgb_rgb01_bhwc(x: tf.Tensor) -> tf.Tensor:
    """Batched images BHWC float32 in [0,1]. Max-RGB style scaling."""
    max_c = tf.reduce_max(x, axis=[1, 2])  # [B, 3]
    ref = tf.reduce_max(max_c, axis=-1, keepdims=True)  # [B, 1]
    scale = ref / (max_c + 1e-6)
    out = x * tf.reshape(scale, (-1, 1, 1, 3))
    return tf.clip_by_value(out, 0.0, 1.0)


def apply_color_rgb01_bhwc(x: tf.Tensor, method: str) -> tf.Tensor:
    if method == "none":
        return x
    if method == "gray_world":
        return gray_world_rgb01_bhwc(x)
    if method == "max_rgb":
        return max_rgb_rgb01_bhwc(x)
    raise ValueError(f"Unknown color correction method {method!r}; expected one of {COLOR_METHODS}")
