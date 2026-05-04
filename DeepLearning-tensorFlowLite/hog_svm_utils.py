#!/usr/bin/env python3
"""
Reusable HOG preprocessing helpers shared by training and inference.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize


def load_image_gray01(image_path: str | Path, img_size: int = 224) -> np.ndarray:
    """
    Load an image, convert it to grayscale, normalize to [0, 1], and resize.

    This matches the preprocessing currently used by train_hog_svm.py.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        img = imread(str(path))
    except Exception as exc:
        raise ValueError(f"Could not read image: {path}") from exc

    if img.ndim == 2:
        gray = img.astype(np.float64)
        if gray.max() > 1.0:
            gray /= 255.0
        return resize(gray, (img_size, img_size), anti_aliasing=True)

    if img.ndim != 3:
        raise ValueError(f"Unsupported image shape for HOG preprocessing: {img.shape}")

    if img.shape[2] == 4:
        img = img[..., :3]

    rgb = img.astype(np.float64)
    if rgb.max() > 1.0:
        rgb /= 255.0
    gray = rgb2gray(rgb)
    return resize(gray, (img_size, img_size), anti_aliasing=True)


def extract_hog_features(
    image_gray01: np.ndarray,
    orientations: int = 9,
    pixels_per_cell: int = 16,
    cells_per_block: int = 2,
    block_norm: str = "L2-Hys",
) -> np.ndarray:
    """Extract a flat HOG feature vector from a grayscale image."""
    return hog(
        image_gray01,
        orientations=orientations,
        pixels_per_cell=(pixels_per_cell, pixels_per_cell),
        cells_per_block=(cells_per_block, cells_per_block),
        block_norm=block_norm,
        visualize=False,
        feature_vector=True,
    )


def prepare_hog_features(
    image_path: str | Path,
    img_size: int = 224,
    orientations: int = 9,
    pixels_per_cell: int = 16,
    cells_per_block: int = 2,
    block_norm: str = "L2-Hys",
) -> np.ndarray:
    """Load an image and return its HOG feature vector."""
    gray = load_image_gray01(image_path, img_size=img_size)
    return extract_hog_features(
        gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm=block_norm,
    )


def load_hog_labels(labels_path: str | Path) -> list[str]:
    """Load one-label-per-line HOG label files, skipping blanks and comments."""
    path = Path(labels_path)
    if not path.is_file():
        raise FileNotFoundError(f"Labels not found: {path}")

    labels: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text and not text.startswith("#"):
            labels.append(text)
    if not labels:
        raise ValueError(f"No labels in {path}")
    return labels
