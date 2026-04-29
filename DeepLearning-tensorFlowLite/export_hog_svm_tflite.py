#!/usr/bin/env python3
"""
Export an existing HOG + linear classifier from train_hog_svm.py to TFLite.

Accepts a joblib containing either ``Pipeline(StandardScaler, LinearSVC)`` or
``HogLinearHeadModel`` (``--linear-head tf``). The joblib does not store HOG geometry; pass the
same ``--img_size`` and HOG flags you used when training so feature dimensions match.

Example:
  python export_hog_svm_tflite.py --joblib_path result/hog_svm_*/hog_svm_model.joblib \\
    --out_tflite plant_classifier_traditional.tflite --img_size 224
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from hog_tf import export_keras_to_tflite_float32, pipeline_weights_for_tflite


def main() -> None:
    p = argparse.ArgumentParser(description="Export HOG+LinearSVM joblib to TFLite.")
    p.add_argument(
        "--joblib_path",
        type=Path,
        required=True,
        help="Saved model (sklearn Pipeline or HogLinearHeadModel from train_hog_svm.py)",
    )
    p.add_argument("--out_tflite", type=Path, required=True, help="Output .tflite path")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--hog_orientations", type=int, default=9)
    p.add_argument("--hog_pixels_per_cell", type=int, default=16)
    p.add_argument("--hog_cells_per_block", type=int, default=2)
    args = p.parse_args()

    jp = args.joblib_path.expanduser().resolve()
    if not jp.is_file():
        raise SystemExit(f"Not a file: {jp}")

    pipe = joblib.load(jp)
    factory, _ = pipeline_weights_for_tflite(
        pipe,
        img_size=args.img_size,
        hog_orientations=args.hog_orientations,
        hog_pixels_per_cell=args.hog_pixels_per_cell,
        hog_cells_per_block=args.hog_cells_per_block,
    )
    model = factory.build()
    out = args.out_tflite.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    export_keras_to_tflite_float32(model, str(out))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
