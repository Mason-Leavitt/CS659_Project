"""
Load experiment hyperparameters from JSON and build stratified train/validation/test splits.

Both train_hog_svm.py and train_export_tflite.py use the same split logic when given the same
seed and fractions so report comparisons are fair.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.model_selection import train_test_split


def load_json_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_nested(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def validate_split_fractions(train_f: float, val_f: float, test_f: float) -> None:
    s = float(train_f) + float(val_f) + float(test_f)
    if abs(s - 1.0) > 1e-5:
        raise ValueError(
            f"train_fraction + validation_fraction + test_fraction must sum to 1.0, got {s:.6f}"
        )
    for name, v in [("train", train_f), ("validation", val_f), ("test", test_f)]:
        if v < 0 or v > 1:
            raise ValueError(f"Invalid {name}_fraction: {v}")


def stratified_train_val_test(
    paths: list[str],
    labels: list[int],
    *,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    random_state: int,
) -> tuple[
    list[str],
    list[str],
    list[str],
    list[int],
    list[int],
    list[int],
]:
    """
    Stratified three-way split. Order matches sklearn: first hold out test, then split remainder into train/val.

    The relative size of val within (train+val) is validation_fraction / (train_fraction + validation_fraction).
    """
    validate_split_fractions(train_fraction, validation_fraction, test_fraction)

    paths_a = np.array(paths, dtype=object)
    labels_a = np.array(labels, dtype=np.int64)

    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        paths_a,
        labels_a,
        test_size=test_fraction,
        stratify=labels_a,
        random_state=random_state,
    )
    rel_val = validation_fraction / (train_fraction + validation_fraction)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=rel_val,
        stratify=train_val_labels,
        random_state=random_state + 1,
    )

    return (
        train_paths.tolist(),
        val_paths.tolist(),
        test_paths.tolist(),
        train_labels.tolist(),
        val_labels.tolist(),
        test_labels.tolist(),
    )


def per_class_counts(labels: list[int], num_classes: int) -> list[int]:
    arr = np.array(labels, dtype=np.int64)
    return [int((arr == c).sum()) for c in range(num_classes)]


def split_summary_dict(
    *,
    train_labels: list[int],
    val_labels: list[int],
    test_labels: list[int],
    num_classes: int,
    class_names: list[str],
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, Any]:
    return {
        "seed": seed,
        "fractions": {
            "train": train_fraction,
            "validation": validation_fraction,
            "test": test_fraction,
        },
        "counts": {
            "train": len(train_labels),
            "validation": len(val_labels),
            "test": len(test_labels),
        },
        "per_class_counts": {
            "train": per_class_counts(train_labels, num_classes),
            "validation": per_class_counts(val_labels, num_classes),
            "test": per_class_counts(test_labels, num_classes),
            "class_names": class_names,
        },
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def merge_config_into_argparse_defaults(
    config: dict[str, Any],
    *,
    section: str,
) -> dict[str, Any]:
    """Flatten split.* and section.* into argparse-friendly default names."""
    out: dict[str, Any] = {}
    split = config.get("split") or {}
    if "seed" in split:
        out["seed"] = split["seed"]
    if "train_fraction" in split:
        out["train_fraction"] = split["train_fraction"]
    if "validation_fraction" in split:
        out["validation_fraction"] = split["validation_fraction"]
    if "test_fraction" in split:
        out["test_fraction"] = split["test_fraction"]

    sec = config.get(section) or {}
    for k, v in sec.items():
        out[k] = v
    return out


def snapshot_full_config(config_path: Optional[Path], runtime_overrides: dict[str, Any]) -> dict[str, Any]:
    base: dict[str, Any] = {}
    if config_path is not None and config_path.is_file():
        base = load_json_config(config_path)
    merged = {**base, **{"runtime_cli_overrides": runtime_overrides}}
    return merged


# --- Dataset layout: flat folder-per-class vs. train/val/test each containing species subfolders ---

_KNOWN_SPLIT_FOLDER_NAMES = frozenset({"train", "val", "validation", "test"})


def _direct_subdirs(data_dir: Path) -> list[str]:
    return sorted(d.name for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith("."))


def is_split_nested_class_layout(data_dir: Path) -> bool:
    """
    True when data_dir looks like Pl@ntNet / ImageFolder-with-splits::

        data_dir/train/<species_id>/...
        data_dir/val/<species_id>/...
        data_dir/test/<species_id>/...

    Top-level directory names must all be split names (train/val/validation/test), and at least
    one split must contain species subfolders. This avoids treating that layout as three classes
    named \"train\", \"val\", \"test\".
    """
    tops = _direct_subdirs(data_dir)
    if len(tops) < 2:
        return False
    if not {t.lower() for t in tops} <= _KNOWN_SPLIT_FOLDER_NAMES:
        return False
    for t in tops:
        split_path = data_dir / t
        if any(c.is_dir() and not c.name.startswith(".") for c in split_path.iterdir()):
            return True
    return False


def discover_class_folder_names(data_dir: Path) -> tuple[list[str], bool]:
    """
    Return (sorted class names, nested_split_layout).

    * Flat layout: ``data_dir/<class_name>/images...``
    * Nested layout: ``data_dir/<train|val|test>/<class_name>/images...``
    """
    if is_split_nested_class_layout(data_dir):
        species: set[str] = set()
        for t in _direct_subdirs(data_dir):
            split_dir = data_dir / t
            for c in split_dir.iterdir():
                if c.is_dir() and not c.name.startswith("."):
                    species.add(c.name)
        if len(species) < 2:
            raise ValueError(
                "Split-style layout detected (top-level train/val/test) but fewer than 2 species "
                "subfolders were found under those splits. Use a flattened folder-per-class dataset "
                "or fix the directory structure."
            )
        return sorted(species), True

    tops = _direct_subdirs(data_dir)
    if len(tops) < 2:
        raise ValueError("Need at least 2 class subfolders under --data_dir")
    return tops, False


def collect_paths_and_labels_for_classes(
    data_dir: Path,
    class_names: list[str],
    *,
    nested_split_layout: bool,
    img_exts: frozenset[str],
) -> tuple[list[str], list[int]]:
    """Pair each image path with its integer label (index in ``class_names``)."""
    idx = {n: i for i, n in enumerate(class_names)}
    paths: list[str] = []
    labels: list[int] = []

    def ingest(sub: Path, label: int) -> None:
        for f in sorted(sub.rglob("*")):
            if f.is_file() and f.suffix.lower() in img_exts:
                paths.append(str(f.resolve()))
                labels.append(label)

    if not nested_split_layout:
        for name in class_names:
            ingest(data_dir / name, idx[name])
    else:
        for split_name in _direct_subdirs(data_dir):
            split_dir = data_dir / split_name
            if not split_dir.is_dir():
                continue
            for name in class_names:
                sub = split_dir / name
                if sub.is_dir():
                    ingest(sub, idx[name])

    if not paths:
        raise ValueError("No image files found under class subfolders (check extensions and paths).")
    return paths, labels
