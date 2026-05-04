#!/usr/bin/env python3
"""
Dataset manifest helpers for PlantNet-style folder layouts.
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from agent.model_imports import REPO_ROOT, ensure_model_project_on_path

SUPPORTED_SPLITS = ("test", "train", "val", "validation")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IGNORED_NAMES = {
    ".DS_Store",
    "plantnet300K_metadata.json",
    "plantnet300K_species_id_2_name.json",
}


def _coerce_dataset_path(dataset_path: str | Path) -> Path:
    path = Path(dataset_path)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path.resolve()


def _is_species_dir(path: Path) -> bool:
    return path.is_dir() and not path.name.startswith(".")


def _looks_like_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and path.name not in IGNORED_NAMES


def _candidate_image_roots(dataset_root: Path, split: str) -> list[Path]:
    split = split.strip().lower()
    candidates: list[Path] = []
    if split not in SUPPORTED_SPLITS:
        return candidates

    names = [split]
    if split == "validation":
        names.append("val")
    elif split == "val":
        names.append("validation")

    if dataset_root.is_dir():
        for name in names:
            candidates.append(dataset_root / "images" / name)
            candidates.append(dataset_root / name)
        if dataset_root.name == "images":
            for name in names:
                candidates.append(dataset_root / name)
        if dataset_root.name in names:
            candidates.append(dataset_root)

        # dataset_sample/<species_id>/*.jpg
        direct_species_dirs = [p for p in dataset_root.iterdir() if _is_species_dir(p)]
        if direct_species_dirs and any(any(_looks_like_image(img) for img in sub.iterdir()) for sub in direct_species_dirs):
            candidates.append(dataset_root)

    seen: set[Path] = set()
    deduped: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(resolved)
    return deduped


def _resolve_image_root(dataset_root: Path, split: str) -> Path:
    for candidate in _candidate_image_roots(dataset_root, split):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve a '{split}' image folder from dataset path: {dataset_root}"
    )


def _extract_display_name(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, dict):
        for key in (
            "scientificNameWithoutAuthor",
            "scientificName",
            "species_name",
            "display_name",
            "name",
            "label",
        ):
            inner = value.get(key)
            if isinstance(inner, str) and inner.strip():
                return inner.strip()
    return None


def _load_json_label_map(dataset_root: Path) -> dict[str, str]:
    candidate_roots = [dataset_root, dataset_root.parent, dataset_root.parent.parent]
    for root in candidate_roots:
        mapping_path = root / "plantnet300K_species_id_2_name.json"
        if not mapping_path.is_file():
            continue
        try:
            raw = json.loads(mapping_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        mapping: dict[str, str] = {}
        if isinstance(raw, dict):
            for key, value in raw.items():
                display = _extract_display_name(value)
                if display:
                    mapping[str(key).strip()] = display
        elif isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    continue
                key = item.get("species_id") or item.get("id") or item.get("speciesId")
                display = _extract_display_name(item)
                if key is not None and display:
                    mapping[str(key).strip()] = display
        if mapping:
            return mapping
    return {}


def _load_fallback_label_map() -> dict[str, str]:
    ensure_model_project_on_path()
    import app_config

    export_path = app_config.DEFAULT_EXPORT_LABELS_PATH
    scientific_path = app_config.DEFAULT_SCIENTIFIC_LABELS_PATH
    if not export_path.is_file() or not scientific_path.is_file():
        return {}

    export_labels = [
        line.strip()
        for line in export_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    scientific_labels = [
        line.strip()
        for line in scientific_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(export_labels) != len(scientific_labels):
        return {}

    return {
        export_labels[idx]: scientific_labels[idx]
        for idx in range(len(export_labels))
        if export_labels[idx]
    }


def load_species_id_name_map(dataset_path: str | Path | None = None) -> dict[str, str]:
    dataset_root = _coerce_dataset_path(dataset_path) if dataset_path is not None else None
    mapping = _load_json_label_map(dataset_root) if dataset_root is not None else {}
    if mapping:
        return mapping
    return _load_fallback_label_map()


def _iter_manifest_rows(image_root: Path, split: str, id_name_map: dict[str, str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    class_dirs = sorted([path for path in image_root.iterdir() if _is_species_dir(path)], key=lambda p: p.name)
    for class_dir in class_dirs:
        label = class_dir.name.strip()
        if not label:
            continue
        display_label = id_name_map.get(label, label)
        for image_path in sorted(class_dir.iterdir(), key=lambda p: p.name.lower()):
            if not _looks_like_image(image_path):
                continue
            rows.append(
                {
                    "image_path": str(image_path.resolve()),
                    "label": label,
                    "display_label": display_label,
                    "split": split,
                    "class_dir": class_dir.name,
                }
            )
    return rows


def _normalize_sampling_mode(sampling_mode: str) -> str:
    mode = str(sampling_mode or "balanced").strip().lower()
    if mode not in {"sorted", "random", "balanced"}:
        raise ValueError("sampling_mode must be one of: sorted, random, balanced")
    return mode


def _group_rows_by_label(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        label = str(row.get("label", "")).strip()
        grouped.setdefault(label, []).append(row)
    return grouped


def _apply_class_cap(
    grouped_rows: dict[str, list[dict[str, str]]],
    *,
    max_images_per_class: int | None,
) -> dict[str, list[dict[str, str]]]:
    if max_images_per_class is None:
        return {label: list(items) for label, items in grouped_rows.items()}
    cap = int(max_images_per_class)
    if cap <= 0:
        raise ValueError("max_images_per_class must be greater than 0 when provided.")
    return {
        label: list(items[:cap])
        for label, items in grouped_rows.items()
    }


def _sample_rows(
    rows: list[dict[str, str]],
    *,
    max_images: int | None,
    sampling_mode: str,
    seed: int,
    max_images_per_class: int | None,
) -> list[dict[str, str]]:
    mode = _normalize_sampling_mode(sampling_mode)
    grouped_rows = _group_rows_by_label(rows)
    rng = random.Random(int(seed))

    prepared: dict[str, list[dict[str, str]]] = {}
    for label in sorted(grouped_rows):
        items = list(grouped_rows[label])
        if mode in {"random", "balanced"}:
            rng.shuffle(items)
        prepared[label] = items

    prepared = _apply_class_cap(prepared, max_images_per_class=max_images_per_class)

    if mode == "sorted":
        sampled = [
            row
            for label in sorted(prepared)
            for row in prepared[label]
        ]
    elif mode == "random":
        sampled = [
            row
            for label in sorted(prepared)
            for row in prepared[label]
        ]
        rng.shuffle(sampled)
    else:
        sampled = []
        active_labels = [label for label in sorted(prepared) if prepared[label]]
        rng.shuffle(active_labels)
        offsets = {label: 0 for label in active_labels}
        while active_labels:
            next_active: list[str] = []
            for label in active_labels:
                idx = offsets[label]
                items = prepared[label]
                if idx >= len(items):
                    continue
                sampled.append(items[idx])
                offsets[label] = idx + 1
                if max_images is not None and len(sampled) >= int(max_images):
                    return sampled
                if offsets[label] < len(items):
                    next_active.append(label)
            active_labels = next_active

    if max_images is not None:
        limit = int(max_images)
        if limit <= 0:
            raise ValueError("max_images must be greater than 0 when provided.")
        sampled = sampled[:limit]
    return sampled


def _manifest_metadata_path(manifest_path: Path) -> Path:
    return manifest_path.with_name(f"{manifest_path.stem}.metadata.json")


def load_manifest_metadata(manifest_path: str | Path) -> dict[str, Any]:
    target = Path(manifest_path)
    metadata_path = _manifest_metadata_path(target)
    if not metadata_path.is_file():
        return {
            "success": False,
            "manifest_path": str(target),
            "metadata_path": str(metadata_path),
            "error": "Manifest metadata file does not exist.",
        }
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Manifest metadata file is not a JSON object.")
        data["success"] = True
        data["manifest_path"] = str(target)
        data["metadata_path"] = str(metadata_path)
        return data
    except Exception as exc:
        return {
            "success": False,
            "manifest_path": str(target),
            "metadata_path": str(metadata_path),
            "error": str(exc),
        }


def scan_dataset_folder(dataset_path: str, split: str = "test") -> dict[str, Any]:
    resolved_dataset_path = _coerce_dataset_path(dataset_path)
    result: dict[str, Any] = {
        "success": False,
        "dataset_path": str(resolved_dataset_path),
        "resolved_image_root": None,
        "split": split,
        "num_images": 0,
        "num_classes": 0,
        "manifest_path": None,
        "preview": [],
        "warning": None,
        "error": None,
    }

    try:
        if split not in SUPPORTED_SPLITS:
            raise ValueError(f"Unsupported split '{split}'. Expected one of: {', '.join(SUPPORTED_SPLITS)}")
        if not resolved_dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {resolved_dataset_path}")

        image_root = _resolve_image_root(resolved_dataset_path, split)
        id_name_map = load_species_id_name_map(resolved_dataset_path)
        class_dirs = sorted([path for path in image_root.iterdir() if _is_species_dir(path)], key=lambda p: p.name)
        if not class_dirs:
            raise FileNotFoundError(
                f"Resolved split folder exists but no class directories were found: {image_root}"
            )
        rows = _iter_manifest_rows(image_root, split, id_name_map)
        if not rows:
            raise FileNotFoundError(
                f"Resolved split folder exists but contains no supported image files: {image_root}"
            )

        result["resolved_image_root"] = str(image_root)
        result["num_images"] = len(rows)
        result["num_classes"] = len({row["label"] for row in rows})
        result["preview"] = rows[:5]
        result["success"] = True
        return result
    except Exception as exc:
        result["error"] = str(exc)
        return result


def build_manifest(
    dataset_path: str,
    split: str = "test",
    out_csv: str | None = None,
    max_images: int | None = None,
    sampling_mode: str = "balanced",
    seed: int = 42,
    max_images_per_class: int | None = None,
) -> dict[str, Any]:
    summary = scan_dataset_folder(dataset_path=dataset_path, split=split)
    if not summary.get("success"):
        return summary

    image_root = Path(str(summary["resolved_image_root"]))
    dataset_root = _coerce_dataset_path(dataset_path)
    id_name_map = load_species_id_name_map(dataset_root)
    rows = _iter_manifest_rows(image_root, split, id_name_map)
    sampling_mode = _normalize_sampling_mode(sampling_mode)
    sampled_rows = _sample_rows(
        rows,
        max_images=max_images,
        sampling_mode=sampling_mode,
        seed=int(seed),
        max_images_per_class=max_images_per_class,
    )
    class_distribution = dict(sorted(Counter(row["label"] for row in sampled_rows).items(), key=lambda item: item[0]))

    result: dict[str, Any] = {
        "success": True,
        "dataset_path": str(dataset_root),
        "resolved_image_root": str(image_root),
        "split": split,
        "num_images": len(sampled_rows),
        "num_classes": len(class_distribution),
        "manifest_path": None,
        "preview": sampled_rows[:5],
        "sampling_mode": sampling_mode,
        "seed": int(seed),
        "max_images_per_class": int(max_images_per_class) if max_images_per_class is not None else None,
        "class_distribution": class_distribution,
        "warning": None,
        "error": None,
    }

    try:
        if not sampled_rows:
            result["success"] = False
            result["error"] = "No images were selected for the manifest."
            return result
        if out_csv is not None:
            target = Path(out_csv)
            if not target.is_absolute():
                target = (REPO_ROOT / target).resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=["image_path", "label", "display_label", "split", "class_dir"],
                )
                writer.writeheader()
                writer.writerows(sampled_rows)
            result["manifest_path"] = str(target)
            metadata_path = _manifest_metadata_path(target)
            metadata_path.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
            result["metadata_path"] = str(metadata_path)
        return result
    except Exception as exc:
        result["success"] = False
        result["error"] = str(exc)
        return result
