#!/usr/bin/env python3
"""
Create a deterministic project-local PlantNet sample in a resolver-compatible layout.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = Path(r"E:\school\659\plantnet_300K\plantnet_300K")
DEFAULT_DEST = REPO_ROOT / "dataset_sample"
SUPPORTED_SPLITS = {"test", "train", "val", "validation"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PRESETS = {
    "quick": {"num_classes": 10, "images_per_class": 5},
    "robust": {"num_classes": 25, "images_per_class": 8},
    "larger": {"num_classes": 50, "images_per_class": 10},
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy a deterministic balanced PlantNet sample into dataset_sample/images/<split>/..."
    )
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--dest", default=str(DEFAULT_DEST))
    parser.add_argument("--split", default="test")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="robust")
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--images-per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-copy", type=int, default=None)
    return parser.parse_args()


def _ensure_within_repo(path: Path) -> Path:
    resolved = path.resolve()
    repo_resolved = REPO_ROOT.resolve()
    try:
        resolved.relative_to(repo_resolved)
    except ValueError as exc:
        raise ValueError(f"Destination must stay inside the project root: {repo_resolved}") from exc
    return resolved


def _source_split_dir(source_root: Path, split: str) -> Path:
    normalized = split.strip().lower()
    if normalized not in SUPPORTED_SPLITS:
        raise ValueError(f"Unsupported split '{split}'. Expected one of: {', '.join(sorted(SUPPORTED_SPLITS))}.")

    candidates = [
        source_root / "images" / normalized,
        source_root / normalized,
    ]
    if normalized == "validation":
        candidates.extend([source_root / "images" / "val", source_root / "val"])
    elif normalized == "val":
        candidates.extend([source_root / "images" / "validation", source_root / "validation"])

    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not resolve source split folder '{split}' under source dataset: {source_root}"
    )


def _iter_images(class_dir: Path) -> list[Path]:
    return [
        path for path in sorted(class_dir.iterdir(), key=lambda item: item.name.lower())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def _selected_classes_and_images(
    source_split_dir: Path,
    *,
    num_classes: int,
    images_per_class: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[str]]:
    warnings: list[str] = []
    if num_classes <= 0:
        raise ValueError("num_classes must be greater than 0.")
    if images_per_class <= 0:
        raise ValueError("images_per_class must be greater than 0.")

    qualifying: list[dict[str, object]] = []
    for class_dir in sorted(source_split_dir.iterdir(), key=lambda item: item.name):
        if not class_dir.is_dir() or class_dir.name.startswith("."):
            continue
        images = _iter_images(class_dir)
        if len(images) < images_per_class:
            continue
        qualifying.append({"species_id": class_dir.name, "class_dir": class_dir, "images": images})

    if not qualifying:
        raise FileNotFoundError(
            f"No qualifying class folders with at least {images_per_class} images were found in {source_split_dir}."
        )

    rng = random.Random(seed)
    shuffled = list(qualifying)
    rng.shuffle(shuffled)

    if len(shuffled) < num_classes:
        warnings.append(
            f"Only {len(shuffled)} qualifying classes were available, which is fewer than the requested {num_classes}."
        )
    selected_classes = shuffled[: min(num_classes, len(shuffled))]

    selections: list[dict[str, object]] = []
    for class_entry in selected_classes:
        species_id = str(class_entry["species_id"])
        class_dir = Path(class_entry["class_dir"])
        images = list(class_entry["images"])
        class_rng = random.Random(f"{seed}:{species_id}")
        class_rng.shuffle(images)
        picked = images[:images_per_class]
        selections.append(
            {
                "species_id": species_id,
                "class_dir": class_dir,
                "images": picked,
            }
        )
    return selections, warnings


def _legacy_split_dir(dest_root: Path, split: str) -> Path:
    return dest_root / split


def _destination_split_dir(dest_root: Path, split: str) -> Path:
    return dest_root / "images" / split


def _remove_existing_outputs(dest_root: Path, split: str) -> None:
    destination_split_dir = _destination_split_dir(dest_root, split)
    if destination_split_dir.exists():
        shutil.rmtree(destination_split_dir)
    for artifact_name in ("sample_manifest.csv", "sample_summary.json", "README.md"):
        artifact_path = dest_root / artifact_name
        if artifact_path.exists():
            artifact_path.unlink()
    images_dir = dest_root / "images"
    if images_dir.exists() and not any(images_dir.iterdir()):
        images_dir.rmdir()
    legacy_dir = _legacy_split_dir(dest_root, split)
    if legacy_dir.exists() and legacy_dir.is_dir() and not any(legacy_dir.iterdir()):
        legacy_dir.rmdir()


def _write_manifest(dest_root: Path, rows: list[dict[str, object]]) -> Path:
    manifest_path = dest_root / "sample_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "species_id",
                "source_path",
                "relative_dest_path",
                "filename",
                "selected_index_within_class",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def _write_summary(dest_root: Path, summary: dict[str, object]) -> Path:
    summary_path = dest_root / "sample_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    return summary_path


def _write_readme(dest_root: Path, dataset_path_to_use: str, split_to_use: str, layout: str) -> Path:
    readme_path = dest_root / "README.md"
    readme_path.write_text(
        "\n".join(
            [
                "# Dataset Sample",
                "",
                "This is a deterministic sample copied from the read-only E: drive PlantNet master dataset.",
                "It is intended for inference-only ablation and parameter sweep testing.",
                "It is not the full dataset.",
                "",
                "## App Settings",
                f"- Dataset path: `{dataset_path_to_use}`",
                f"- Split: `{split_to_use}`",
                f"- Layout: `{layout}`",
                "",
                "The image layout is compatible with the project dataset resolver.",
            ]
        ),
        encoding="utf-8",
    )
    return readme_path


def main() -> int:
    args = _parse_args()
    source_root = Path(args.source).resolve()
    dest_root = _ensure_within_repo(Path(args.dest))
    split = str(args.split).strip().lower()
    preset = PRESETS[args.preset]
    num_classes = int(args.num_classes if args.num_classes is not None else preset["num_classes"])
    images_per_class = int(
        args.images_per_class if args.images_per_class is not None else preset["images_per_class"]
    )
    total_expected = num_classes * images_per_class

    if args.max_copy is not None and total_expected > int(args.max_copy):
        raise ValueError(
            f"Requested {total_expected} images, which exceeds --max-copy {int(args.max_copy)}."
        )
    if not source_root.exists():
        raise FileNotFoundError(f"Source dataset root not found: {source_root}")

    source_split_dir = _source_split_dir(source_root, split)
    destination_split_dir = _destination_split_dir(dest_root, split)
    selections, warnings = _selected_classes_and_images(
        source_split_dir,
        num_classes=num_classes,
        images_per_class=images_per_class,
        seed=int(args.seed),
    )

    rows: list[dict[str, object]] = []
    classes_selected: list[str] = []
    images_per_class_summary: dict[str, int] = {}

    for class_entry in selections:
        species_id = str(class_entry["species_id"])
        classes_selected.append(species_id)
        picked_images = list(class_entry["images"])
        images_per_class_summary[species_id] = len(picked_images)
        for selected_index, source_image in enumerate(picked_images, start=1):
            relative_dest_path = Path("images") / split / species_id / source_image.name
            rows.append(
                {
                    "split": split,
                    "species_id": species_id,
                    "source_path": str(Path(source_image).resolve()),
                    "relative_dest_path": str(relative_dest_path).replace("\\", "/"),
                    "filename": source_image.name,
                    "selected_index_within_class": selected_index,
                }
            )

    if dest_root.exists() and any(dest_root.iterdir()) and not args.overwrite and not args.dry_run:
        canonical_split_dir = destination_split_dir
        existing_artifacts = [
            dest_root / "sample_manifest.csv",
            dest_root / "sample_summary.json",
            dest_root / "README.md",
        ]
        if canonical_split_dir.exists() and any(canonical_split_dir.rglob("*")):
            raise FileExistsError(
                f"Destination already contains sample data under {canonical_split_dir}. Use --overwrite to replace it."
            )
        if any(path.exists() for path in existing_artifacts):
            raise FileExistsError(
                f"Destination already contains sample artifacts under {dest_root}. Use --overwrite to replace them."
            )

    if not args.dry_run:
        dest_root.mkdir(parents=True, exist_ok=True)
        if args.overwrite:
            _remove_existing_outputs(dest_root, split)
        copied_images = 0
        for class_entry in selections:
            species_id = str(class_entry["species_id"])
            target_class_dir = destination_split_dir / species_id
            picked_images = list(class_entry["images"])
            for source_image in picked_images:
                target_class_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_image, target_class_dir / source_image.name)
                copied_images += 1

        dataset_path_to_use = str(dest_root)
        layout = f"images/{split}/<species_id>/*.jpg"
        manifest_path = _write_manifest(dest_root, rows)
        summary = {
            "source_root": str(source_root),
            "source_split_dir": str(source_split_dir),
            "destination_root": str(dest_root),
            "destination_split_dir": str(destination_split_dir),
            "dataset_path_to_use": dataset_path_to_use,
            "split_to_use": split,
            "layout": layout,
            "preset": args.preset,
            "seed": int(args.seed),
            "num_classes_requested": num_classes,
            "images_per_class_requested": images_per_class,
            "total_images_expected": total_expected,
            "total_images_copied": copied_images,
            "classes_selected": classes_selected,
            "images_per_class": images_per_class_summary,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "warnings": warnings,
            "dry_run": False,
        }
        summary_path = _write_summary(dest_root, summary)
        readme_path = _write_readme(dest_root, dataset_path_to_use, split, layout)
        print(f"Created sample dataset at: {dest_root}")
        print(f"Manifest: {manifest_path}")
        print(f"Summary: {summary_path}")
        print(f"README: {readme_path}")
    else:
        copied_images = len(rows)
        print("Dry run only. No files were copied.")

    print(f"Source split dir: {source_split_dir}")
    print(f"Destination split dir: {destination_split_dir}")
    print(f"Preset: {args.preset}")
    print(f"Classes selected: {len(classes_selected)}")
    print(f"Images per class: {images_per_class}")
    print(f"Total images selected: {len(rows)}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")
    print("")
    print("Use this dataset path in the app:")
    print(dest_root)
    print("")
    print("Use this split:")
    print(split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
