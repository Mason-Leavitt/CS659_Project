#!/usr/bin/env python3
"""
Merge Pl@ntNet-300K-style layout into one folder per class for train_hog_svm.py and train_export_tflite.py.

Expected source layout::

    <source>/
      train/<species_id>/*.jpg
      val/<species_id>/*.jpg
      test/<species_id>/*.jpg

Output layout (same as those training scripts)::

    <out>/
      <species_id>/*.jpg   # symlinks or copies from any split

Then train with::

    python train_export_tflite.py --data_dir <out> ...
    python train_hog_svm.py --data_dir <out> ...

Class names are sorted folder names (numeric species IDs sort lexically; use zero-padding in
IDs if you need a different order — PlantNet IDs are usually plain integers as strings).
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

_IMG_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"})


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] {msg}", flush=True)


def _list_image_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for f in sorted(root.rglob("*")):
        if f.is_file() and f.suffix.lower() in _IMG_EXTS:
            out.append(f)
    return out


def _parse_splits(arg: str) -> list[str]:
    parts = [p.strip().lower() for p in arg.split(",") if p.strip()]
    if not parts:
        raise SystemExit("--splits must list at least one of: train, val, test")
    norm: list[str] = []
    for p in parts:
        if p == "validation":
            p = "val"
        if p not in ("train", "val", "test"):
            raise SystemExit(f"Unknown split name: {p!r} (use train, val, test)")
        norm.append(p)
    seen: set[str] = set()
    uniq: list[str] = []
    for p in norm:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def _load_species_list(path: Path) -> list[str]:
    lines: list[str] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    if not lines:
        raise SystemExit(f"No species ids in {path}")
    return lines


def _count_images_per_species(source: Path, splits: list[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for sp_name in splits:
        split_dir = source / sp_name
        if not split_dir.is_dir():
            _log(f"warning: missing split directory: {split_dir}")
            continue
        for species_dir in sorted(split_dir.iterdir()):
            if not species_dir.is_dir() or species_dir.name.startswith("."):
                continue
            n = len(_list_image_files(species_dir))
            counts[species_dir.name] += n
    return dict(counts)


def _link_or_copy(
    src: Path,
    dst: Path,
    *,
    copy: bool,
    rel: bool,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(dst)
    if copy:
        import shutil

        shutil.copy2(src, dst)
        return
    try:
        if rel:
            rel_target = os.path.relpath(src, start=dst.parent)
            dst.symlink_to(rel_target)
        else:
            dst.symlink_to(src.resolve())
    except OSError as e:
        raise SystemExit(
            f"Symlink failed ({e}). On some systems symlink creation requires privileges; retry with --copy."
        ) from e


def main() -> None:
    p = argparse.ArgumentParser(
        description="Flatten PlantNet-300K train/val/test/<species_id>/ into <out>/<species_id>/ for folder-per-class training."
    )
    p.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Root that contains train/, val/, and/or test/ subfolders (e.g. plantnet_300K/images).",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output root: one subfolder per species id with merged images.",
    )
    p.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated splits to merge (default: train,val,test).",
    )
    p.add_argument(
        "--species-list",
        type=Path,
        default=None,
        help="Optional file: one species id per line (# comments allowed). Only these classes are created.",
    )
    p.add_argument(
        "--max-species",
        type=int,
        default=None,
        metavar="N",
        help="If set, keep at most N species that have at least one image (after --species-list if any).",
    )
    p.add_argument(
        "--min-images",
        type=int,
        default=1,
        metavar="K",
        help="Only include species with at least this many total images across merged splits (default: 1).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for --max-species subsampling (default: 42).",
    )
    p.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of symlinks (slower, more disk; works without symlink permission).",
    )
    p.add_argument(
        "--absolute-symlinks",
        action="store_true",
        help="Use absolute symlink targets (default: relative targets under --out).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned counts and exit without creating files.",
    )
    args = p.parse_args()

    source = args.source.resolve()
    out_root = args.out.resolve()
    splits = _parse_splits(args.splits)

    if not source.is_dir():
        raise SystemExit(f"Not a directory: {source}")

    counts = _count_images_per_species(source, splits)
    candidates = sorted(k for k, v in counts.items() if v >= args.min_images)

    if args.species_list is not None:
        wanted = set(_load_species_list(args.species_list.resolve()))
        candidates = sorted(set(candidates) & wanted)
        missing = sorted(wanted - set(counts.keys()))
        if missing:
            _log(f"warning: {len(missing)} id(s) from --species-list have no images under --source (showing up to 10): {missing[:10]}")

    if not candidates:
        raise SystemExit("No species left after filters; check --source, --splits, --species-list, --min-images.")

    if args.max_species is not None:
        if args.max_species < 1:
            raise SystemExit("--max-species must be >= 1")
        rng = random.Random(args.seed)
        if len(candidates) > args.max_species:
            candidates = sorted(rng.sample(candidates, args.max_species))

    manifest: dict[str, object] = {
        "source": str(source),
        "out": str(out_root),
        "splits": splits,
        "seed": args.seed,
        "copy": bool(args.copy),
        "species_ids": candidates,
        "n_species": len(candidates),
        "min_images": args.min_images,
    }

    if args.dry_run:
        total = sum(counts[sid] for sid in candidates)
        manifest["approx_total_images"] = total
        print(json.dumps(manifest, indent=2))
        _log(f"dry-run: would link/copy ~{total} images into {len(candidates)} class folders under {out_root}")
        return

    out_root.mkdir(parents=True, exist_ok=True)

    linked = 0
    skipped_dup = 0
    use_rel = not args.absolute_symlinks and not args.copy

    for sid in candidates:
        class_dir = out_root / sid
        class_dir.mkdir(parents=True, exist_ok=True)
        used_names: set[str] = set()
        collision = 0

        for split_name in splits:
            src_dir = source / split_name / sid
            if not src_dir.is_dir():
                continue
            for f in _list_image_files(src_dir):
                base = f.name
                name = base
                if name in used_names:
                    name = f"{split_name}_{base}"
                while name in used_names:
                    collision += 1
                    stem, suf = f.stem, f.suffix
                    name = f"{split_name}_{stem}_{collision}{suf}"
                used_names.add(name)
                dst = class_dir / name
                try:
                    _link_or_copy(f.resolve(), dst, copy=args.copy, rel=use_rel)
                    linked += 1
                except FileExistsError:
                    skipped_dup += 1

    manifest["images_linked_or_copied"] = linked
    manifest["skipped_duplicate_paths"] = skipped_dup

    man_path = out_root / "flatten_plantnet_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _log(f"Wrote {man_path}")
    _log(f"Done: {linked} files into {len(candidates)} classes under {out_root}")


if __name__ == "__main__":
    main()
