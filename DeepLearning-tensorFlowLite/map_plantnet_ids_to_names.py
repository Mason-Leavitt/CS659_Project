#!/usr/bin/env python3
"""
Map PlantNet numeric species IDs in a label file (e.g. plant_labels_export.txt) to scientific names.

Class index order must not change — only the text per line. Use the output as
app/src/main/assets/ml/plant_labels.txt (same line count and order as the model).

Requires plantnet300K_species_id_2_name.json from the PlantNet-300K dataset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Label file with one species id per line (e.g. plant_labels_export.txt)",
    )
    p.add_argument(
        "--species_json",
        type=Path,
        required=True,
        help="Path to plantnet300K_species_id_2_name.json",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("plant_labels_scientific.txt"),
        help="Output path (UTF-8, one scientific name per line)",
    )
    args = p.parse_args()

    # PlantNet JSON: species id string -> scientific name (or display string from the dataset file).
    mapping = json.loads(args.species_json.read_text(encoding="utf-8"))
    # Preserve one line per class in file order; skip blanks and # comments (same as app label files).
    lines = [
        ln.strip()
        for ln in args.labels.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    out_lines: list[str] = []
    missing: list[str] = []
    for sid in lines:
        name = mapping.get(sid)
        if name is None:
            missing.append(sid)
            # Keep raw id so line count/order still matches the trained model.
            out_lines.append(sid)
        else:
            out_lines.append(name)

    args.out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out.resolve()} ({len(out_lines)} lines)")
    if missing:
        print(f"Warning: {len(missing)} id(s) not in JSON — left as raw id (first few: {missing[:5]})")


if __name__ == "__main__":
    main()
