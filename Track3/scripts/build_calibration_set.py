"""
Build a domain-matched calibration set for AIMET quantization.

Usage:
    python build_calibration_set.py \
        --annotations annotations_train.json \
        --output_dir calibration_data/domain_mixed \
        --n_real 150 --n_ai 150

This is run AFTER auto_annotate.py and BEFORE the AIMET QPM tutorial.
The resulting directory replaces the COCO-only calibration used in the baseline.
"""

import argparse
import json
import random
import shutil
from pathlib import Path


def build(annotations_path: Path, output_dir: Path, n_real: int, n_ai: int) -> None:
    with open(annotations_path) as f:
        records = json.load(f)

    real_records = [r for r in records if r["label"] == "Real"]
    ai_records   = [r for r in records if r["label"] == "AI-Generated"]

    print(f"Available — Real: {len(real_records)}, AI: {len(ai_records)}")

    if len(real_records) < n_real:
        print(f"[warn] Only {len(real_records)} real images, requested {n_real}. Using all.")
        n_real = len(real_records)
    if len(ai_records) < n_ai:
        print(f"[warn] Only {len(ai_records)} AI images, requested {n_ai}. Using all.")
        n_ai = len(ai_records)

    random.seed(42)
    selected = random.sample(real_records, n_real) + random.sample(ai_records, n_ai)
    random.shuffle(selected)

    output_dir.mkdir(parents=True, exist_ok=True)

    failed = 0
    for i, record in enumerate(selected):
        src = Path(record["image"])
        if not src.exists():
            print(f"  [warn] Missing: {src}")
            failed += 1
            continue
        dst = output_dir / f"calib_{i:04d}_{record['label'].replace('-','').lower()}{src.suffix}"
        shutil.copy(src, dst)

    n_copied = len(selected) - failed
    print(f"\nCalibration set built: {n_copied} images → {output_dir}")
    print(f"  Real: {n_real}, AI-Generated: {n_ai}, Failed: {failed}")

    # Write a manifest so we can audit later
    manifest = {
        "source_annotations": str(annotations_path),
        "n_real": n_real,
        "n_ai": n_ai,
        "total": n_copied,
        "files": [f"calib_{i:04d}_{r['label'].replace('-','').lower()}{Path(r['image']).suffix}"
                  for i, r in enumerate(selected) if Path(r["image"]).exists()],
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest → {output_dir / 'manifest.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output_dir",  required=True, type=Path)
    parser.add_argument("--n_real", type=int, default=150)
    parser.add_argument("--n_ai",   type=int, default=150)
    args = parser.parse_args()

    build(args.annotations, args.output_dir, args.n_real, args.n_ai)


if __name__ == "__main__":
    main()
