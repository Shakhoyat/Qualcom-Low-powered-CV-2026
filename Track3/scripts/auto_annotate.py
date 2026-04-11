"""
Auto-annotate images with GPT-4o to generate per-criterion evidence.
Run this locally (or on Colab) ONCE to build the training dataset.

Usage:
    python auto_annotate.py --input_dir ./data/cifake/train --label AI-Generated \
        --output annotations_ai.json --limit 5000

    python auto_annotate.py --input_dir ./data/coco/val2017 --label Real \
        --output annotations_real.json --limit 5000

Then merge:
    python auto_annotate.py --merge annotations_ai.json annotations_real.json \
        --output annotations_train.json --val_split 0.1

Cost estimate: ~$0.002/image with GPT-4o → 10,000 images ≈ $20
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

from prompts import CRITERIA, VALID_LABELS, validate_output

# ---------------------------------------------------------------------------
# GPT-4o annotation prompt (separate from the inference prompts — this one
# tells GPT-4o what to generate, not what Qwen should say)
# ---------------------------------------------------------------------------
ANNOTATION_SYSTEM = (
    "You are an expert forensic image analyst. "
    "You provide precise, evidence-based analysis of whether images are AI-generated or real photographs. "
    "Your analysis is specific, citing concrete visual details you observe."
)

ANNOTATION_USER_TEMPLATE = (
    "Analyze this image and provide a structured forensic assessment.\n\n"
    "For each of these 8 criteria, provide:\n"
    "- score: 0 if this aspect looks natural/authentic, 1 if it shows AI-generation artifacts\n"
    "- evidence: a specific, detailed observation citing what you actually see (1-2 sentences, "
    "mention specific regions, colors, or features — avoid generic statements)\n\n"
    "Criteria:\n"
    "1. Lighting & Shadows Consistency\n"
    "2. Edges & Boundaries\n"
    "3. Texture & Resolution\n"
    "4. Perspective & Spatial Relationships\n"
    "5. Physical & Common Sense Logic\n"
    "6. Text & Symbols\n"
    "7. Human & Biological Structure Integrity\n"
    "8. Material & Object Details\n\n"
    "Known label: this image is {label}. Score accordingly — if Real, most scores should be 0 "
    "(natural); if AI-Generated, at least some scores should be 1 (suspicious).\n\n"
    "Also set overall_likelihood to exactly \"{label}\".\n\n"
    "Respond ONLY with valid JSON:\n"
    "{{\n"
    '  "overall_likelihood": "{label}",\n'
    '  "per_criterion": [\n'
    '    {{"criterion": "Lighting & Shadows Consistency", "score": 0, "evidence": "..."}},\n'
    "    ...\n"
    "  ]\n"
    "}}"
)


def encode_image(path: Path) -> tuple[str, str]:
    """Return (base64_string, mime_type)."""
    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode(), mime


def annotate_one(client, image_path: Path, label: str, retries: int = 3) -> dict | None:
    """Call GPT-4o on a single image. Returns the parsed JSON or None on failure."""
    import openai

    b64, mime = encode_image(image_path)
    user_text = ANNOTATION_USER_TEMPLATE.format(label=label)

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": ANNOTATION_SYSTEM},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "low"},
                            },
                            {"type": "text", "text": user_text},
                        ],
                    },
                ],
                max_tokens=900,
                temperature=0.2,
            )
            raw = response.choices[0].message.content.strip()

            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            parsed = json.loads(raw)

            # Force label consistency
            parsed["overall_likelihood"] = label

            if validate_output(parsed):
                return parsed

            print(f"  [warn] Schema invalid for {image_path.name}, attempt {attempt + 1}")

        except json.JSONDecodeError as e:
            print(f"  [warn] JSON parse error for {image_path.name}: {e}, attempt {attempt + 1}")
        except Exception as e:
            print(f"  [warn] API error for {image_path.name}: {e}, attempt {attempt + 1}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # exponential backoff

    return None


def load_checkpoint(checkpoint_path: Path) -> dict:
    """Load existing results to resume interrupted runs."""
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            data = json.load(f)
        print(f"Resuming from checkpoint: {len(data)} annotations already done.")
        return data
    return {}


def save_checkpoint(checkpoint_path: Path, results: dict) -> None:
    with open(checkpoint_path, "w") as f:
        json.dump(results, f, indent=2)


def run_annotation(
    input_dir: Path,
    label: str,
    output_path: Path,
    limit: int,
    rate_limit_rpm: int,
) -> list[dict]:
    """Annotate all images in input_dir with GPT-4o. Resume-safe."""
    import openai

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY env variable.")

    client = openai.OpenAI(api_key=api_key)

    # Collect image paths
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    all_images = sorted([p for p in input_dir.rglob("*") if p.suffix.lower() in exts])
    if limit:
        all_images = all_images[:limit]

    print(f"Found {len(all_images)} images in {input_dir} (limit={limit})")

    # Load checkpoint
    checkpoint_path = output_path.with_suffix(".checkpoint.json")
    done: dict = load_checkpoint(checkpoint_path)

    # Rate limiting
    delay = 60.0 / rate_limit_rpm
    results = []
    failed = 0

    for i, img_path in enumerate(all_images):
        key = str(img_path)

        if key in done:
            results.append(done[key])
            continue

        annotation = annotate_one(client, img_path, label)

        if annotation is not None:
            record = {
                "image": str(img_path),
                "label": label,
                "output": annotation,
                "reasoning_text": (
                    f"Forensic analysis of this {label.lower()} image:\n"
                    + "\n".join(
                        f"- {c['criterion']}: {c['evidence']}"
                        for c in annotation["per_criterion"]
                    )
                ),
            }
            done[key] = record
            results.append(record)
        else:
            failed += 1
            print(f"  [fail] Skipping {img_path.name} after all retries.")

        # Progress report
        if (i + 1) % 25 == 0:
            print(f"  Progress: {i+1}/{len(all_images)} | done={len(results)} failed={failed}")
            save_checkpoint(checkpoint_path, done)

        time.sleep(delay)

    # Final save
    save_checkpoint(checkpoint_path, done)

    print(f"\nAnnotation complete: {len(results)} OK, {failed} failed.")
    return results


def merge_and_split(
    input_files: list[Path],
    output_path: Path,
    val_split: float,
) -> None:
    """Merge annotation files and produce train/val splits."""
    import random

    all_records = []
    for f in input_files:
        with open(f) as fh:
            data = json.load(fh)
        if isinstance(data, list):
            all_records.extend(data)
        else:
            all_records.extend(data.values())

    random.seed(42)
    random.shuffle(all_records)

    n_val = int(len(all_records) * val_split)
    val = all_records[:n_val]
    train = all_records[n_val:]

    train_path = output_path.parent / (output_path.stem + "_train.json")
    val_path = output_path.parent / (output_path.stem + "_val.json")

    with open(train_path, "w") as f:
        json.dump(train, f, indent=2)
    with open(val_path, "w") as f:
        json.dump(val, f, indent=2)

    print(f"Merged {len(all_records)} total samples.")
    print(f"  Train: {len(train)} → {train_path}")
    print(f"  Val:   {len(val)} → {val_path}")

    # Class balance report
    ai_count = sum(1 for r in all_records if r["label"] == "AI-Generated")
    real_count = sum(1 for r in all_records if r["label"] == "Real")
    print(f"  AI-Generated: {ai_count} | Real: {real_count}")


def main():
    parser = argparse.ArgumentParser(description="Auto-annotate images with GPT-4o")
    subparsers = parser.add_subparsers(dest="command")

    # Annotate command
    ann = subparsers.add_parser("annotate", help="Annotate a directory of images")
    ann.add_argument("--input_dir", required=True, type=Path)
    ann.add_argument("--label", required=True, choices=["AI-Generated", "Real"])
    ann.add_argument("--output", required=True, type=Path)
    ann.add_argument("--limit", type=int, default=0, help="Max images (0 = all)")
    ann.add_argument("--rpm", type=int, default=100, help="Requests per minute limit")

    # Merge command
    mrg = subparsers.add_parser("merge", help="Merge annotation files and create train/val split")
    mrg.add_argument("--inputs", nargs="+", required=True, type=Path)
    mrg.add_argument("--output", required=True, type=Path)
    mrg.add_argument("--val_split", type=float, default=0.1)

    args = parser.parse_args()

    if args.command == "annotate":
        records = run_annotation(
            input_dir=args.input_dir,
            label=args.label,
            output_path=args.output,
            limit=args.limit,
            rate_limit_rpm=args.rpm,
        )
        with open(args.output, "w") as f:
            json.dump(records, f, indent=2)
        print(f"Saved {len(records)} annotations → {args.output}")

    elif args.command == "merge":
        merge_and_split(
            input_files=args.inputs,
            output_path=args.output,
            val_split=args.val_split,
        )

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
