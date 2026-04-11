"""
Pipeline 2 — Zoom-In GPT-4o annotation strategy.

Two-stage prompting:
  Stage 1 (global scan): GPT-4o performs a free-form global analysis of the image.
  Stage 2 (local justify): GPT-4o fills in each criterion using the global scan as context.

Key differences from auto_annotate.py (Pipeline 1):
  - Two-stage Zoom-In strategy (reduces hallucination)
  - Evidence strings capped at 30 words per criterion
  - Uses response_format={"type": "json_object"} for reliable JSON
  - Criteria in global-to-local order (matching prompts_p2.py)
  - Targets 2,500 high-quality samples (quality > quantity)

Usage:
    python auto_annotate_p2.py annotate \
        --input_dir ../data/fakeclue/images \
        --label AI-Generated \
        --output ../data/annotations_p2_ai.json \
        --limit 1250

    python auto_annotate_p2.py annotate \
        --input_dir ../data/coco/val2017 \
        --label Real \
        --output ../data/annotations_p2_real.json \
        --limit 1250

    python auto_annotate_p2.py merge \
        --inputs ../data/annotations_p2_ai.json ../data/annotations_p2_real.json \
        --output ../data/annotations_p2 \
        --val_split 0.1
"""

import argparse
import base64
import json
import random
import time
from pathlib import Path

CRITERIA_P2 = [
    "Perspective & Spatial Relationships",
    "Physical & Common Sense Logic",
    "Lighting & Shadows Consistency",
    "Human & Biological Structure Integrity",
    "Material & Object Details",
    "Texture & Resolution",
    "Edges & Boundaries",
    "Text & Symbols",
]

STAGE1_SYSTEM = (
    "You are an expert forensic image analyst. "
    "Describe what you see in this image and identify any anomalies or "
    "suspicious features that might suggest it is AI-generated. "
    "Be specific about spatial, physical, lighting, and structural observations. "
    "Keep your response to 3-5 sentences."
)

STAGE2_SYSTEM = (
    "You are an expert forensic image analyst. "
    "You have been given a global analysis of an image. "
    "Your task is to produce a structured JSON forensic report. "
    "Each evidence string MUST be under 30 words — specific and observational, not generic."
)

STAGE2_USER_TEMPLATE = """Image label (ground truth): {label}

Global scan notes:
{global_notes}

Based on this analysis, produce a JSON forensic report. The JSON must have this exact structure:
{{
  "overall_likelihood": "{label}",
  "per_criterion": [
    {{"criterion": "Perspective & Spatial Relationships", "score": <0 or 1>, "evidence": "<under 30 words>"}},
    {{"criterion": "Physical & Common Sense Logic", "score": <0 or 1>, "evidence": "<under 30 words>"}},
    {{"criterion": "Lighting & Shadows Consistency", "score": <0 or 1>, "evidence": "<under 30 words>"}},
    {{"criterion": "Human & Biological Structure Integrity", "score": <0 or 1>, "evidence": "<under 30 words>"}},
    {{"criterion": "Material & Object Details", "score": <0 or 1>, "evidence": "<under 30 words>"}},
    {{"criterion": "Texture & Resolution", "score": <0 or 1>, "evidence": "<under 30 words>"}},
    {{"criterion": "Edges & Boundaries", "score": <0 or 1>, "evidence": "<under 30 words>"}},
    {{"criterion": "Text & Symbols", "score": <0 or 1>, "evidence": "<under 30 words>"}}
  ]
}}

Rules:
- score=1 means SUSPICIOUS/anomalous for that criterion, score=0 means natural/clean.
- For label=AI-Generated: expect multiple score=1 entries.
- For label=Real: most scores should be 0 unless there is a genuine visual ambiguity.
- overall_likelihood MUST exactly match the label: "{label}"
- Evidence must be specific — name the actual visual observation, not a generic description.
- Respond with ONLY the JSON object, no markdown fences."""


def encode_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def stage1_global_scan(client, image_path: Path, retries: int = 3) -> str | None:
    """Get free-form global analysis of the image."""
    b64 = encode_image(image_path)
    ext = image_path.suffix.lstrip(".").lower()
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=200,
                messages=[
                    {"role": "system", "content": STAGE1_SYSTEM},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime};base64,{b64}",
                                    "detail": "low",
                                },
                            },
                            {"type": "text", "text": "Analyze this image for AI-generated artifacts."},
                        ],
                    },
                ],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt == retries - 1:
                print(f"  [stage1 fail] {image_path.name}: {e}")
                return None
            time.sleep(2 ** attempt)
    return None


def stage2_local_justify(
    client, image_path: Path, label: str, global_notes: str, retries: int = 3
) -> dict | None:
    """Fill in per-criterion JSON using global notes as context."""
    b64 = encode_image(image_path)
    ext = image_path.suffix.lstrip(".").lower()
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"

    user_msg = STAGE2_USER_TEMPLATE.format(label=label, global_notes=global_notes)

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=700,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": STAGE2_SYSTEM},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime};base64,{b64}",
                                    "detail": "low",
                                },
                            },
                            {"type": "text", "text": user_msg},
                        ],
                    },
                ],
            )
            raw = resp.choices[0].message.content.strip()
            parsed = json.loads(raw)

            # Enforce label consistency
            parsed["overall_likelihood"] = label

            # Validate structure
            if "per_criterion" not in parsed or len(parsed["per_criterion"]) != 8:
                raise ValueError(f"Wrong per_criterion count: {len(parsed.get('per_criterion', []))}")

            # Enforce criterion order matches CRITERIA_P2
            for i, item in enumerate(parsed["per_criterion"]):
                if item.get("criterion") != CRITERIA_P2[i]:
                    raise ValueError(f"Criterion mismatch at {i}: got {item.get('criterion')}")
                if item.get("score") not in (0, 1):
                    raise ValueError(f"Invalid score at {i}: {item.get('score')}")
                # Enforce token budget: truncate evidence at 200 chars (~30 words)
                if len(item.get("evidence", "")) > 200:
                    item["evidence"] = item["evidence"][:200].rsplit(" ", 1)[0]

            return parsed

        except Exception as e:
            if attempt == retries - 1:
                print(f"  [stage2 fail] {image_path.name}: {e}")
                return None
            time.sleep(2 ** attempt)
    return None


def annotate_one_zoom_in(client, image_path: Path, label: str) -> dict | None:
    """Full Zoom-In pipeline: stage1 → stage2 → validated record."""
    global_notes = stage1_global_scan(client, image_path)
    if global_notes is None:
        return None

    output = stage2_local_justify(client, image_path, label, global_notes)
    if output is None:
        return None

    return {
        "image": str(image_path.resolve()),
        "label": label,
        "output": output,
        "global_notes": global_notes,
    }


def run_annotation(
    input_dir: Path,
    label: str,
    output_path: Path,
    limit: int,
    rate_limit_rpm: int,
) -> list[dict]:
    import os
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Load checkpoint if exists
    checkpoint_path = output_path.with_suffix(".checkpoint.json")
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            records = json.load(f)
        done_paths = {r["image"] for r in records}
        print(f"Resuming from checkpoint: {len(records)} already done.")
    else:
        records = []
        done_paths = set()

    # Collect images
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    images = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in exts)
    if limit:
        images = images[:limit]
    images = [p for p in images if str(p.resolve()) not in done_paths]
    print(f"Annotating {len(images)} images with label={label} ...")

    interval = 60.0 / rate_limit_rpm
    checkpoint_every = 25

    for i, img_path in enumerate(images):
        t0 = time.time()
        record = annotate_one_zoom_in(client, img_path, label)
        if record:
            records.append(record)

        status = "OK" if record else "SKIP"
        print(f"  [{i+1}/{len(images)}] {status} {img_path.name}")

        if (i + 1) % checkpoint_every == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(records, f)
            print(f"  Checkpoint saved ({len(records)} records)")

        elapsed = time.time() - t0
        sleep_time = max(0, interval - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

    # Save final output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} records → {output_path}")

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    return records


def merge_and_split(
    input_paths: list[Path],
    output_prefix: Path,
    val_split: float,
) -> None:
    all_records = []
    for p in input_paths:
        with open(p) as f:
            all_records.extend(json.load(f))

    random.seed(42)
    random.shuffle(all_records)

    n_val = max(1, int(len(all_records) * val_split))
    val_records = all_records[:n_val]
    train_records = all_records[n_val:]

    train_path = output_prefix.parent / (output_prefix.name + "_p2_train.json")
    val_path = output_prefix.parent / (output_prefix.name + "_p2_val.json")

    with open(train_path, "w") as f:
        json.dump(train_records, f, indent=2)
    with open(val_path, "w") as f:
        json.dump(val_records, f, indent=2)

    print(f"Train: {len(train_records)} → {train_path}")
    print(f"Val:   {len(val_records)} → {val_path}")

    # Print label distribution
    for split_name, records in [("Train", train_records), ("Val", val_records)]:
        ai_count = sum(1 for r in records if r["label"] == "AI-Generated")
        real_count = sum(1 for r in records if r["label"] == "Real")
        print(f"  {split_name}: AI={ai_count}, Real={real_count}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")

    ann = sub.add_parser("annotate")
    ann.add_argument("--input_dir", required=True, type=Path)
    ann.add_argument("--label", required=True, choices=["AI-Generated", "Real"])
    ann.add_argument("--output", required=True, type=Path)
    ann.add_argument("--limit", type=int, default=1250)
    ann.add_argument("--rpm", type=int, default=60, dest="rate_limit_rpm")

    mrg = sub.add_parser("merge")
    mrg.add_argument("--inputs", nargs="+", required=True, type=Path)
    mrg.add_argument("--output", required=True, type=Path)
    mrg.add_argument("--val_split", type=float, default=0.1)

    args = parser.parse_args()

    if args.command == "annotate":
        run_annotation(
            args.input_dir,
            args.label,
            args.output,
            args.limit,
            args.rate_limit_rpm,
        )
    elif args.command == "merge":
        merge_and_split(args.inputs, args.output, args.val_split)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
