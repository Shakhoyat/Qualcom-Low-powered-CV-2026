"""
Evaluate a Qwen2-VL model (base or fine-tuned) on the Track 3 detection task.

Usage:
    # Evaluate fine-tuned merged model
    python evaluate.py \
        --model_path /path/to/qwen2vl_merged \
        --val_json annotations_val.json \
        --output eval_results.json

    # Evaluate base model (zero-shot baseline)
    python evaluate.py \
        --model_path Qwen/Qwen2-VL-2B-Instruct \
        --val_json annotations_val.json \
        --output eval_baseline.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# Add scripts dir to path so we can import prompts
sys.path.insert(0, str(Path(__file__).parent))
from prompts import CRITERIA, PROMPT_1, PROMPT_2, SYSTEM_PROMPT, validate_output


def load_model(model_path: str, device: str):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )
    model.eval()
    return model, processor


@torch.no_grad()
def run_inference(image_path: str, model, processor, device: str) -> dict | None:
    """Two-step inference: reasoning → JSON. Returns parsed dict or None."""
    image = Image.open(image_path).convert("RGB")

    # ---- Step 1: chain-of-thought reasoning ----
    messages_1 = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT_1},
            ],
        },
    ]
    text_1 = processor.apply_chat_template(messages_1, tokenize=False, add_generation_prompt=True)
    inputs_1 = processor(text=[text_1], images=[image], return_tensors="pt").to(device)

    out_1 = model.generate(
        **inputs_1,
        max_new_tokens=600,
        temperature=0.1,
        do_sample=True,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    reasoning = processor.decode(
        out_1[0][inputs_1["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

    # ---- Step 2: structured JSON extraction ----
    messages_2 = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT_1},
            ],
        },
        {"role": "assistant", "content": reasoning},
        {"role": "user", "content": PROMPT_2},
    ]
    text_2 = processor.apply_chat_template(messages_2, tokenize=False, add_generation_prompt=True)
    inputs_2 = processor(text=[text_2], images=[image], return_tensors="pt").to(device)

    out_2 = model.generate(
        **inputs_2,
        max_new_tokens=512,
        temperature=0.0,
        do_sample=False,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    raw_json = processor.decode(
        out_2[0][inputs_2["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

    # Strip markdown fences if model adds them
    if raw_json.startswith("```"):
        parts = raw_json.split("```")
        raw_json = parts[1] if len(parts) > 1 else raw_json
        if raw_json.startswith("json"):
            raw_json = raw_json[4:].strip()

    try:
        parsed = json.loads(raw_json)
        return parsed, reasoning
    except json.JSONDecodeError:
        return None, reasoning


def compute_metrics(results: list[dict]) -> dict:
    """Compute detection accuracy, per-criterion score rates, and JSON validity."""
    valid = [r for r in results if r["valid"]]
    total = len(results)
    n_valid = len(valid)

    if n_valid == 0:
        return {"error": "No valid outputs"}

    correct = sum(1 for r in valid if r["pred_label"] == r["true_label"])
    accuracy = correct / total  # penalize invalid outputs as wrong

    # Per-criterion: what fraction of outputs flag each criterion as suspicious
    criterion_flag_rate = {}
    for c in CRITERIA:
        flags_when_ai = []
        flags_when_real = []
        for r in valid:
            for item in r["output"]["per_criterion"]:
                if item["criterion"] == c:
                    if r["true_label"] == "AI-Generated":
                        flags_when_ai.append(item["score"])
                    else:
                        flags_when_real.append(item["score"])
        criterion_flag_rate[c] = {
            "flag_rate_when_ai": sum(flags_when_ai) / max(len(flags_when_ai), 1),
            "flag_rate_when_real": sum(flags_when_real) / max(len(flags_when_real), 1),
        }

    return {
        "total_samples": total,
        "valid_json": n_valid,
        "json_validity_rate": n_valid / total,
        "detection_accuracy": accuracy,
        "correct": correct,
        "per_criterion_flag_rates": criterion_flag_rate,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--val_json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0, help="Evaluate first N samples only")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path} on {args.device}...")
    model, processor = load_model(args.model_path, args.device)

    with open(args.val_json) as f:
        val_data = json.load(f)

    if args.limit:
        val_data = val_data[: args.limit]

    print(f"Evaluating {len(val_data)} samples...")

    results = []
    for i, sample in enumerate(val_data):
        img_path = sample["image"]
        true_label = sample["label"]

        t0 = time.time()
        output, reasoning = run_inference(img_path, model, processor, args.device)
        elapsed = time.time() - t0

        valid = output is not None and validate_output(output)
        pred_label = output.get("overall_likelihood", "INVALID") if output else "INVALID"

        result = {
            "image": img_path,
            "true_label": true_label,
            "pred_label": pred_label,
            "correct": pred_label == true_label,
            "valid": valid,
            "output": output,
            "reasoning": reasoning,
            "inference_time_s": round(elapsed, 2),
        }
        results.append(result)

        status = "OK" if valid else "FAIL"
        correct_str = "✓" if pred_label == true_label else "✗"
        print(f"  [{i+1}/{len(val_data)}] {status} {correct_str} pred={pred_label} true={true_label} ({elapsed:.1f}s)")

    metrics = compute_metrics(results)

    output_data = {
        "model": args.model_path,
        "n_samples": len(val_data),
        "metrics": metrics,
        "per_sample": results,
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Total samples:      {metrics['total_samples']}")
    print(f"  Valid JSON:         {metrics['valid_json']} ({metrics['json_validity_rate']:.1%})")
    print(f"  Detection accuracy: {metrics['detection_accuracy']:.3f} ({metrics['correct']}/{metrics['total_samples']})")
    print("\n  Per-criterion flag rates (AI images → should be high, Real → should be low):")
    for c, rates in metrics["per_criterion_flag_rates"].items():
        print(f"    {c[:40]:40s}  AI={rates['flag_rate_when_ai']:.2f}  Real={rates['flag_rate_when_real']:.2f}")
    print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()
