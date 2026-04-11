# Track 3 Pipeline 1 — Qwen2-VL-2B QLoRA · 9,000 samples · Two-pass inference

## What this is

Qwen2-VL-2B-Instruct fine-tuned with 4-bit QLoRA (r=64, rank-stabilized).
Two-step inference: chain-of-thought reasoning → structured JSON extraction.
Standard approach, highest data volume (9,000 GPT-4o annotated samples ~$20).

## Architecture

```
Input image
    │
    ▼
Qwen2-VL-2B-Instruct (QLoRA r=64, NF4 4-bit)
    │
    ├─ Pass 1: PROMPT_1 → chain-of-thought reasoning (max 600 tokens)
    │
    └─ Pass 2: PROMPT_2 (reasoning as context) → JSON (max 512 tokens)
               {"overall_likelihood": "AI-Generated"|"Real",
                "per_criterion": [...8 criteria...]}
```

## Files

| File | Location |
|---|---|
| Training notebook | `kaggle/track3_training.py` |
| Prompts (source of truth) | `scripts/prompts.py` |
| GPT-4o annotation | `scripts/auto_annotate.py` |
| Evaluation | `scripts/evaluate.py` |
| Calibration builder | `scripts/build_calibration_set.py` |
| Experiment log | `scripts/experiment_log.json` |
| Run order | `RUN_ORDER.md` |

All scripts are in `Track3/scripts/` and `Track3/kaggle/`.

---

## Step 1 — Annotate training data (local machine, ~$20)

```bash
cd Track3/scripts/
export OPENAI_API_KEY="sk-..."

# AI-Generated images (CIFAKE train/FAKE — 5,000 images)
python auto_annotate.py annotate \
    --input_dir ../data/cifake/train/FAKE \
    --label AI-Generated \
    --output ../data/annotations_ai_cifake.json \
    --limit 5000 --rpm 100

# Real images (CIFAKE train/REAL — 5,000 images)
python auto_annotate.py annotate \
    --input_dir ../data/cifake/train/REAL \
    --label Real \
    --output ../data/annotations_real_cifake.json \
    --limit 5000 --rpm 100

# Merge + 90/10 split
python auto_annotate.py merge \
    --inputs ../data/annotations_ai_cifake.json ../data/annotations_real_cifake.json \
    --output ../data/annotations \
    --val_split 0.1

# Output: annotations_train.json (9,000) + annotations_val.json (1,000)
```

**Cost:** 10,000 images × $0.002 = **~$20**

---

## Step 2 — Upload to Kaggle

1. Zip `annotations_train.json` + `annotations_val.json`
2. Create Kaggle dataset: **track3-annotations**
3. Upload `Track3/kaggle/track3_training.py` as a notebook script

---

## Step 3 — Set Kaggle secret

1. Notebook → Add-ons → Secrets
2. Add secret:
   - **Name:** `QAI_HUB_API_TOKEN`
   - **Value:** `wyfb5d78py0bgow6gz10krdqyd8qq0l9wlfw9swc`

---

## Step 4 — Run training on Kaggle

Settings:
- Accelerator: **GPU T4 x2**
- Internet: **On**
- Input dataset: `track3-annotations`

Click **Run All**.

**Expected runtime:** 6–8 hours on T4 x2 (9,000 samples × 3 epochs).

**Download from `/kaggle/working/`:**
- `qwen2vl_merged/` (~4.5 GB) → upload to Drive

---

## Step 5 — Build calibration set (local)

```bash
python build_calibration_set.py \
    --annotations ../data/annotations_train.json \
    --output_dir  ../data/calibration_domain_mixed \
    --n_real 150 --n_ai 150
```

---

## Step 6 — Evaluate merged model

```bash
python evaluate.py \
    --model_path ../models/qwen2vl_merged \
    --val_json   ../data/annotations_val.json \
    --output     ../experiments/run_p1_eval.json \
    --limit 100
```

Check: `detection_accuracy` must be > 0.80 to proceed to quantization.

---

## Step 7 — AIMET W4A8 PTQ (QPM tutorial)

1. Install QPM: https://qpm.qualcomm.com
2. Download **"Tutorial for Qwen2_VL_2b (IoT)"**
3. Docker image:
   ```
   artifacts.codelinaro.org/codelinaro-aimet/aimet-dev:1.34.0.torch-gpu-pt113
   ```
4. In `Example1A_veg.ipynb`, set:
   ```python
   model_path           = "/path/to/qwen2vl_merged"
   calibration_image_dir = "/path/to/data/calibration_domain_mixed"
   ```
5. Run: Example 1A → 1B → 2A → 2B in order

---

## Step 8 — AIHub simulation

```bash
export QAI_HUB_API_TOKEN="wyfb5d78py0bgow6gz10krdqyd8qq0l9wlfw9swc"

python inference_multi.py \
    --model_path ./ar*-ar*-cl*/weight_sharing_model_1_of_1.serialized.bin \
    --device "Snapdragon 8 Gen 5 QRD" \
    --inputs_dir ./baseline_outputs/inputs/ \
    --output_dir ./aihub_outputs/ \
    --api_token $QAI_HUB_API_TOKEN

python compute_score_multi_aihub.py \
    --baseline_dir ./baseline_outputs/ \
    --aihub_dir    ./aihub_outputs/ \
    --output_report score_report_p1.json
```

---

## Step 9 — Submit

1. Share compile job with `lowpowervision@gmail.com`
2. Package ZIP per checklist in `Track3/README.md`
3. Submit at https://lpcv.ai/2026LPCVC/submission/track3

---

## Key hyperparameters

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2-VL-2B-Instruct` |
| Adapter | QLoRA NF4 4-bit |
| LoRA rank | 64 |
| LoRA alpha | 128 |
| use_rslora | True |
| Epochs | 3 |
| Batch size | 1 (T4) |
| Grad accum | 16 (eff. batch 16) |
| LR | 2e-4, cosine |
| Optimizer | paged_adamw_8bit |
| Training samples | 9,000 |
| Inference | Two-pass (reasoning→JSON) |
| Quant target | W4A8 PTQ (AIMET) |

## Expected results

| Metric | Expected |
|---|---|
| Detection accuracy | 84–94% |
| JSON validity rate | >95% |
| Latency | Within 15 TPS gate |
