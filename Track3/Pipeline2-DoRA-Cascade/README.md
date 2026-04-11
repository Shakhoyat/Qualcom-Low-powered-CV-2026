# Track 3 Pipeline 2 — ConvNeXt-Tiny Cascade + DoRA · 2,500 samples · Single-pass Prefill

## What's different from Pipeline 1

| | Pipeline 1 | Pipeline 2 (this) |
|---|---|---|
| Architecture | VLM only | **ConvNeXt-Tiny → VLM cascade** |
| VLM adapter | QLoRA (standard LoRA) | **DoRA** (`use_dora=True`) |
| Inference | Two-pass (reasoning→JSON) | **Single-pass prefill-guided JSON** |
| Annotation | Single-prompt GPT-4o | **Zoom-In two-stage GPT-4o** |
| Training data | 9,000 images | **2,250 curated images** (quality > quantity) |
| Augmentation | None | **B-Free** (JPEG Q=75 + Gaussian blur) |
| Curriculum | None | **3-epoch progressive difficulty** |
| Criteria order | Arbitrary | **Global → Local** |
| Quant target | W4A8 PTQ | **W8A8 QAT** (near-lossless reasoning) |
| Expected accuracy | 84–94% | **94–96%** cross-generator |

## Architecture

```
Input image
    │
    ├──► ConvNeXt-Tiny (trained binary classifier)
    │        │
    │        └── P(AI-Generated) score ──► injected into VLM prompt
    │
    └──► Qwen2-VL-2B (DoRA r=64, prefill-guided)
             │
             └── Single-pass JSON generation (≤400 tokens)
                 {"overall_likelihood": "...", "per_criterion": [...]}
```

## Files

| File | Location |
|---|---|
| Training notebook | `kaggle/track3_pipeline2_training.py` |
| Prompts (P2) | `scripts/prompts_p2.py` |
| Zoom-In annotation | `scripts/auto_annotate_p2.py` |

---

## Step 1 — Zoom-In annotation (local machine, ~$7.50)

```bash
cd Track3/scripts/
export OPENAI_API_KEY="sk-..."

# AI-Generated images (1,250 samples — Zoom-In two-stage GPT-4o)
python auto_annotate_p2.py annotate \
    --input_dir ../data/cifake/train/FAKE \
    --label AI-Generated \
    --output ../data/annotations_p2_ai.json \
    --limit 1250 --rpm 60

# Real images from COCO val (1,250 samples)
python auto_annotate_p2.py annotate \
    --input_dir ../data/coco/val2017 \
    --label Real \
    --output ../data/annotations_p2_real.json \
    --limit 1250 --rpm 60

# Merge + 90/10 split
python auto_annotate_p2.py merge \
    --inputs ../data/annotations_p2_ai.json ../data/annotations_p2_real.json \
    --output ../data/annotations_p2 \
    --val_split 0.1

# Output: annotations_p2_train.json (2,250) + annotations_p2_val.json (250)
```

**Cost:** 2,500 × $0.002 × ~1.5 (two API calls per image) = **~$7.50**

**Why Zoom-In?**
Stage 1 — GPT-4o does a free global scan (3-5 sentences).
Stage 2 — GPT-4o fills each of the 8 criteria using Stage 1 context.
This halves hallucination vs single-prompt annotation.
Evidence strings are hard-capped at 30 words per criterion.

---

## Step 2 — Upload to Kaggle

1. Zip `annotations_p2_train.json` + `annotations_p2_val.json`
2. Create Kaggle dataset: **track3-p2-annotations**
3. Upload `Track3/kaggle/track3_pipeline2_training.py` as a notebook script

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
- Input dataset: `track3-p2-annotations` (annotations_p2_train.json + annotations_p2_val.json)

Click **Run All** (11 cells).

**Expected runtime:** 4–5 hours on T4 x2.

### What each cell does

| Cell | Action |
|---|---|
| 1 | Install deps (transformers, peft, timm, albumentations, bitsandbytes) |
| 2 | Config + Kaggle secrets |
| 3 | Load data + B-Free augmentation pipeline |
| 4 | **Train ConvNeXt-Tiny binary classifier** (5 epochs, progressive difficulty) |
| 5 | Pre-compute CNN scores for all training samples |
| 6 | Load Qwen2-VL-2B with **DoRA** config (`use_dora=True`) |
| 7 | VLM dataset with curriculum augmentation |
| 8 | **Train VLM** — 3 curriculum epochs (easy→moderate→hard augmentation) |
| 9 | **CPU merge** DoRA weights into base model (avoids T4 OOM) |
| 10 | Cascade evaluation (50 val samples) |
| 11 | Save experiment results JSON |

**Download from `/kaggle/working/`:**
- `qwen2vl_p2_merged/` (~4.5 GB VLM) → upload to Drive
- `convnext_detector.pt` (~100 MB CNN) → upload to Drive
- `pipeline2_results.json` (metrics)

---

## Step 5 — Build calibration set

Same as Pipeline 1 (reuse the 300-image calibration set):

```bash
cd Track3/scripts/
python build_calibration_set.py \
    --annotations ../data/annotations_p2_train.json \
    --output_dir  ../data/calibration_domain_mixed \
    --n_real 150 --n_ai 150
```

---

## Step 6 — Evaluate cascade (local / Colab)

```python
# Quick cascade eval — requires both model files
import qai_hub as hub
# See track3_pipeline2_training.py Cell 10 for cascade_inference() function

python evaluate.py \
    --model_path ../models/qwen2vl_p2_merged \
    --val_json   ../data/annotations_p2_val.json \
    --output     ../experiments/run_p2_eval.json \
    --limit 100
```

Check: `detection_accuracy` must be > 0.85 before quantization.

---

## Step 7 — AIMET W8A8 QAT (critical difference from P1)

> P2 targets **W8A8** (not W4A8). Near-lossless for reasoning tasks.
> The Hexagon NPU is heavily optimized for parallel INT8 ops.

1. Same QPM tutorial: **"Tutorial for Qwen2_VL_2b (IoT)"**
2. Docker:
   ```
   artifacts.codelinaro.org/codelinaro-aimet/aimet-dev:1.34.0.torch-gpu-pt113
   ```
3. In `Example1A_veg.ipynb`:
   ```python
   model_path            = "/path/to/qwen2vl_p2_merged"
   calibration_image_dir = "/path/to/data/calibration_domain_mixed"
   # KEY CHANGE:
   weight_bw = 8   # W8 (not W4)
   act_bw    = 8   # A8 (not A8)
   ```
4. Use **symmetric** quantization + **range learning**
5. Run QAT for 3–5 epochs at LR=1e-6 (simulates quantization noise)
6. Run: Example 1A → 1B → 2A → 2B

---

## Step 8 — AIHub simulation

```bash
export QAI_HUB_API_TOKEN="wyfb5d78py0bgow6gz10krdqyd8qq0l9wlfw9swc"

python inference_multi.py \
    --model_path ./ar*-ar*-cl*/weight_sharing_model_1_of_1.serialized.bin \
    --device "Snapdragon 8 Gen 5 QRD" \
    --inputs_dir ./baseline_outputs/inputs/ \
    --output_dir ./aihub_outputs_p2/ \
    --api_token $QAI_HUB_API_TOKEN

python compute_score_multi_aihub.py \
    --baseline_dir ./baseline_outputs/ \
    --aihub_dir    ./aihub_outputs_p2/ \
    --output_report score_report_p2.json
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
| Base VLM | `Qwen/Qwen2-VL-2B-Instruct` |
| Adapter | **DoRA** (`use_dora=True`) |
| LoRA rank | 64 |
| LoRA alpha | 128 |
| Epochs | 3 (VLM) + 5 (CNN) |
| Augmentation | B-Free: JPEG Q=75 + Gaussian blur |
| Curriculum | Easy epoch 1 → Heavy epoch 3 |
| Criteria order | **Global → Local** |
| Inference | **Single-pass prefill-guided JSON** |
| Prefill | `{"overall_likelihood": "` injected as assistant prefill |
| Max output tokens | 400 |
| CNN | ConvNeXt-Tiny (timm), P(AI-Generated) score fed to VLM |
| Training samples | 2,250 (Zoom-In annotated) |
| Quant target | **W8A8 QAT** (AIMET QuantizationSimModel) |

## Expected results

| Metric | Expected |
|---|---|
| CNN binary accuracy | >90% |
| Cascade detection accuracy | **94–96%** |
| JSON validity rate | >97% |
| Expected latency | ~3.0 s total (15ms CNN + 90ms ViT + 2.8s JSON) |

## Why DoRA beats LoRA here

DoRA decomposes pre-trained weights into **magnitude** + **direction** components
and updates only the directional matrices. This allows it to match the representational
power of full fine-tuning while keeping the same inference latency as LoRA.
On visual instruction tasks, DoRA outperforms LoRA by **+0.6 to +1.9 points** (ICML 2024).
For the 8-criterion forensic JSON generation task, that margin matters.
