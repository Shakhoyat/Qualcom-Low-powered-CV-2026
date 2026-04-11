# Track 3 — Execution Order
### Run these steps in exact order. Each step's output feeds the next.

---

## Step 0 — One-time setup (local machine)

```bash
cd Track3/

# Install local deps (annotation + evaluation only — not training)
pip install openai pillow tqdm

# Set secrets (never commit these)
export OPENAI_API_KEY="sk-..."
export QAI_HUB_API_TOKEN="..."
```

---

## Step 1 — Download datasets (local or Kaggle)

```bash
# CIFAKE (Kaggle — fastest way)
kaggle datasets download -d bird/cifake-real-and-ai-generated-synthetic-images
unzip cifake-real-and-ai-generated-synthetic-images.zip -d data/cifake/

# COCO 2017 val (real images)
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip -d data/coco/

# ArtiFact (optional — larger, better generalization)
pip install huggingface_hub
huggingface-cli download awsaf49/artifact --repo-type dataset --local-dir data/artifact/
```

---

## Step 2 — Auto-annotate with GPT-4o (local machine)

```bash
cd scripts/

# Annotate AI-generated images (CIFAKE train/FAKE)
python auto_annotate.py annotate \
    --input_dir ../data/cifake/train/FAKE \
    --label AI-Generated \
    --output ../data/annotations_ai_cifake.json \
    --limit 5000 \
    --rpm 100

# Annotate real images (CIFAKE train/REAL)
python auto_annotate.py annotate \
    --input_dir ../data/cifake/train/REAL \
    --label Real \
    --output ../data/annotations_real_cifake.json \
    --limit 5000 \
    --rpm 100

# Merge + train/val split
python auto_annotate.py merge \
    --inputs ../data/annotations_ai_cifake.json ../data/annotations_real_cifake.json \
    --output ../data/annotations \
    --val_split 0.1

# Output: data/annotations_train.json (9000 samples) + data/annotations_val.json (1000 samples)
```

**Cost estimate:** 10,000 images × $0.002 = ~$20

---

## Step 3 — Build calibration set (local machine)

```bash
python build_calibration_set.py \
    --annotations ../data/annotations_train.json \
    --output_dir ../data/calibration_domain_mixed \
    --n_real 150 \
    --n_ai 150

# Output: data/calibration_domain_mixed/ (300 images)
```

---

## Step 4 — Upload to Kaggle

1. Zip `data/annotations_train.json` + `data/annotations_val.json`
2. Create Kaggle dataset: "track3-annotations"
3. Upload `kaggle/track3_training.py` as a Kaggle notebook script
4. Add Kaggle secrets: `QAI_HUB_API_TOKEN`
5. Set accelerator: GPU T4 x2

---

## Step 5 — Run training on Kaggle

Open `kaggle/track3_training.py` in Kaggle and run all cells.

Expected runtime: ~6-8 hours on T4 x2 for 9000 samples × 3 epochs.

Download output: `/kaggle/working/qwen2vl_merged/` → Drive (it's ~4.5 GB)

---

## Step 6 — Evaluate merged model (local or Colab)

```bash
cd scripts/

python evaluate.py \
    --model_path ../models/qwen2vl_merged \
    --val_json ../data/annotations_val.json \
    --output ../experiments/run_002_eval.json \
    --limit 100

# Check: detection_accuracy should be > 0.80 to proceed
```

---

## Step 7 — AIMET Quantization (QPM tutorial)

1. Download "Tutorial for Qwen2_VL_2b (IoT)" from QPM
2. In `Example1A_veg.ipynb`, change model path:
   ```python
   model_path = "/path/to/qwen2vl_merged"
   calibration_image_dir = "/path/to/data/calibration_domain_mixed"
   ```
3. Run Example 1A → 1B → 2A → 2B in order
4. Collect output artifacts (see submission checklist in README.md)

---

## Step 8 — AIHub simulation

```bash
# From sample solution repo root
python inference_multi.py \
    --model_path ./ar*-ar*-cl*/weight_sharing_model_1_of_1.serialized.bin \
    --device "Snapdragon 8 Gen 5 QRD" \
    --inputs_dir ./baseline_outputs/inputs/ \
    --output_dir ./aihub_outputs/ \
    --api_token $QAI_HUB_API_TOKEN

python compute_score_multi_aihub.py \
    --baseline_dir ./baseline_outputs/ \
    --aihub_dir ./aihub_outputs/ \
    --output_report score_report.json
```

Record: compile job ID, inference job ID, quantization error metrics.

---

## Step 9 — Submit

1. Share compile job with `lowpowervision@gmail.com`
2. Package ZIP (see README.md checklist)
3. Submit at https://lpcv.ai/2026LPCVC/submission/track3
4. Log submission in `scripts/experiment_log.json`

---

---

# Pipeline 2 — ConvNeXt-Tiny Cascade + DoRA + Curriculum

> Run these **in parallel with Pipeline 1** (separate Kaggle session).
> Uses different dataset (2,500 curated), different adapter (DoRA), different inference (single-pass prefill).

## P2 Step 1 — Annotate with Zoom-In strategy (local machine)

```bash
cd Track3/scripts/

# AI images from FakeClue or CIFAKE FAKE (1,250 samples)
python auto_annotate_p2.py annotate \
    --input_dir ../data/cifake/train/FAKE \
    --label AI-Generated \
    --output ../data/annotations_p2_ai.json \
    --limit 1250 \
    --rpm 60

# Real images from COCO val (1,250 samples)
python auto_annotate_p2.py annotate \
    --input_dir ../data/coco/val2017 \
    --label Real \
    --output ../data/annotations_p2_real.json \
    --limit 1250 \
    --rpm 60

# Merge + split (90/10)
python auto_annotate_p2.py merge \
    --inputs ../data/annotations_p2_ai.json ../data/annotations_p2_real.json \
    --output ../data/annotations_p2 \
    --val_split 0.1

# Output: annotations_p2_train.json (2,250) + annotations_p2_val.json (250)
```

**Cost: 2,500 × $0.002 × ~1.5 (two API calls) = ~$7.50**

---

## P2 Step 2 — Upload to Kaggle

1. Zip `annotations_p2_train.json` + `annotations_p2_val.json`
2. Create Kaggle dataset: **track3-p2-annotations**
3. Upload `kaggle/track3_pipeline2_training.py` as notebook script
4. GPU: T4 x2 — expected runtime ~4-5 hours

---

## P2 Step 3 — Run training on Kaggle

Open `kaggle/track3_pipeline2_training.py` and run all 11 cells.

Downloads needed after training:
- `/kaggle/working/qwen2vl_p2_merged/` → Drive (~4.5 GB VLM)
- `/kaggle/working/convnext_detector.pt` → Drive (~100 MB CNN)
- `/kaggle/working/pipeline2_results.json` → check metrics

---

## P2 Step 4 — AIMET W8A8 QAT (QPM tutorial)

> P2 targets W8A8 (not W4A8) — near-lossless quantization for reasoning tasks.

1. Same QPM tutorial as Pipeline 1: "Tutorial for Qwen2_VL_2b (IoT)"
2. In `Example1A_veg.ipynb`, change model path to `qwen2vl_p2_merged`
3. Set `weight_bw=8, act_bw=8` (symmetric) — this is the key change
4. Calibrate with `data/calibration_domain_mixed/` (same 300-image set)
5. Run QAT for 3-5 epochs at LR=1e-6 (AIMET QuantizationSimModel)
6. Export QNN binary

---

## Current status

### Pipeline 1 (LoRA, 9,000 samples)

| Step | Status |
|------|--------|
| Dataset download | pending |
| Auto-annotation (10k) | pending |
| Calibration set | pending |
| Kaggle training | pending |
| Evaluation | pending |
| AIMET W4A8 PTQ | pending |
| AIHub simulation | pending |
| Submission | pending |

### Pipeline 2 (DoRA cascade, 2,500 samples)

| Step | Status |
|------|--------|
| Zoom-In annotation (2.5k) | pending |
| Upload to Kaggle | pending |
| Kaggle training (cascade) | pending |
| P2 Evaluation | pending |
| AIMET W8A8 QAT | pending |
| AIHub simulation (P2) | pending |
| Submission (best of P1/P2) | pending |
