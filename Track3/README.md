# Track 3 — AI Generated Image Detection
### LPCVC 2026 · Qualcomm Snapdragon 8 Gen 5 · Qwen2-VL-2B-Instruct

> **Submission window:** March 1 – April 30, 2026 (ET)  
> **Hardware target:** Snapdragon 8 Gen 5 QRD  
> **Official page:** https://lpcv.ai/2026LPCVC/tracks/track3/  
> **Sample solution:** https://github.com/lpcvai/26LPCVC_Track3_Sample_Solution

---

## Table of Contents

1. [Task Overview](#1-task-overview)  
2. [Scoring Formula](#2-scoring-formula)  
3. [Environment Setup](#3-environment-setup)  
4. [Step 1 — Dataset Collection](#step-1--dataset-collection)  
5. [Step 2 — Calibration Data Preparation](#step-2--calibration-data-preparation)  
6. [Step 3 — Model Quantization with AIMET (Kaggle GPU)](#step-3--model-quantization-with-aimet-kaggle-gpu)  
7. [Step 4 — ONNX Conversion to Qualcomm NN](#step-4--onnx-conversion-to-qualcomm-nn)  
8. [Step 5 — Build inputs.json](#step-5--build-inputsjson)  
9. [Step 6 — Quantization Error Check via AIHub Simulation](#step-6--quantization-error-check-via-aihub-simulation)  
10. [Step 7 — Local On-Device Inference (optional, with hardware)](#step-7--local-on-device-inference-optional-with-hardware)  
11. [Step 8 — Build Submission Package](#step-8--build-submission-package)  
12. [Step 9 — Final Submission to LPCV](#step-9--final-submission-to-lpcv)  
13. [API Key Setup](#api-key-setup)  
14. [Submission Package Checklist](#submission-package-checklist)  
15. [FAQ / Gotchas](#faq--gotchas)

---

## 1. Task Overview

Track 3 asks you to detect whether an image is **AI-generated or real** and provide a **structured explanation** across **8 perceptual criteria**.

| Aspect | Requirement |
|--------|-------------|
| Model | Any VLM — organizers use **Qwen2-VL-2B-Instruct** as baseline |
| Reasoning | Two-step prompt pipeline (detect → explain) |
| Output format | Structured JSON: per-criterion evidence + overall likelihood score |
| Execution platform | Snapdragon 8 Gen 5 via Qualcomm AI Hub / Genie runtime |
| Validity gate | Stage 1: model must execute within time limit |
| Quality scoring | Stage 2: detection accuracy + explanation quality |

The two-step prompt pipeline works as:
- **Prompt 1** → ask the VLM to reason over the image using 8 criteria
- **Prompt 2** → ask the VLM to emit a structured JSON summary from its own reasoning

The 8 criteria include (but are not limited to): texture coherence, lighting consistency, edge artifacts, facial geometry, background plausibility, object proportionality, noise patterns, and metadata cues.

---

## 2. Scoring Formula

Scoring is two-stage:

**Stage 1 — Validity Gate**  
The model must complete inference within the allowed execution time on Snapdragon 8 Gen 5. Jobs that exceed this are disqualified.

**Stage 2 — Combined Score**

```
Final Score = α × Detection_Score + β × Explanation_Score
```

Where:
- `Detection_Score` = binary/soft classification accuracy (real vs. AI-generated)
- `Explanation_Score` = quality of per-criterion evidence in the JSON output
- `α` and `β` depend on the data category (the organizers define category weights)

Higher is better. Always use the official `compute_score_multi_aihub.py` from the sample solution repo to replicate organizer scoring locally.

---

## 3. Environment Setup

### 3.1 AI Hub API Key (do this first — never commit to Git)

**PowerShell (Windows):**
```powershell
$env:QAI_HUB_API_TOKEN = "your_actual_api_key_here"
$env:QUALCOMM_WORKBENCH_API_KEY = $env:QAI_HUB_API_TOKEN
```

**Bash / Linux / Kaggle secrets:**
```bash
export QAI_HUB_API_TOKEN="your_actual_api_key_here"
```

**Python usage pattern (always use this — never hardcode the key):**
```python
import os

api_token = os.getenv("QAI_HUB_API_TOKEN") or os.getenv("QUALCOMM_WORKBENCH_API_KEY")
if not api_token:
    raise RuntimeError("Missing AI Hub API token. Set QAI_HUB_API_TOKEN in your environment.")
```

Get your API key from: https://app.aihub.qualcomm.com/ → Account → API Keys

### 3.2 Python dependencies

```bash
pip install qai-hub transformers torch torchvision pillow accelerate
pip install gdown  # for Drive downloads
```

### 3.3 QPM Tutorial Download

Download the **"Tutorial for Qwen2_VL_2b (IoT)"** package from:  
https://qpm.qualcomm.com/  
(Search for "Qwen2 VL 2b IoT tutorial" — requires a free Qualcomm developer account)

This gives you the `qnn_model_execution.ipynb` and example notebooks referenced in Steps 3–5.

---

## Step 1 — Dataset Collection

Track 3 evaluation uses the organizer's private test dataset. You do **not** submit the dataset — only the model.

However, you will need:

### 1a. For training / fine-tuning (optional but recommended)

Collect a balanced dataset of real vs. AI-generated images:

| Source | Type | How to get |
|--------|------|-----------|
| COCO 2017 | Real images | `wget http://images.cocodataset.org/zips/val2017.zip` |
| LAION-400M subset | Real images | HuggingFace: `laion/laion400m` |
| Stable Diffusion XL | AI-generated | Generate locally or use existing datasets |
| Midjourney / DALL-E | AI-generated | Use public datasets on Kaggle |
| ArtiFact dataset | Labeled real+AI | https://github.com/awsaf49/artifact |
| CIFAKE | Labeled real+AI | Kaggle: `bird/cifake-real-and-ai-generated-synthetic-images` |

Recommended class balance: **50% real, 50% AI-generated**, minimum 1000 images per class.

### 1b. For AIMET calibration (required for quantization)

Download COCO 2017 validation images (used by the QPM tutorial):
```bash
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip -d ./calibration_data/coco/
```

Download the LLaVA calibration JSON:
```bash
# From HuggingFace Hub
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='liuhaotian/LLaVA-Instruct-150K',
    filename='llava_v1_5_mix665k_300.json',
    local_dir='./calibration_data/'
)"
```

---

## Step 2 — Calibration Data Preparation

The AIMET quantization requires two calibration files:

| File | Purpose | Location |
|------|---------|---------|
| `llava_v1_5_mix665k_300.json` | Text calibration data | `./calibration_data/` |
| COCO 2017 val images | Image calibration data | `./calibration_data/coco/val2017/` |
| `wiki103_test_long.json` | PPL evaluation | Qualcomm support forum (see note) |

> **Note on `wiki103_test_long.json`**: This file format details are available on the Qualcomm AI Hub support forum. Check the QPM tutorial's documentation folder after download.

Verify your calibration setup:
```python
import json
from pathlib import Path

with open("calibration_data/llava_v1_5_mix665k_300.json") as f:
    data = json.load(f)
print(f"Calibration samples: {len(data)}")

coco_path = Path("calibration_data/coco/val2017")
print(f"COCO images: {len(list(coco_path.glob('*.jpg')))}")
```

---

## Step 3 — Model Quantization with AIMET (Kaggle GPU)

> Use Kaggle GPU (T4 x2 or P100) for this step. AIMET needs ~16 GB VRAM for Qwen2-VL-2B.

### 3.1 Set up Kaggle Notebook

1. Create a new Kaggle notebook
2. Enable GPU: Settings → Accelerator → **GPU T4 x2**
3. Add your AI Hub key as a Kaggle secret: Settings → Secrets → `QAI_HUB_API_TOKEN`

### 3.2 Pull the AIMET Docker image (for local use) or install in Kaggle

**For Kaggle (pip-based install):**
```python
# In Kaggle notebook cell
!pip install aimet-torch==1.34.0 --quiet
```

**For local Docker (Linux machine — corrected tag from QPM tutorial):**
```bash
# Note: the QPM tutorial has a typo — use this exact tag
docker pull artifacts.codelinaro.org/codelinaro-aimet/aimet-dev:1.34.0.torch-gpu-pt113

# Run the container
docker run -it --gpus all \
  -v $(pwd):/workspace \
  artifacts.codelinaro.org/codelinaro-aimet/aimet-dev:1.34.0.torch-gpu-pt113 \
  bash

# Inside container — navigate to correct build directory
cd aimetpro-release-1.34.0_build-*.torch-gpu-pt113-release
```

### 3.3 Run Example 1A — PyTorch Quantization

Inside the QPM tutorial, open `Example1A_veg.ipynb` and run all cells.

This generates:
- `mask.raw` — attention mask  
- `position_ids_cos.raw` — cosine positional encodings  
- `position_ids_sin.raw` — sine positional encodings  
- `tokenizer.json` — vocabulary  
- `embedding_weights*.raw` — embedding tensor

> **Tip:** The "Other lib" dependency cells in `Example1A_veg.ipynb` can be commented out safely — they are not required for the quantization flow.

### 3.4 Run Example 1B — VEG Model Export

Open `Example1B_veg.ipynb` and run all cells.

This generates:
- `veg.serialized.bin` — the VEG (visual encoding graph) model binary

### 3.5 Save outputs from Kaggle

```python
# In Kaggle — save all generated files to output
import shutil, os

output_files = [
    "mask.raw", "position_ids_cos.raw", "position_ids_sin.raw",
    "tokenizer.json", "veg.serialized.bin"
]
for f in output_files:
    shutil.copy(f, f"/kaggle/working/{f}")
    print(f"Saved: {f}")
```

Download them from the Kaggle notebook output panel.

---

## Step 4 — ONNX Conversion to Qualcomm NN

### 4.1 QNN SDK Environment Setup

After AIMET, configure QNN paths (set `QNN_SDK_ROOT` to your extracted SDK):

```bash
export QNN_SDK_ROOT=/path/to/qnn_sdk

export PYTHONPATH=$QNN_SDK_ROOT/lib/python:$PYTHONPATH
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH
```

**On Kaggle**, install via the qai-hub SDK instead:
```python
!pip install qai-hub qai_hub_models --quiet
import qai_hub
qai_hub.configure(api_token=os.environ["QAI_HUB_API_TOKEN"])
```

### 4.2 Run Example 2A — ONNX Export

Open `Example2A_onnx.ipynb` in the QPM tutorial and run all cells.

Outputs:
- `qwen2vl_model.onnx` (large — ~2 GB, store in Drive not Git)

### 4.3 Run Example 2B — Compile to QNN Binary

Open `Example2B_compile.ipynb` in the QPM tutorial and run all cells.

This generates:
- `ar*-ar*-cl*/weight_sharing_model_1_of_1.serialized.bin` — the quantized QNN model

> **Alternative for Qwen2.5-VL models:** replace `mask.raw` with:
> - `full_attention_mask.raw`
> - `window_attention_mask.raw`

### 4.4 Compile via AI Hub Python API (alternative to local compile)

```python
import qai_hub
import os

# Upload the ONNX model to AI Hub
model = qai_hub.upload_model("qwen2vl_model.onnx")

# Compile for Snapdragon 8 Gen 5
compile_job = qai_hub.submit_compile_job(
    model=model,
    device=qai_hub.Device("Snapdragon 8 Gen 5 QRD"),
    options="--target_runtime qnn_context_binary",
)

print(f"Compile job ID: {compile_job.job_id}")
compile_job.wait()

# Download compiled artifact
compiled_model = compile_job.get_target_model()
compiled_model.download("weight_sharing_model_1_of_1.serialized.bin")
```

Record your compile job ID — you will need it for submission.

---

## Step 5 — Build inputs.json

Extract parameters from `qnn_model_execution.ipynb` (in QPM tutorial) and build `inputs.json`:

```json
{
  "qwen_vl_processor": "Qwen/Qwen2-VL-2B-Instruct",
  "llm_config": "Qwen/Qwen2-VL-2B-Instruct",
  "data_preprocess_inp_h": 342,
  "data_preprocess_inp_w": 512,
  "run_veg_n_tokens": 216,
  "run_veg_embedding_dim": 1536,
  "genie_config": {
    "version": "0.0.1",
    "models": [
      {
        "name": "qwen2_vl",
        "lib": "libQnnHtp.so",
        "model_prepare": {
          "model_name": "qwen2_vl",
          "share_weights": true,
          "num_cores": 4
        },
        "model_parameters": {
          "context_length": 1024,
          "max_new_tokens": 512,
          "temperature": 0.1,
          "top_p": 0.9
        }
      }
    ]
  }
}
```

> **Important:** In `genie_config`, all boolean values must be **lowercase** (`true`/`false`), not Python-style (`True`/`False`). JSON is case-sensitive.

Validate your JSON:
```bash
python -c "import json; json.load(open('inputs.json')); print('inputs.json is valid')"
```

---

## Step 6 — Quantization Error Check via AIHub Simulation

Use this to validate your model without a physical Snapdragon device.

### 6.1 Generate baseline inputs/outputs

```bash
python llm_inout.py \
  --inputs_json inputs.json \
  --output_dir ./baseline_outputs/
```

This runs the full-precision model on CPU/GPU and saves reference I/O tensors.

### 6.2 Submit inference job to AIHub

```bash
python inference_multi.py \
  --model_path ./ar*-ar*-cl*/weight_sharing_model_1_of_1.serialized.bin \
  --device "Snapdragon 8 Gen 5 QRD" \
  --inputs_dir ./baseline_outputs/inputs/ \
  --output_dir ./aihub_outputs/ \
  --api_token $QAI_HUB_API_TOKEN
```

Or using the Python API directly:
```python
import qai_hub
import os

qai_hub.configure(api_token=os.environ["QAI_HUB_API_TOKEN"])

# Upload your compiled model
compiled_model = qai_hub.get_model("your_compiled_model_id")

# Run inference on AIHub simulation
inference_job = qai_hub.submit_inference_job(
    model=compiled_model,
    device=qai_hub.Device("Snapdragon 8 Gen 5 QRD"),
    inputs={"input_ids": your_input_tensors},
)

print(f"Inference job ID: {inference_job.job_id}")
inference_job.wait()
outputs = inference_job.download_output_data()
```

### 6.3 Compute quantization error score

```bash
python compute_score_multi_aihub.py \
  --baseline_dir ./baseline_outputs/ \
  --aihub_dir ./aihub_outputs/ \
  --output_report score_report.json
```

Check `score_report.json` — aim for low quantization error before packaging the submission.

---

## Step 7 — Local On-Device Inference (optional, with hardware)

If you have access to a Snapdragon 8 Gen 5 device (QRD):

```bash
# Place submission files in contestant_uploads/
mkdir -p contestant_uploads/your_team_name/
cp -r ar*-ar*-cl*/ contestant_uploads/your_team_name/
cp serialized_binaries/veg.serialized.bin contestant_uploads/your_team_name/serialized_binaries/
cp embedding_weights*.raw mask.raw position_ids_cos.raw position_ids_sin.raw \
   tokenizer.json inputs.json contestant_uploads/your_team_name/

# Run on-device inference
python inference_script.py \
  --submission_dir ./contestant_uploads/your_team_name/ \
  --output_dir ./Host_Outputs/
```

Outputs appear in `./Host_Outputs/`.

---

## Step 8 — Build Submission Package

The submission ZIP must have **exactly this structure**:

```
your_team_name.zip
└── your_team_name/
    ├── ar*-ar*-cl*/
    │   └── weight_sharing_model_1_of_1.serialized.bin
    ├── serialized_binaries/
    │   └── veg.serialized.bin
    ├── embedding_weights*.raw
    ├── inputs.json
    ├── mask.raw                    ← Qwen2-VL only
    ├── position_ids_cos.raw
    ├── position_ids_sin.raw
    └── tokenizer.json
```

> For **Qwen2.5-VL** models: replace `mask.raw` with `full_attention_mask.raw` + `window_attention_mask.raw`.

Build the ZIP:
```bash
# Linux / Kaggle
zip -r your_team_name.zip your_team_name/

# Windows PowerShell
Compress-Archive -Path "your_team_name\" -DestinationPath "your_team_name.zip"
```

Verify the ZIP size — expect ~1.8 GB compressed, ~2.4 GB uncompressed.

---

## Step 9 — Final Submission to LPCV

### 9.1 Share model permission with organizers

Before submitting the form, share your compiled model/job with the organizers:

```python
import qai_hub

compiled_model = qai_hub.get_model("your_compiled_model_id")
compiled_model.modify_sharing(
    add_emails=["lowpowervision@gmail.com"]
)
print("Sharing done.")
```

Or share the compile job:
```python
compile_job = qai_hub.get_job("your_compile_job_id")
# Share from AI Hub web UI: https://app.aihub.qualcomm.com/jobs/
```

### 9.2 Upload submission ZIP

Go to: https://lpcv.ai/2026LPCVC/submission/track3

Fill in:
- Team name
- Compile job ID (from Step 4)
- Any additional metadata requested
- Upload `your_team_name.zip`

### 9.3 Record your submission

Keep a log entry with:
```
Date: YYYY-MM-DD
Team: your_team_name
Compile Job ID: <id>
Inference Job ID: <id>  (if run)
Score (local estimate): <value>
Submission confirmation: <screenshot or form ID>
```

---

## API Key Setup

| Platform | How to configure |
|----------|-----------------|
| **Local (Windows PowerShell)** | `$env:QAI_HUB_API_TOKEN = "key"` |
| **Local (Linux/Mac bash)** | `export QAI_HUB_API_TOKEN="key"` |
| **Kaggle** | Settings → Add-ons → Secrets → add `QAI_HUB_API_TOKEN` |
| **Google Colab** | `from google.colab import userdata; key = userdata.get("QAI_HUB_API_TOKEN")` |

**Python universal pattern:**
```python
import os
import qai_hub

token = os.getenv("QAI_HUB_API_TOKEN") or os.getenv("QUALCOMM_WORKBENCH_API_KEY")
if not token:
    raise RuntimeError("Set QAI_HUB_API_TOKEN env variable")
qai_hub.configure(api_token=token)
```

Get your key at: https://app.aihub.qualcomm.com/ → top-right profile → API Token

**Never commit your API key to Git.** The `.gitignore` already covers `.env` files — keep keys there.

---

## Submission Package Checklist

Before uploading, verify every item:

- [ ] `ar*-ar*-cl*/weight_sharing_model_1_of_1.serialized.bin` exists
- [ ] `serialized_binaries/veg.serialized.bin` exists
- [ ] `embedding_weights*.raw` exists
- [ ] `inputs.json` is valid JSON (no Python booleans — lowercase `true`/`false`)
- [ ] `mask.raw` (or `full_attention_mask.raw` + `window_attention_mask.raw` for Qwen2.5)
- [ ] `position_ids_cos.raw` exists
- [ ] `position_ids_sin.raw` exists
- [ ] `tokenizer.json` exists
- [ ] Folder name inside ZIP matches team name exactly
- [ ] ZIP file named `your_team_name.zip`
- [ ] Model/compile job shared with `lowpowervision@gmail.com`
- [ ] Submission form filled at https://lpcv.ai/2026LPCVC/submission/track3
- [ ] Compile job ID recorded in team log
- [ ] Form confirmation saved

---

## FAQ / Gotchas

**Q: The QPM Docker tag in the tutorial doesn't work.**  
A: Use the corrected tag:  
`artifacts.codelinaro.org/codelinaro-aimet/aimet-dev:1.34.0.torch-gpu-pt113`  
Navigate to `aimetpro-release-1.34.0_build-*.torch-gpu-pt113-release` inside the container.

**Q: `inputs.json` keeps failing validation on AIHub.**  
A: Check that all booleans are lowercase (`true`/`false`). Python's `json.dumps()` handles this correctly — do not write JSON by hand.

**Q: "Other lib" dependencies in Example1A fail to install.**  
A: Comment those cells out — they are not required for the Track 3 submission pipeline.

**Q: I don't have a Snapdragon 8 Gen 5 device.**  
A: Use AIHub simulation: Steps 6.1–6.3 cover the full no-device workflow using `inference_multi.py` + `compute_score_multi_aihub.py`.

**Q: Where do I store ONNX and model binaries?**  
A: Google Drive only. Never commit files >50 MB to this repository. Use `gdown` to pull them:
```bash
gdown "https://drive.google.com/uc?id=FILE_ID" -O Track3/models/model.onnx
```

**Q: How large is the full solution?**  
A: ~1.8 GB compressed, ~2.4 GB uncompressed. Download from the organizer-provided SharePoint link in the QPM tutorial package.

**Q: The submission deadline passed?**  
A: Submission window is March 1 – April 30, 2026 (ET). Check the official page for any extensions.

---

## Resources

| Resource | Link |
|----------|------|
| Official Track 3 page | https://lpcv.ai/2026LPCVC/tracks/track3/ |
| Sample solution repo | https://github.com/lpcvai/26LPCVC_Track3_Sample_Solution |
| Submission portal | https://lpcv.ai/2026LPCVC/submission/track3 |
| Leaderboard | https://lpcv.ai/2026LPCVC/leaderboard/track3 |
| Qualcomm AI Hub | https://app.aihub.qualcomm.com/ |
| AI Hub Workbench docs | https://workbench.aihub.qualcomm.com/docs/ |
| QPM (model tutorial) | https://qpm.qualcomm.com/ |
| CIFAKE dataset | https://www.kaggle.com/datasets/bird/cifake-real-and-ai-generated-synthetic-images |
| ArtiFact dataset | https://github.com/awsaf49/artifact |
| COCO 2017 validation | http://images.cocodataset.org/zips/val2017.zip |
| LLaVA calibration data | https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K |
| Team usage guide | ../competition-conext/team-usage-guide.md |
| Workbench notes | ../competition-conext/workbench-docs.md |
