# Pipeline 1 — R(2+1)D-18 · 88.94% val acc · 26.85 ms latency

## What this is

R(2+1)D-18 fine-tuned on QEVD (91 classes), 11 epochs.
Standard NCDHW input layout, FP16 TFLite compiled on AI Hub.

**Verified results from previous run:**

| Metric | Value |
|---|---|
| Val accuracy | 88.94% |
| Compile job | `jgj0wmy8p` |
| Profile job | `j5mwd2lqp` |
| Latency (best) | **26.85 ms** |
| Latency (mean) | 27.88 ms |
| Peak memory | 8.61 MB |
| Status | VALID (7.15 ms margin) |

---

## Files

| File | Purpose |
|---|---|
| `kaggle_training.py` | Train R(2+1)D-18 on Kaggle GPU T4 x2 |
| `aihub_deploy.py` | Compile + profile + infer on AI Hub |

---

## Step 1 — Skip training (model already done)

The 88% model is already trained and on Drive. Download it:

```bash
pip install gdown

# checkpoint
gdown "https://drive.google.com/uc?id=1m9aK8JgjFa6ewDhLpIb5rAepUybs5veU" -O best_r2plus1d_qevd.pth

# ONNX (self-contained, 119 MB)
gdown "https://drive.google.com/uc?id=1tEObF3rGGO69y7DvEM3xeieUcuwLs6WH" -O lpcvc_final_unified.onnx
```

Go straight to Step 3.

---

## Step 2 — (Optional) Retrain on Kaggle

Only needed if you want to train from scratch or improve further.

### 2a. Prepare Kaggle notebook

1. Go to https://kaggle.com → New Notebook → "Import from file"
2. Upload `kaggle_training.py`
3. Settings → Accelerator → **GPU T4 x2**
4. Settings → Internet → **On**
5. Add input dataset: your preprocessed QEVD tensors
   - Dataset path format: `manifest.jsonl` + `class_labels.json` + `class_map.json`
   - Update `TENSORS_ROOT` in Cell 2 to match your dataset slug

### 2b. Add Kaggle secret

1. Notebook → Add-ons → Secrets
2. Add secret: Name = `QAI_HUB_API_TOKEN`, Value = `wyfb5d78py0bgow6gz10krdqyd8qq0l9wlfw9swc`

### 2c. Run

Click **Run All**. Expected time: ~2-3 hours on T4 x2 for 11 epochs.

### 2d. Download outputs

After training completes, download from `/kaggle/working/`:
- `best_r2plus1d_qevd.pth`
- `lpcvc_final_unified.onnx`

---

## Step 3 — Compile + Profile on AI Hub

### 3a. Install

```bash
pip install qai-hub onnx
```

### 3b. Set token

```powershell
# Windows PowerShell
$env:QAI_HUB_API_TOKEN = "wyfb5d78py0bgow6gz10krdqyd8qq0l9wlfw9swc"
```

```bash
# Linux / Mac
export QAI_HUB_API_TOKEN="wyfb5d78py0bgow6gz10krdqyd8qq0l9wlfw9swc"
```

Or configure permanently:

```bash
qai-hub configure --api_token wyfb5d78py0bgow6gz10krdqyd8qq0l9wlfw9swc
```

### 3c. Run deploy script

```bash
python aihub_deploy.py --onnx lpcvc_final_unified.onnx --output_dir aihub_results_p1
```

This will:
1. Upload ONNX to AI Hub
2. Compile to TFLite FP16 on `Dragonwing IQ-9075 EVK`
3. Profile latency (100 runs)
4. Run inference with dummy input
5. Save compile + profile job IDs to `aihub_results_p1/experiment_log_p1.json`

Expected output:
```
Inference time (best): 26.85 ms
Inference time (mean): 27.88 ms
Peak memory:           8.61 MB
VALID — margin: 7.15 ms
```

### 3d. Skip inference test (faster)

```bash
python aihub_deploy.py --onnx lpcvc_final_unified.onnx --skip_infer
```

---

## Step 4 — Share compile job

1. Open https://workbench.aihub.qualcomm.com/jobs/YOUR_COMPILE_JOB_ID/
2. Click **Share** → enter `lowpowervision@gmail.com`
3. Confirm share

---

## Step 5 — Submit

1. Open https://lpcv.ai/2026LPCVC/submission/track2
2. Fill in:
   - Team info
   - Compile job ID (from `aihub_results_p1/experiment_log_p1.json`)
3. Submit

---

## Compile options reference

| Option | Value | Why |
|---|---|---|
| `--target_runtime` | `tflite` | TFLite runtime on IQ-9075 EVK |
| Input shape | `(1, 3, 16, 112, 112)` | NCDHW standard layout |
| Quantization | None (FP16) | Already passes 34ms gate |

To compile manually without the script:

```python
import qai_hub as hub, os
os.environ["QAI_HUB_TOKEN"] = "wyfb5d78py0bgow6gz10krdqyd8qq0l9wlfw9swc"
client = hub.Client()
device = client.get_devices(name="Dragonwing IQ-9075 EVK")[0]
model  = client.upload_model("lpcvc_final_unified.onnx")
job    = client.submit_compile_job(
    model=model, device=device,
    options="--target_runtime tflite",
    input_specs={"input": (1, 3, 16, 112, 112)},
)
job.wait()
print(job.job_id)
```
