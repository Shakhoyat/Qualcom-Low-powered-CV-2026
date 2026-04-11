# Pipeline 2 — R(2+1)D-18 · ~89% val acc · NDHWC + INT8 QNN

## What's different from Pipeline 1

| | Pipeline 1 | Pipeline 2 (this) |
|---|---|---|
| Input layout | NCDHW `(1,3,16,112,112)` | **NDHWC `(1,16,112,112,3)`** — Hexagon native |
| Normalization | CPU-side (before model) | **Baked into ONNX graph** (zero CPU overhead) |
| Compile target | `tflite` | **`qnn_context_binary`** |
| Quantization | FP16 (no calibration) | **INT8** with 100 calibration samples |
| Training | 11 epochs, fc+layer4 only | **15 epochs, progressive 3-phase unfreezing** |
| Expected latency | ~27 ms | **Target <20 ms (INT8 gain)** |

---

## Files

| File | Purpose |
|---|---|
| `kaggle_training.py` | Train + export NDHWC ONNX + generate calibration data |
| `aihub_deploy.py` | FP16 + INT8 compile, profile, infer, log |

---

## Step 1 — Skip training (pre-trained model on Drive)

```bash
pip install gdown

# NDHWC ONNX with baked normalization (~120 MB)
gdown "https://drive.google.com/uc?id=13QB9_7sFMzYg_AoGcqfbHCG9BthXBw0I" -O qualcomm_r2plus1d.onnx

# INT8 calibration data (100 samples, NDHWC)
gdown "https://drive.google.com/uc?id=12N6WmkBA2vg1rpoSe7PkaPOK7jy22Ny7" -O calibration_inputs.npy

# Checkpoint (optional — only needed if you want to re-export)
gdown "https://drive.google.com/uc?id=1txn4uzy8rdl-XtK6P1KTQkry61diOn1b" -O best_r2plus1d_qevd.pth
```

Go straight to Step 3.

---

## Step 2 — (Optional) Retrain on Kaggle

### 2a. Create notebook on Kaggle

1. https://kaggle.com → New Notebook → Import file → upload `kaggle_training.py`
2. Settings → Accelerator → **GPU T4 x2**
3. Settings → Internet → **On**
4. Add-ons → Datasets → Add your preprocessed QEVD dataset
   - Update `TENSORS_ROOT` in Cell 2 to your dataset slug, e.g.:
     ```
     /kaggle/input/YOUR-DATASET-SLUG/preprocessed_tensors
     ```

### 2b. Add Kaggle secret

Add-ons → Secrets → Add:
- **Name:** `QAI_HUB_API_TOKEN`
- **Value:** `wyfb5d78py0bgow6gz10krdqyd8qq0l9wlfw9swc`

### 2c. Run all cells

Click **Run All**. Expected time: ~4-5 hours on T4 x2 (15 epochs).

Training schedule:
- Epochs 1–5: FC + layer4 only (LR = 3e-4)
- Epochs 6–10: + layer3 (LR = 9e-5)
- Epochs 11–15: + layer2 (LR = 3e-5)

### 2d. Download from Kaggle

After run completes, download from `/kaggle/working/`:
- `best_r2plus1d_qevd.pth`
- `qualcomm_r2plus1d.onnx` ← NDHWC wrapper + baked normalization
- `calibration_data/calibration_inputs.npy` ← 100 samples for INT8

---

## Step 3 — Compile + Profile on AI Hub

### 3a. Install

```bash
pip install qai-hub onnx numpy
```

### 3b. Set token

```powershell
# Windows PowerShell
$env:QAI_HUB_API_TOKEN = "wyfb5d78py0bgow6gz10krdqyd8qq0l9wlfw9swc"
```

```bash
# Linux / Mac / WSL
export QAI_HUB_API_TOKEN="wyfb5d78py0bgow6gz10krdqyd8qq0l9wlfw9swc"
```

Or configure permanently:

```bash
qai-hub configure --api_token wyfb5d78py0bgow6gz10krdqyd8qq0l9wlfw9swc
```

### 3c. Run full deploy (FP16 baseline + INT8 target)

```bash
python aihub_deploy.py \
    --onnx  qualcomm_r2plus1d.onnx \
    --calib calibration_inputs.npy \
    --output_dir aihub_results_p2
```

This runs:
1. **FP16 compile** → `qnn_context_binary` → profile latency
2. **INT8 compile** with calibration → `qnn_context_binary` → profile latency
3. On-device inference test for both
4. Saves `aihub_results_p2/experiment_log_p2.json` with all job IDs

### 3d. FP16 only (skip INT8 — faster)

```bash
python aihub_deploy.py --onnx qualcomm_r2plus1d.onnx --fp16_only
```

### 3e. Skip inference test

```bash
python aihub_deploy.py --onnx qualcomm_r2plus1d.onnx --calib calibration_inputs.npy --skip_infer
```

---

## Step 4 — Manual compile (no script)

```python
import qai_hub as hub, os, numpy as np

os.environ["QAI_HUB_TOKEN"] = "wyfb5d78py0bgow6gz10krdqyd8qq0l9wlfw9swc"
client = hub.Client()
device = client.get_devices(name="Dragonwing IQ-9075 EVK")[0]

# Upload model
model = client.upload_model("qualcomm_r2plus1d.onnx")

# Load calibration data
calib_data = np.load("calibration_inputs.npy").astype("float32")
calib = {"input": [calib_data[i:i+1] for i in range(len(calib_data))]}

# INT8 compile
job = client.submit_compile_job(
    model=model,
    device=device,
    options="--target_runtime qnn_context_binary --quantize_full_type int8 --quantize_io",
    input_specs={"input": (1, 16, 112, 112, 3)},   # NDHWC
    calibration_data=calib,
)
job.wait()
print("Compile job ID:", job.job_id)
target = job.get_target_model()

# Profile
profile_job = client.submit_profile_job(model=target, device=device)
profile_job.wait()
profile_job.download_results(artifacts_dir=".")
```

---

## Step 5 — Share compile job

1. Open: `https://workbench.aihub.qualcomm.com/jobs/YOUR_INT8_COMPILE_JOB_ID/`
2. Click **Share** → enter `lowpowervision@gmail.com`
3. Confirm

The compile job ID is saved in `aihub_results_p2/experiment_log_p2.json` under `submit_job_id`.

---

## Step 6 — Submit

1. Open https://lpcv.ai/2026LPCVC/submission/track2
2. Fill in compile job ID from `experiment_log_p2.json`
3. Submit

---

## Compile options reference

| Option | Value | Why |
|---|---|---|
| `--target_runtime` | `qnn_context_binary` | QNN native binary — faster than TFLite on QCS9100 |
| `--quantize_full_type` | `int8` | INT8 weights + activations — 2x+ speedup |
| `--quantize_io` | (flag) | Quantize I/O tensors too |
| Input shape | `(1, 16, 112, 112, 3)` | NDHWC — Hexagon NPU native layout |
| Calibration | 100 samples NDHWC float32 | Required for INT8 static quantization |

## Why NDHWC + baked normalization wins

The Qualcomm Hexagon NPU processes data in NDHWC order natively.
With NCDHW input (Pipeline 1), the compiler inserts an implicit transpose at the boundary — wasting cycles.
With NDHWC, data flows directly into the NPU convolution kernels without reordering.

Baking normalization (`(x - mean) / std`) into the ONNX graph via `constant_folding=True` moves
those 2 operations from the CPU into the NPU graph where they execute for free during the first layer.
