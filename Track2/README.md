# Track 2 — Video Action Recognition (QEVD / Dragonwing IQ-9075 EVK)

## Two Pipelines

| | Pipeline 1 — 88% | Pipeline 2 — 89% |
|---|---|---|
| **Val accuracy** | 88.94% | ~89% |
| **Latency (best)** | 26.85 ms | TBD (INT8 target <20ms) |
| **Input layout** | NCDHW `(1,3,16,112,112)` | NDHWC `(1,16,112,112,3)` |
| **Normalization** | CPU-side | Baked into ONNX graph |
| **Compile target** | `tflite` | `qnn_context_binary` |
| **Quantization** | FP16 | INT8 (with calibration data) |
| **Training epochs** | 11 | 15 (progressive unfreezing) |
| **Dropout** | 0.3 | — |
| **Status** | Submitted, verified <34ms | Training ready |

## AI Hub Token

```
wyfb5d78py0bgow6gz10krdqyd8qq0l9wlfw9swc
```

Store as Kaggle secret `QAI_HUB_API_TOKEN`. Never commit directly.

## Device

`Dragonwing IQ-9075 EVK` — Latency gate: **< 34 ms**

## Quick start

```
Track2/
├── Pipeline1-88pct/
│   ├── README.md           ← How to run P1 on Kaggle + AI Hub
│   ├── kaggle_training.py  ← Training script (Kaggle T4 x2)
│   └── aihub_deploy.py     ← Compile + profile + infer (local or Kaggle)
└── Pipeline2-89pct/
    ├── README.md           ← How to run P2 on Kaggle + AI Hub
    ├── kaggle_training.py  ← Training script (NDHWC + baked norm)
    └── aihub_deploy.py     ← INT8 compile + profile + infer
```

## Drive artifacts

### Pipeline 1 (88%)
- `best_r2plus1d_qevd.pth` — https://drive.google.com/file/d/1m9aK8JgjFa6ewDhLpIb5rAepUybs5veU/view
- `lpcvc_final_unified.onnx` — https://drive.google.com/file/d/1tEObF3rGGO69y7DvEM3xeieUcuwLs6WH/view
- `qualcomm_r2plus1d.onnx` — https://drive.google.com/file/d/1phVY0DqCkBqSfTdZcvfVa8Kc3nE9zi0s/view

### Pipeline 2 (89%)
- `best_r2plus1d_qevd.pth` — https://drive.google.com/file/d/1txn4uzy8rdl-XtK6P1KTQkry61diOn1b/view
- `qualcomm_r2plus1d.onnx` — https://drive.google.com/file/d/13QB9_7sFMzYg_AoGcqfbHCG9BthXBw0I/view
- `calibration_inputs.npy` — https://drive.google.com/file/d/12N6WmkBA2vg1rpoSe7PkaPOK7jy22Ny7/view

## Submission checklist

1. Compile job submitted on AI Hub
2. Compile job shared with `lowpowervision@gmail.com`
3. Latency verified < 34 ms
4. Official form submitted at https://lpcv.ai/2026LPCVC/submission/track2 with compile job ID
