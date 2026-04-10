# Track 1 Sample Solution Notes

Source: https://github.com/lpcvai/26LPCVC_Track1_Sample_Solution

## Purpose
- Reference pipeline for image-text retrieval using CLIP-style encoders on AI Hub.

## Main Steps
1. Export ONNX encoders.
2. Compile and profile on AI Hub.
3. Upload evaluation dataset in expected format.
4. Run inference and compute retrieval metric (Recall@10).

## Important Files
- export_onnx.py
- compile_and_profile.py
- upload_dataset.py
- inference.py

## Team Use
- Reuse this flow for checking I/O format and baseline submission readiness.
- Adapt preprocessing carefully to maintain shape and tokenizer compatibility.
