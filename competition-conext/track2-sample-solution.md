# Track 2 Sample Solution Notes

Source: https://github.com/lpcvai/26LPCVC_Track2_Sample_Solution

## Purpose
- End-to-end baseline for QEVD action recognition and AI Hub deployment.

## Main Pipeline
1. Refactor and validate QEVD videos.
2. Train or load checkpoint.
3. Preprocess videos to tensors.
4. Export/compile model on AI Hub.
5. Run on-device inference.
6. Evaluate Top-1 and Top-5.

## Important Files
- preprocess_and_save.py
- example_export.py
- compile_and_profile.py
- run_inference.py
- evaluate.py

## Team Use
- Match frame count and tensor layout across preprocess, compile, and inference.
- Store compiled assets and manifests with experiment identifiers.
