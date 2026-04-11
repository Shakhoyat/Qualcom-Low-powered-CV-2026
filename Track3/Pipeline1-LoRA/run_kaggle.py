"""
Pipeline 1 — Kaggle launcher shim.

This file is a pointer. The actual training notebook is at:
    Track3/kaggle/track3_training.py

Upload THAT file to Kaggle, not this one.

Quick reference — what to upload to Kaggle:
    1. track3_training.py         (notebook script)
    2. annotations_train.json     (as dataset "track3-annotations")
    3. annotations_val.json       (same dataset)

Kaggle secrets needed:
    QAI_HUB_API_TOKEN = wyfb5d78py0bgow6gz10krdqyd8qq0l9wlfw9swc

GPU setting: T4 x2

After training, download:
    /kaggle/working/qwen2vl_merged/   (~4.5 GB)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kaggle"))

print("Pipeline 1 Kaggle script is at: Track3/kaggle/track3_training.py")
print("Upload that file directly to Kaggle.")
print("See Pipeline1-LoRA/README.md for full step-by-step instructions.")
