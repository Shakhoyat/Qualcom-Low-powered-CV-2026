# Track 3 — Pipeline Comparison & Run Guide

## Two pipelines, run both, submit the best

| | Pipeline 1 — LoRA | Pipeline 2 — DoRA Cascade |
|---|---|---|
| **Folder** | `Pipeline1-LoRA/` | `Pipeline2-DoRA-Cascade/` |
| **Kaggle script** | `kaggle/track3_training.py` | `kaggle/track3_pipeline2_training.py` |
| **Annotation script** | `scripts/auto_annotate.py` | `scripts/auto_annotate_p2.py` |
| **Prompts** | `scripts/prompts.py` | `scripts/prompts_p2.py` |
| **Adapter** | QLoRA NF4 4-bit | DoRA (`use_dora=True`) |
| **Architecture** | VLM only | ConvNeXt-Tiny + VLM cascade |
| **Inference** | Two-pass (reasoning → JSON) | Single-pass prefill-guided JSON |
| **Training data** | 9,000 samples (~$20) | 2,250 samples (~$7.50) |
| **Augmentation** | None | B-Free JPEG+blur + curriculum |
| **Criteria order** | Arbitrary | Global → Local |
| **Quant target** | W4A8 PTQ | W8A8 QAT |
| **Expected accuracy** | 84–94% | 94–96% |
| **Kaggle GPU time** | ~6–8 h T4 x2 | ~4–5 h T4 x2 |
| **Annotation cost** | ~$20 | ~$7.50 |

## API Key (AI Hub)

```
wyfb5d78py0bgow6gz10krdqyd8qq0l9wlfw9swc
```

Store as Kaggle secret `QAI_HUB_API_TOKEN`. Use for both Track 2 and Track 3.

## Shared scripts (used by both pipelines)

```
Track3/scripts/
├── build_calibration_set.py   ← Build 300-image AIMET calibration set
├── evaluate.py                ← Two-step inference evaluation
└── experiment_log.json        ← Log all run results here
```

## Full execution order

```
# PIPELINE 1
cd Track3/scripts
python auto_annotate.py annotate --input_dir ../data/cifake/train/FAKE --label AI-Generated --output ../data/annotations_ai.json --limit 5000
python auto_annotate.py annotate --input_dir ../data/cifake/train/REAL --label Real       --output ../data/annotations_real.json --limit 5000
python auto_annotate.py merge --inputs ../data/annotations_ai.json ../data/annotations_real.json --output ../data/annotations --val_split 0.1
# → Upload to Kaggle as "track3-annotations", run track3_training.py (T4 x2, ~7h)
# → Download qwen2vl_merged/

python build_calibration_set.py --annotations ../data/annotations_train.json --output_dir ../data/calibration_domain_mixed --n_real 150 --n_ai 150
# → AIMET W4A8 PTQ via QPM tutorial
# → AIHub simulation → submit

# PIPELINE 2 (run in parallel in a separate Kaggle session)
python auto_annotate_p2.py annotate --input_dir ../data/cifake/train/FAKE --label AI-Generated --output ../data/annotations_p2_ai.json --limit 1250
python auto_annotate_p2.py annotate --input_dir ../data/coco/val2017      --label Real          --output ../data/annotations_p2_real.json --limit 1250
python auto_annotate_p2.py merge --inputs ../data/annotations_p2_ai.json ../data/annotations_p2_real.json --output ../data/annotations_p2 --val_split 0.1
# → Upload to Kaggle as "track3-p2-annotations", run track3_pipeline2_training.py (T4 x2, ~5h)
# → Download qwen2vl_p2_merged/ + convnext_detector.pt

# → AIMET W8A8 QAT via QPM tutorial (weight_bw=8, act_bw=8)
# → AIHub simulation → submit best of P1/P2
```

## Submission

- Submit both pipelines to https://lpcv.ai/2026LPCVC/submission/track3
- Share both compile jobs with `lowpowervision@gmail.com`
- The leaderboard picks your best submission automatically
