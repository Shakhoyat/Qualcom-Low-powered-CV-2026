# Track 3 — SOTA Winning Pipeline
### AI Generated Image Detection · LPCVC 2026

> **Strategy in one line:** Fine-tune Qwen2-VL-2B with LoRA on a purpose-built AI-detection dataset,
> engineer chain-of-thought prompts for all 8 criteria, then minimize quantization loss with
> domain-specific calibration data — maximizing both detection accuracy and explanation quality simultaneously.

---

## Why This Pipeline Beats the Baseline

| Factor | Baseline (sample solution) | This Pipeline |
|--------|---------------------------|---------------|
| Model weights | Off-the-shelf Qwen2-VL-2B | LoRA fine-tuned on AI detection |
| Training data | None (zero-shot) | 120k+ labeled real+AI images |
| Prompting | Generic | Expert chain-of-thought across all 8 criteria |
| Calibration data | COCO images (generic) | Mixed real+AI images (domain-matched) |
| Expected detection | ~60-70% (zero-shot VLM) | ~85-92% (fine-tuned) |
| Explanation quality | Generic descriptions | Criterion-specific, evidence-grounded JSON |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT IMAGE                              │
└───────────────────────────┬─────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  PROMPT STEP 1  │  ← Chain-of-thought prompt
                    │  "Analyze image │     across all 8 criteria
                    │   across 8      │
                    │   criteria..."  │
                    └────────┬────────┘
                             │
              ┌──────────────▼──────────────┐
              │   Qwen2-VL-2B-Instruct      │
              │   + LoRA (r=64, α=128)      │  ← Fine-tuned on
              │   Fine-tuned for AI         │    AI detection task
              │   image detection           │
              └──────────────┬──────────────┘
                             │ Free-form reasoning text
                    ┌────────▼────────┐
                    │  PROMPT STEP 2  │  ← "Convert your analysis
                    │  "Emit JSON..." │     to this exact schema"
                    └────────┬────────┘
                             │
              ┌──────────────▼──────────────┐
              │   Same VLM (second pass)    │
              └──────────────┬──────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  OUTPUT JSON                                                      │
│  {                                                                │
│    "overall_likelihood": "AI-Generated" | "Real",                │
│    "per_criterion": [                                             │
│      { "criterion": "Lighting & Shadows Consistency",            │
│        "score": 0 | 1,                                           │
│        "evidence": "..." },                                       │
│      ... × 8                                                      │
│    ]                                                              │
│  }                                                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1 — Dataset Collection (Kaggle + Local)

### 1.1 Datasets to Use

| Dataset | Size | Source | Command |
|---------|------|--------|---------|
| CIFAKE | 120k (60k real, 60k AI) | Kaggle | `kaggle datasets download -d bird/cifake-real-and-ai-generated-synthetic-images` |
| ArtiFact | ~2.4M images (labeled) | GitHub/HF | `git clone https://github.com/awsaf49/artifact` |
| GenImage | ~1.35M multi-generator | HuggingFace | `huggingface-cli download 'tessio/GenImage'` |
| COCO 2017 val | 5k real | COCO | `wget http://images.cocodataset.org/zips/val2017.zip` |
| Raise dataset | 8k real raw photos | raise.disi.unitn.it | Manual download |

### 1.2 Supplement with Generated Images (improves generalization)

Generate 5-10k additional AI images covering generators that may be in the test set:

```python
# Option A: Stable Diffusion XL (local)
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")

prompts = [
    "a photo of a person smiling outdoors",
    "a cityscape at night with lights",
    "a close-up photo of a cat",
    # ... add ~50 diverse prompts
]

for i, prompt in enumerate(prompts):
    for j in range(100):  # 100 images per prompt = 5k total
        img = pipe(prompt, num_inference_steps=30).images[0]
        img.save(f"ai_generated/sdxl_{i}_{j}.jpg")
```

```python
# Option B: DALL-E 3 via API (diverse style)
import openai, requests, os
from pathlib import Path

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
Path("ai_generated/dalle3").mkdir(parents=True, exist_ok=True)

prompts = ["photo of a mountain lake", "portrait of a businessman", ...]
for i, prompt in enumerate(prompts):
    response = client.images.generate(model="dall-e-3", prompt=prompt, size="1024x1024")
    img_data = requests.get(response.data[0].url).content
    with open(f"ai_generated/dalle3/img_{i}.jpg", "wb") as f:
        f.write(img_data)
```

### 1.3 Build Annotation File

Each training sample needs the image path + the target JSON output:

```python
import json

def build_annotation(image_path: str, is_ai: bool, criteria_scores: dict) -> dict:
    """
    criteria_scores: dict mapping criterion name → (score 0/1, evidence string)
    """
    return {
        "image": image_path,
        "label": "AI-Generated" if is_ai else "Real",
        "output": {
            "overall_likelihood": "AI-Generated" if is_ai else "Real",
            "per_criterion": [
                {
                    "criterion": c,
                    "score": criteria_scores[c][0],
                    "evidence": criteria_scores[c][1]
                }
                for c in [
                    "Lighting & Shadows Consistency",
                    "Edges & Boundaries",
                    "Texture & Resolution",
                    "Perspective & Spatial Relationships",
                    "Physical & Common Sense Logic",
                    "Text & Symbols",
                    "Human & Biological Structure Integrity",
                    "Material & Object Details",
                ]
            ]
        }
    }
```

> **Bootstrap trick:** For datasets without per-criterion annotations, use GPT-4o or Claude 3.5 Sonnet
> to auto-generate the evidence strings for each criterion. Run at scale offline. Cost ~$0.002/image.
> This becomes your synthetic supervision signal.

```python
# Auto-annotate with GPT-4o (run offline, one time)
import base64, json, openai
from pathlib import Path

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ANNOTATION_PROMPT = """You are an expert at detecting AI-generated images.
Analyze this image and provide a structured assessment.

For each of these 8 criteria, provide:
- score: 0 if looks natural/authentic, 1 if suspicious/AI-artifact
- evidence: a specific, detailed observation (1-2 sentences)

Criteria:
1. Lighting & Shadows Consistency
2. Edges & Boundaries  
3. Texture & Resolution
4. Perspective & Spatial Relationships
5. Physical & Common Sense Logic
6. Text & Symbols
7. Human & Biological Structure Integrity
8. Material & Object Details

Also provide overall_likelihood: "AI-Generated" or "Real"

Respond ONLY with valid JSON matching this schema:
{
  "overall_likelihood": "AI-Generated" | "Real",
  "per_criterion": [
    {"criterion": "...", "score": 0|1, "evidence": "..."},
    ...
  ]
}"""

def annotate_image(image_path: str, known_label: str) -> dict:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    
    ext = Path(image_path).suffix.lower()
    mime = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": ANNOTATION_PROMPT + f"\n\nHint: this image is {known_label}"}
            ]
        }],
        max_tokens=800
    )
    return json.loads(response.choices[0].message.content)
```

---

## Phase 2 — Prompt Engineering (Critical for Score)

The prompts are **locked into the model weights during fine-tuning** — use these exact templates.

### Prompt 1 — Chain-of-Thought Analysis

```python
SYSTEM_PROMPT = """You are a forensic image analyst specializing in detecting AI-generated content.
You examine images with scientific precision, looking for subtle artifacts that differentiate
AI-generated images from authentic photographs."""

PROMPT_1_TEMPLATE = """Carefully examine this image and analyze it across these 8 forensic criteria.
For each criterion, think step by step about what you observe.

1. **Lighting & Shadows Consistency**: Are light sources consistent? Do shadows fall correctly?
   Look for impossible highlights, missing cast shadows, or inconsistent shadow directions.

2. **Edges & Boundaries**: Examine object boundaries carefully. AI images often show
   unnatural smoothing, halos, or blending artifacts at edges between objects.

3. **Texture & Resolution**: Is texture uniformly sharp or strangely uniform?
   AI images often have either over-smooth textures or repetitive texture patterns.

4. **Perspective & Spatial Relationships**: Do objects maintain correct relative sizes?
   Are parallel lines converging correctly? Does depth look physically plausible?

5. **Physical & Common Sense Logic**: Do all elements make physical sense together?
   Look for impossible object combinations or physically implausible configurations.

6. **Text & Symbols**: Examine any text, logos, signs, numbers, or symbols closely.
   AI models frequently corrupt text into illegible or nonsensical characters.

7. **Human & Biological Structure Integrity**: Check hands (finger count/shape),
   eyes (symmetry/reflections), ears, teeth, hair strand coherence, skin texture.

8. **Material & Object Details**: Are material properties (glass, metal, fabric, skin)
   rendered with realistic light interaction? Look for plastic-looking skin or
   unrealistic material specularity.

Analyze each criterion and provide your detailed forensic observations."""
```

### Prompt 2 — JSON Extraction

```python
PROMPT_2_TEMPLATE = """Based on your forensic analysis above, provide your final assessment
in the following exact JSON format. Use only the criterion names as specified.

{{
  "overall_likelihood": "<AI-Generated or Real>",
  "per_criterion": [
    {{
      "criterion": "Lighting & Shadows Consistency",
      "score": <0 for natural, 1 for suspicious>,
      "evidence": "<your specific observation from the analysis>"
    }},
    {{
      "criterion": "Edges & Boundaries",
      "score": <0 or 1>,
      "evidence": "<specific observation>"
    }},
    {{
      "criterion": "Texture & Resolution",
      "score": <0 or 1>,
      "evidence": "<specific observation>"
    }},
    {{
      "criterion": "Perspective & Spatial Relationships",
      "score": <0 or 1>,
      "evidence": "<specific observation>"
    }},
    {{
      "criterion": "Physical & Common Sense Logic",
      "score": <0 or 1>,
      "evidence": "<specific observation>"
    }},
    {{
      "criterion": "Text & Symbols",
      "score": <0 or 1>,
      "evidence": "<specific observation>"
    }},
    {{
      "criterion": "Human & Biological Structure Integrity",
      "score": <0 or 1>,
      "evidence": "<specific observation>"
    }},
    {{
      "criterion": "Material & Object Details",
      "score": <0 or 1>,
      "evidence": "<specific observation>"
    }}
  ]
}}

Respond with ONLY the JSON. No markdown, no explanation."""
```

---

## Phase 3 — LoRA Fine-tuning on Kaggle GPU

### 3.1 Kaggle Notebook Setup

```python
# Cell 1: Install dependencies
!pip install transformers==4.45.0 peft==0.13.0 trl==0.12.0 \
    bitsandbytes accelerate datasets pillow --quiet

import os
import json
import torch
from pathlib import Path
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

# Load API token from Kaggle secrets
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
os.environ["QAI_HUB_API_TOKEN"] = secrets.get_secret("QAI_HUB_API_TOKEN")
```

### 3.2 Load Base Model with QLoRA (4-bit to fit in Kaggle T4)

```python
# Cell 2: Load model in 4-bit for QLoRA training
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    min_pixels=256*28*28,
    max_pixels=512*28*28,
)
```

### 3.3 Configure LoRA

```python
# Cell 3: LoRA config — target all attention + MLP layers
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64,                          # rank — higher = more capacity
    lora_alpha=128,                # scaling factor
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",   # attention
        "gate_proj", "up_proj", "down_proj",         # MLP
    ],
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected: ~40-50M trainable params out of 2B total (~2%)
```

### 3.4 Dataset Formatting

```python
# Cell 4: Format training samples
def format_sample(sample: dict, processor) -> dict:
    """Convert annotation to model input format."""
    image = Image.open(sample["image"]).convert("RGB")
    
    # Build the two-turn conversation
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT_1_TEMPLATE}
            ]
        },
        {
            "role": "assistant",
            "content": sample["reasoning_text"]  # chain-of-thought reasoning
        },
        {
            "role": "user",
            "content": PROMPT_2_TEMPLATE
        },
        {
            "role": "assistant",
            "content": json.dumps(sample["output"], indent=2)  # target JSON
        }
    ]
    
    # Apply chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )
    
    inputs["labels"] = inputs["input_ids"].clone()
    return inputs

# Load your annotation file
with open("annotations_train.json") as f:
    annotations = json.load(f)

train_dataset = Dataset.from_list(annotations)
```

### 3.5 Train

```python
# Cell 5: Training config and run
training_args = SFTConfig(
    output_dir="/kaggle/working/qwen2vl_lora_checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=1,   # T4 memory limit
    gradient_accumulation_steps=16,  # effective batch = 16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,
    logging_steps=10,
    save_steps=200,
    save_total_limit=3,
    dataloader_num_workers=2,
    remove_unused_columns=False,
    max_seq_length=2048,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=lambda x: x,  # pre-formatted
)

trainer.train()

# Save LoRA weights
model.save_pretrained("/kaggle/working/qwen2vl_lora_final")
processor.save_pretrained("/kaggle/working/qwen2vl_lora_final")
print("Training done. Merge and export next.")
```

### 3.6 Merge LoRA → Full Float Model

```python
# Cell 6: Merge LoRA weights back into base model (needed for AIMET)
from peft import PeftModel
from transformers import Qwen2VLForConditionalGeneration

# Reload in float16 for merge (no quantization)
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.float16,
    device_map="cpu",  # CPU merge to avoid OOM
)

merged_model = PeftModel.from_pretrained(
    base_model, "/kaggle/working/qwen2vl_lora_final"
)
merged_model = merged_model.merge_and_unload()

merged_model.save_pretrained(
    "/kaggle/working/qwen2vl_merged",
    safe_serialization=True,
)
processor.save_pretrained("/kaggle/working/qwen2vl_merged")
print("Merged model saved. Ready for AIMET quantization.")
```

---

## Phase 4 — Domain-Aware AIMET Quantization

> The key insight: calibrating with AI-detection images (not just COCO) reduces quantization error
> on the actual task, which means the quantized model behaves closer to the float model at test time.

### 4.1 Build Domain-Specific Calibration Set

```python
# Calibration data: mix of real + AI images from your training set
import random, shutil
from pathlib import Path

CALIB_DIR = Path("calibration_data/domain_mixed")
CALIB_DIR.mkdir(parents=True, exist_ok=True)

# Take 150 real + 150 AI images from your training annotations
with open("annotations_train.json") as f:
    all_samples = json.load(f)

real_samples = [s for s in all_samples if s["label"] == "Real"][:150]
ai_samples = [s for s in all_samples if s["label"] == "AI-Generated"][:150]

for i, s in enumerate(real_samples + ai_samples):
    shutil.copy(s["image"], CALIB_DIR / f"calib_{i}.jpg")

print(f"Calibration set: {len(list(CALIB_DIR.glob('*.jpg')))} images")
```

### 4.2 Run AIMET with Your Merged Model

Replace the QPM tutorial's Example 1A — substitute the model path:

```python
# In Example1A_veg.ipynb, change model loading to:
model_path = "/kaggle/working/qwen2vl_merged"  # your fine-tuned model

# And change calibration data path to your domain-specific set:
calibration_image_dir = "calibration_data/domain_mixed"
calibration_text_data = "calibration_data/llava_v1_5_mix665k_300.json"
```

Then run through Example 1A → 1B → 2A → 2B exactly as in the baseline README.

---

## Phase 5 — Evaluation Before Submission

### 5.1 Score Your Model Locally

```python
import json
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.float16,
).to("cuda")

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

def detect_image(image_path: str) -> dict:
    image = Image.open(image_path).convert("RGB")
    
    # Step 1: chain-of-thought reasoning
    messages_1 = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT_1_TEMPLATE}
        ]}
    ]
    text_1 = processor.apply_chat_template(messages_1, add_generation_prompt=True)
    inputs_1 = processor(text=[text_1], images=[image], return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        out_1 = model.generate(**inputs_1, max_new_tokens=600, temperature=0.1)
    reasoning = processor.decode(out_1[0][inputs_1["input_ids"].shape[1]:], skip_special_tokens=True)
    
    # Step 2: structured JSON extraction
    messages_2 = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT_1_TEMPLATE}
        ]},
        {"role": "assistant", "content": reasoning},
        {"role": "user", "content": PROMPT_2_TEMPLATE}
    ]
    text_2 = processor.apply_chat_template(messages_2, add_generation_prompt=True)
    inputs_2 = processor(text=[text_2], images=[image], return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        out_2 = model.generate(**inputs_2, max_new_tokens=512, temperature=0.0)
    raw_json = processor.decode(out_2[0][inputs_2["input_ids"].shape[1]:], skip_special_tokens=True)
    
    return json.loads(raw_json)

# Evaluate on your held-out validation set
correct = 0
total = 0
with open("annotations_val.json") as f:
    val_set = json.load(f)

for sample in val_set:
    try:
        pred = detect_image(sample["image"])
        if pred["overall_likelihood"] == sample["label"]:
            correct += 1
        total += 1
    except (json.JSONDecodeError, KeyError):
        total += 1  # count as wrong

print(f"Detection accuracy: {correct/total:.3f} ({correct}/{total})")
```

### 5.2 Track Your Experiments

```
experiments/
├── run_001_baseline_zero_shot/
│   ├── score.json
│   └── notes.md
├── run_002_lora_r16_cifake_only/
│   ├── score.json
│   └── notes.md
└── run_003_lora_r64_all_datasets/   ← target
    ├── score.json
    └── notes.md
```

---

## Key Decisions Summary

| Decision | Choice | Reason |
|----------|--------|--------|
| LoRA rank | r=64 | Higher capacity for complex visual reasoning |
| Batch size | 1 + grad accum 16 | T4 16GB VRAM constraint |
| Epochs | 3 | Prevent overfitting on synthetic annotations |
| Calibration data | Domain-mixed (real+AI) | Reduces quantization error on task-relevant inputs |
| Temperature at inference | 0.1 (step 1), 0.0 (step 2) | Step 2 must be deterministic JSON |
| Max tokens step 1 | 600 | Enough for thorough 8-criterion reasoning |
| Max tokens step 2 | 512 | Enough for complete JSON output |

---

## Estimated Score Gain Over Baseline

```
Baseline (zero-shot Qwen2-VL-2B):      ~60-65% detection accuracy
+ Expert prompt engineering:           +10-12%  → ~72-77%
+ LoRA fine-tuning (CIFAKE only):      + 8-10%  → ~80-87%
+ Full dataset + auto-annotation:      + 3-5%   → ~83-92%
+ Domain-matched calibration:          + 1-2%   → ~84-94% (lower quant error)
```

These are estimates — your actual gain depends on the organizer's private test distribution.

---

## What to Store in Drive (not Git)

```
Track3/models/
├── qwen2vl_merged/           # merged float model (~4.5 GB)
├── qwen2vl_lora_final/       # LoRA weights only (~200 MB)  
├── calibration_data/         # calibration images (~500 MB)
└── submission_package/       # final ZIP (~1.8 GB)
```
