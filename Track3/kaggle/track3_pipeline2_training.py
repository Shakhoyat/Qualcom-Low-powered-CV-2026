# %% [markdown]
# # Track 3 — Pipeline 2: ConvNeXt-Tiny + Qwen2-VL-2B DoRA
# **LPCVC 2026 · AI Generated Image Detection**
#
# ## What's different from Pipeline 1
# | | Pipeline 1 | Pipeline 2 (this) |
# |---|---|---|
# | Base model | Qwen2-VL-2B + QLoRA | Qwen2-VL-2B + DoRA |
# | Architecture | VLM only (two-pass) | ConvNeXt-Tiny cascade + VLM (single-pass) |
# | Inference | Reasoning → JSON | Prefill-guided JSON (one shot) |
# | Training data | 9,000 GPT-4o annotated | 2,250 Zoom-In annotated (quality > quantity) |
# | Augmentation | None | B-Free: JPEG Q=75 + Gaussian blur |
# | Curriculum | Off | 3-epoch progressive difficulty |
# | Quantization target | W4A8 PTQ | W8A8 QAT (set in AIMET step) |
# | Criteria order | Arbitrary | Global → Local (sequential attention) |
#
# ## How to use
# 1. Run `auto_annotate_p2.py` locally → upload `annotations_p2_train.json` + `annotations_p2_val.json`
#    as a Kaggle dataset named **track3-p2-annotations**
# 2. Add this file as a Kaggle notebook script
# 3. GPU: **T4 x2** (or A100 for faster curriculum)
# 4. Kaggle secret: `QAI_HUB_API_TOKEN`
# 5. Run all cells top to bottom
#
# **Outputs:**
# - `/kaggle/working/convnext_detector.pt` — ConvNeXt-Tiny binary classifier weights
# - `/kaggle/working/qwen2vl_p2_merged/` — DoRA-merged float model ready for AIMET W8A8 QAT

# %% [markdown]
# ## Cell 1 — Install Dependencies

# %%
import subprocess
subprocess.run([
    "pip", "install", "-q",
    "transformers==4.45.2",
    "peft==0.13.2",
    "trl==0.12.0",
    "bitsandbytes==0.44.1",
    "accelerate==1.1.1",
    "timm==1.0.9",         # ConvNeXt-Tiny
    "albumentations==1.4.15",  # B-Free augmentation
    "pillow",
    "torch",
    "torchvision",
], check=True)

print("Dependencies installed.")

# %% [markdown]
# ## Cell 2 — Config & Secrets

# %%
import os
import json
import random
import io
import shutil
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

try:
    from kaggle_secrets import UserSecretsClient
    os.environ["QAI_HUB_API_TOKEN"] = UserSecretsClient().get_secret("QAI_HUB_API_TOKEN")
    print("AI Hub token loaded.")
except Exception:
    print("Kaggle secrets not available — QAI_HUB_API_TOKEN must be set externally.")

# ── Paths ────────────────────────────────────────────────────────────────────
TRAIN_JSON = "/kaggle/input/track3-p2-annotations/annotations_p2_train.json"
VAL_JSON   = "/kaggle/input/track3-p2-annotations/annotations_p2_val.json"

CNN_SAVE_PATH  = "/kaggle/working/convnext_detector.pt"
LORA_SAVE_DIR  = "/kaggle/working/qwen2vl_p2_lora"
MERGED_DIR     = "/kaggle/working/qwen2vl_p2_merged"

# ── Model ─────────────────────────────────────────────────────────────────────
BASE_VLM   = "Qwen/Qwen2-VL-2B-Instruct"
CNN_MODEL  = "convnext_tiny"     # timm model name

# ── VLM hyperparams (DoRA) ───────────────────────────────────────────────────
LORA_RANK        = 64
LORA_ALPHA       = 128
LORA_DROPOUT     = 0.05
LEARNING_RATE    = 2e-4
NUM_EPOCHS       = 3
BATCH_SIZE       = 1            # per device — T4 limit
GRAD_ACCUM_STEPS = 16
MAX_SEQ_LEN      = 1536         # reduced vs P1 (single-pass is shorter)
WARMUP_RATIO     = 0.05
SAVE_STEPS       = 200
LOGGING_STEPS    = 10

# ── CNN hyperparams ──────────────────────────────────────────────────────────
CNN_EPOCHS        = 5
CNN_LR            = 1e-4
CNN_BATCH_SIZE    = 32
CNN_IMG_SIZE      = 224

# ── Inference config ──────────────────────────────────────────────────────────
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 512 * 28 * 28

# ── Criteria order (global → local — must match prompts_p2.py) ───────────────
CRITERIA_P2 = [
    "Perspective & Spatial Relationships",
    "Physical & Common Sense Logic",
    "Lighting & Shadows Consistency",
    "Human & Biological Structure Integrity",
    "Material & Object Details",
    "Texture & Resolution",
    "Edges & Boundaries",
    "Text & Symbols",
]

LABEL2INT = {"Real": 0, "AI-Generated": 1}

print(f"Device count: {torch.cuda.device_count()}")
print(f"CUDA available: {torch.cuda.is_available()}")

# %% [markdown]
# ## Cell 3 — Load Data + B-Free Augmentation

# %%
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np

# ── B-Free augmentation (destroys high-frequency generator artifacts) ─────────
# Epoch 1: No augmentation (obvious fakes — learn basic patterns)
# Epoch 2: Moderate JPEG + blur (intermediate difficulty)
# Epoch 3: Heavy JPEG + blur (subtle semantic reasoning required)

def get_augmentation_pipeline(difficulty: int):
    """difficulty: 1=none, 2=moderate, 3=heavy"""
    base = [
        A.Resize(CNN_IMG_SIZE, CNN_IMG_SIZE),
        A.HorizontalFlip(p=0.5),
    ]
    if difficulty == 2:
        base += [
            A.ImageCompression(quality_lower=70, quality_upper=85, p=0.7),
            A.GaussianBlur(blur_limit=(3, 5), p=0.4),
        ]
    elif difficulty == 3:
        base += [
            A.ImageCompression(quality_lower=60, quality_upper=75, p=0.9),  # JPEG Q~75 per B-Free
            A.GaussianBlur(blur_limit=(3, 7), p=0.7),
            A.GaussNoise(var_limit=(5.0, 15.0), p=0.3),
        ]
    base += [
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
    return A.Compose(base)


# ── CNN Dataset ────────────────────────────────────────────────────────────────
class CNNDataset(Dataset):
    def __init__(self, records: list[dict], difficulty: int = 1):
        self.records = records
        self.transform = get_augmentation_pipeline(difficulty)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        img = np.array(Image.open(r["image"]).convert("RGB"))
        augmented = self.transform(image=img)
        x = augmented["image"]
        y = float(LABEL2INT[r["label"]])
        return x, torch.tensor(y, dtype=torch.float32)


# ── Load annotations ──────────────────────────────────────────────────────────
with open(TRAIN_JSON) as f:
    train_records = json.load(f)
with open(VAL_JSON) as f:
    val_records = json.load(f)

print(f"Train: {len(train_records)} | Val: {len(val_records)}")
ai_count = sum(1 for r in train_records if r["label"] == "AI-Generated")
real_count = sum(1 for r in train_records if r["label"] == "Real")
print(f"  Train distribution — AI: {ai_count}, Real: {real_count}")

# %% [markdown]
# ## Cell 4 — Train ConvNeXt-Tiny Binary Detector

# %%
import timm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_convnext(train_records, val_records):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training ConvNeXt-Tiny on {device}")

    model = timm.create_model(CNN_MODEL, pretrained=True, num_classes=1)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=CNN_LR, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    best_val_acc = 0.0

    for epoch in range(1, CNN_EPOCHS + 1):
        # Progressive difficulty: epoch 1=easy, 2=moderate, 3+=hard
        difficulty = min(epoch, 3)
        train_ds = CNNDataset(train_records, difficulty=difficulty)
        val_ds   = CNNDataset(val_records,   difficulty=1)  # val always clean

        train_loader = DataLoader(train_ds, batch_size=CNN_BATCH_SIZE, shuffle=True,
                                  num_workers=2, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=CNN_BATCH_SIZE, shuffle=False,
                                  num_workers=2, pin_memory=True)

        scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader))

        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x).squeeze(1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = (torch.sigmoid(model(x).squeeze(1)) > 0.5).float()
                correct += (preds == y).sum().item()
                total += len(y)
        val_acc = correct / total

        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch}/{CNN_EPOCHS} | loss={avg_loss:.4f} | val_acc={val_acc:.3f} (difficulty={difficulty})")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CNN_SAVE_PATH)
            print(f"    → Saved best model (acc={val_acc:.3f})")

    print(f"\nConvNeXt-Tiny training done. Best val acc: {best_val_acc:.3f}")
    print(f"Saved → {CNN_SAVE_PATH}")
    return best_val_acc


cnn_best_acc = train_convnext(train_records, val_records)

# %% [markdown]
# ## Cell 5 — Build CNN-Conditioned VLM Dataset

# %%
import timm

def load_convnext_for_inference(path: str, device):
    model = timm.create_model(CNN_MODEL, pretrained=False, num_classes=1)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval().to(device)
    return model


def get_cnn_score(cnn_model, image_path: str, device) -> float:
    """Returns P(AI-Generated) in [0, 1] from ConvNeXt-Tiny."""
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((CNN_IMG_SIZE, CNN_IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = cnn_model(x).squeeze()
        score = torch.sigmoid(logit).item()
    return round(score, 3)


# ── System prompt + prompts (inline — no external file dependency on Kaggle) ──
SYSTEM_PROMPT_P2 = (
    "You are a forensic image analyst. "
    "When shown an image, you evaluate 8 physical and visual criteria strictly in the order given, "
    "scanning from global scene structure down to fine local details. "
    "You output ONLY a single valid JSON object — no preamble, no markdown, no extra text."
)

PROMPT_P2 = (
    "A fast binary classifier estimates P(AI-Generated)={cnn_score:.3f}. "
    "Use this as a prior but ground your reasoning in visual evidence.\n\n"
    "Examine this image forensically across 8 criteria (in the exact order below). "
    "For each criterion: score=1 if suspicious/anomalous, score=0 if natural. "
    "Evidence strings MUST be under 30 words — specific and observational.\n\n"
    "1. Perspective & Spatial Relationships — vanishing points, relative sizes, depth plausibility.\n"
    "2. Physical & Common Sense Logic — physics violations, impossible configurations.\n"
    "3. Lighting & Shadows Consistency — light source direction, cast shadows, highlights.\n"
    "4. Human & Biological Structure Integrity — fingers, eyes, skin, hair coherence.\n"
    "5. Material & Object Details — glass/metal/fabric realism, specularity.\n"
    "6. Texture & Resolution — uniform smoothness, repetitive patterns.\n"
    "7. Edges & Boundaries — halos, blending artifacts, unnatural contours.\n"
    "8. Text & Symbols — legibility of any text, logos, numbers.\n\n"
    "Respond with ONLY the JSON object (overall_likelihood + per_criterion array):"
)

# Prefill: injected at the START of the assistant's generation to force JSON mode
PREFILL_JSON_START = '{"overall_likelihood": "'


def build_training_text(record: dict, processor, cnn_score: float) -> str:
    """
    Build the full chat-template text for one training sample.
    The prefill is included as part of the assistant's content so loss is
    computed starting from the JSON itself (we mask the prompt tokens later).
    """
    image = Image.open(record["image"]).convert("RGB")
    label = record["label"]
    output_json = record["output"]
    output_str = json.dumps(output_json, ensure_ascii=False)

    user_content = PROMPT_P2.format(cnn_score=cnn_score)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_P2},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_content},
            ],
        },
        # Assistant response starts with prefill JSON prefix embedded in the target
        {"role": "assistant", "content": output_str},
    ]

    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


# ── Pre-compute CNN scores for all training records ──────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = load_convnext_for_inference(CNN_SAVE_PATH, device)
print("Computing CNN scores for training data...")

for r in train_records + val_records:
    r["cnn_score"] = get_cnn_score(cnn_model, r["image"], device)

ai_mean   = sum(r["cnn_score"] for r in train_records if r["label"] == "AI-Generated") / max(1, ai_count)
real_mean = sum(r["cnn_score"] for r in train_records if r["label"] == "Real") / max(1, real_count)
print(f"CNN score mean — AI: {ai_mean:.3f}, Real: {real_mean:.3f}")

# Clean up CNN model from GPU to free VRAM for VLM
del cnn_model
torch.cuda.empty_cache()

# %% [markdown]
# ## Cell 6 — Load Qwen2-VL-2B with DoRA (4-bit base for VRAM)

# %%
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

print(f"Loading {BASE_VLM} with DoRA...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    BASE_VLM,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)
base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)

processor = AutoProcessor.from_pretrained(
    BASE_VLM,
    min_pixels=MIN_PIXELS,
    max_pixels=MAX_PIXELS,
)

# ── DoRA config (key difference from Pipeline 1) ─────────────────────────────
# use_dora=True decomposes weights into magnitude + direction components.
# Outperforms standard LoRA on visual instruction tasks by +0.6 to +1.9 points.
dora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    use_dora=True,  # ← Pipeline 2 key change
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(base_model, dora_config)
model.print_trainable_parameters()

# %% [markdown]
# ## Cell 7 — VLM Dataset with Curriculum Augmentation

# %%
from PIL import Image as PILImage
import io as _io

def apply_bfree_augmentation(image: PILImage.Image, difficulty: int) -> PILImage.Image:
    """
    B-Free augmentation: destroys high-frequency generator artifacts so the model
    must rely on semantic (physical/logical) reasoning rather than pixel noise.
    """
    if difficulty == 1:
        return image
    buf = _io.BytesIO()
    quality = 75 if difficulty == 3 else 82
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    img = PILImage.open(buf).convert("RGB")

    if difficulty >= 2:
        from PIL import ImageFilter
        sigma = 0.8 if difficulty == 2 else 1.2
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def find_assistant_response_spans(input_ids: list[int], tokenizer) -> list[tuple[int, int]]:
    """
    Locate <|im_start|>assistant ... <|im_end|> spans.
    Loss is computed ONLY on these tokens — prompt tokens are masked to -100.
    """
    assistant_start_ids = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
    end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]

    spans = []
    i = 0
    while i < len(input_ids):
        # Check for assistant start token sequence
        if input_ids[i:i + len(assistant_start_ids)] == assistant_start_ids:
            start = i + len(assistant_start_ids)
            # Skip newline token after 'assistant' header
            if start < len(input_ids) and input_ids[start] in (
                tokenizer.encode("\n", add_special_tokens=False) + [198]
            ):
                start += 1
            # Find end token
            end = start
            while end < len(input_ids) and input_ids[end] != end_id:
                end += 1
            if end < len(input_ids):
                spans.append((start, end))
            i = end + 1
        else:
            i += 1
    return spans


class VLMDataset(Dataset):
    """
    Single-pass VLM dataset with CNN score conditioning and B-Free augmentation.
    Curriculum difficulty is set externally before each epoch.
    """

    def __init__(self, records: list[dict], processor, difficulty: int = 1):
        self.records = records
        self.processor = processor
        self.difficulty = difficulty

    def set_difficulty(self, difficulty: int):
        self.difficulty = difficulty

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        cnn_score = r.get("cnn_score", 0.5)
        label = r["label"]
        output_json = r["output"]
        output_str = json.dumps(output_json, ensure_ascii=False)

        user_content = PROMPT_P2.format(cnn_score=cnn_score)

        image = PILImage.open(r["image"]).convert("RGB")
        image = apply_bfree_augmentation(image, self.difficulty)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_P2},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_content},
                ],
            },
            {"role": "assistant", "content": output_str},
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            max_length=MAX_SEQ_LEN,
            truncation=True,
            padding=False,
        )

        input_ids = inputs["input_ids"][0].tolist()
        labels = [-100] * len(input_ids)

        spans = find_assistant_response_spans(input_ids, self.processor.tokenizer)
        for start, end in spans:
            for j in range(start, end):
                labels[j] = input_ids[j]

        return {
            "input_ids": inputs["input_ids"][0],
            "attention_mask": inputs["attention_mask"][0],
            "pixel_values": inputs.get("pixel_values"),
            "image_grid_thw": inputs.get("image_grid_thw"),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def vlm_collate_fn(batch: list[dict]) -> dict:
    """
    Collate for Qwen2-VL:
    - Pad input_ids, attention_mask, labels to max length in batch
    - pixel_values and image_grid_thw must be CAT not STACK (Qwen2-VL constraint)
    """
    max_len = max(item["input_ids"].shape[0] for item in batch)
    pad_id = 0
    label_pad = -100

    input_ids_padded = []
    attn_mask_padded = []
    labels_padded = []

    for item in batch:
        seq_len = item["input_ids"].shape[0]
        pad_len = max_len - seq_len

        input_ids_padded.append(
            F.pad(item["input_ids"], (0, pad_len), value=pad_id)
        )
        attn_mask_padded.append(
            F.pad(item["attention_mask"], (0, pad_len), value=0)
        )
        labels_padded.append(
            F.pad(item["labels"], (0, pad_len), value=label_pad)
        )

    result = {
        "input_ids": torch.stack(input_ids_padded),
        "attention_mask": torch.stack(attn_mask_padded),
        "labels": torch.stack(labels_padded),
    }

    # Qwen2-VL: cat pixel_values and image_grid_thw along batch dim
    pixel_values = [item["pixel_values"] for item in batch if item["pixel_values"] is not None]
    image_grid_thw = [item["image_grid_thw"] for item in batch if item["image_grid_thw"] is not None]
    if pixel_values:
        result["pixel_values"] = torch.cat(pixel_values, dim=0)
    if image_grid_thw:
        result["image_grid_thw"] = torch.cat(image_grid_thw, dim=0)

    return result

# %% [markdown]
# ## Cell 8 — Train VLM with Curriculum Learning

# %%
from transformers import TrainingArguments, Trainer

# ── Curriculum learning: 3 epochs of progressive difficulty ──────────────────
# Epoch 1: No augmentation — model sees clean, obvious fakes → learns basic patterns
# Epoch 2: Moderate JPEG + blur — intermediate difficulty
# Epoch 3: Heavy JPEG + blur — semantic reasoning required (B-Free regime)

CURRICULUM = {1: 1, 2: 2, 3: 3}  # epoch → difficulty level

train_dataset = VLMDataset(train_records, processor, difficulty=1)
val_dataset   = VLMDataset(val_records,   processor, difficulty=1)


class CurriculumTrainer(Trainer):
    """Trainer that updates dataset difficulty at the start of each epoch."""

    def __init__(self, curriculum: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.curriculum = curriculum

    def training_step(self, model, inputs, num_items_in_batch=None):
        current_epoch = int(self.state.epoch) + 1
        new_difficulty = self.curriculum.get(current_epoch, 3)
        if self.train_dataset.difficulty != new_difficulty:
            self.train_dataset.set_difficulty(new_difficulty)
            if self.args.local_rank in (-1, 0):
                print(f"\n[Curriculum] Epoch {current_epoch} → difficulty={new_difficulty}")
        if num_items_in_batch is not None:
            return super().training_step(model, inputs, num_items_in_batch)
        return super().training_step(model, inputs)


training_args = TrainingArguments(
    output_dir=LORA_SAVE_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",
    fp16=True,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    logging_steps=LOGGING_STEPS,
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",
    dataloader_num_workers=0,
    remove_unused_columns=False,
)

trainer = CurriculumTrainer(
    curriculum=CURRICULUM,
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=vlm_collate_fn,
)

print(f"Starting curriculum VLM training: {NUM_EPOCHS} epochs × {len(train_records)} samples")
print(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM_STEPS}")
trainer.train()
trainer.save_model(LORA_SAVE_DIR)
processor.save_pretrained(LORA_SAVE_DIR)
print(f"DoRA adapter saved → {LORA_SAVE_DIR}")

# %% [markdown]
# ## Cell 9 — Merge DoRA Weights + Save (CPU to avoid OOM)

# %%
from transformers import Qwen2VLForConditionalGeneration
from peft import PeftModel

print("Merging DoRA weights on CPU (avoids T4 OOM)...")

# Load base model in float16 on CPU
base_for_merge = Qwen2VLForConditionalGeneration.from_pretrained(
    BASE_VLM,
    torch_dtype=torch.float16,
    device_map="cpu",
)

# Load DoRA adapter
merged_model = PeftModel.from_pretrained(base_for_merge, LORA_SAVE_DIR)

# Merge and unload LoRA/DoRA weights into base
merged_model = merged_model.merge_and_unload()

Path(MERGED_DIR).mkdir(parents=True, exist_ok=True)
merged_model.save_pretrained(MERGED_DIR, safe_serialization=True)
processor.save_pretrained(MERGED_DIR)

print(f"Merged model saved → {MERGED_DIR}")
print("Size on disk:")
import subprocess
subprocess.run(["du", "-sh", MERGED_DIR])

# %% [markdown]
# ## Cell 10 — Quick Cascade Evaluation

# %%
@torch.no_grad()
def cascade_inference(image_path: str, cnn_model, vlm_model, processor, device: str) -> dict | None:
    """
    Single-pass cascade inference:
    1. ConvNeXt-Tiny → P(AI-Generated) score (fast, ~15ms)
    2. Qwen2-VL-2B with prefill-guided JSON generation
    """
    from torchvision import transforms

    # ── Stage 1: CNN detection ────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize((CNN_IMG_SIZE, CNN_IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img_pil = PILImage.open(image_path).convert("RGB")
    x = transform(img_pil).unsqueeze(0).to(device)
    logit = cnn_model(x).squeeze()
    cnn_score = torch.sigmoid(logit).item()

    # ── Stage 2: VLM single-pass with prefill ─────────────────────────────────
    user_content = PROMPT_P2.format(cnn_score=round(cnn_score, 3))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_P2},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_pil},
                {"type": "text", "text": user_content},
            ],
        },
    ]

    text_base = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Append prefill to force JSON generation from the first token
    text_with_prefill = text_base + PREFILL_JSON_START

    inputs = processor(
        text=[text_with_prefill],
        images=[img_pil],
        return_tensors="pt",
    ).to(device)

    out = vlm_model.generate(
        **inputs,
        max_new_tokens=400,   # JSON ~300 tokens + headroom
        temperature=0.0,
        do_sample=False,
        pad_token_id=processor.tokenizer.eos_token_id,
    )

    generated = processor.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

    # Reconstruct full JSON (prepend the prefill we consumed)
    full_json_str = PREFILL_JSON_START + generated
    if full_json_str.endswith("```"):
        full_json_str = full_json_str[: full_json_str.rfind("```")]

    try:
        parsed = json.loads(full_json_str)
        return parsed, cnn_score
    except json.JSONDecodeError:
        return None, cnn_score


def evaluate_cascade(records, cnn_model, vlm_model, processor, device, limit=50):
    cnn_model.eval()
    vlm_model.eval()
    results = []

    for i, r in enumerate(records[:limit]):
        output, cnn_score = cascade_inference(r["image"], cnn_model, vlm_model, processor, device)
        valid = output is not None
        pred = output.get("overall_likelihood", "INVALID") if output else "INVALID"
        correct = pred == r["label"]
        results.append({"correct": correct, "valid": valid, "cnn_score": cnn_score})

        icon = "✓" if correct else "✗"
        print(f"  [{i+1}/{limit}] {icon} pred={pred} true={r['label']} cnn={cnn_score:.3f}")

    total = len(results)
    accuracy = sum(1 for r in results if r["correct"]) / total
    validity = sum(1 for r in results if r["valid"]) / total
    print(f"\nCascade eval ({total} samples):")
    print(f"  Detection accuracy: {accuracy:.3f}")
    print(f"  JSON validity rate: {validity:.3f}")
    return accuracy, validity


# Load merged model for evaluation
print("Loading merged model for cascade evaluation...")
eval_vlm = Qwen2VLForConditionalGeneration.from_pretrained(
    MERGED_DIR,
    torch_dtype=torch.float16,
    device_map="auto",
)
eval_vlm.eval()

eval_cnn = load_convnext_for_inference(CNN_SAVE_PATH, device)

acc, valid_rate = evaluate_cascade(
    val_records, eval_cnn, eval_vlm, processor, device="cuda", limit=50
)

print("\n" + "=" * 60)
print("PIPELINE 2 EVALUATION SUMMARY")
print("=" * 60)
print(f"  ConvNeXt-Tiny val accuracy:  {cnn_best_acc:.3f}")
print(f"  Cascade detection accuracy:  {acc:.3f}")
print(f"  JSON validity rate:          {valid_rate:.3f}")
print(f"\nNext step: Download {MERGED_DIR} and {CNN_SAVE_PATH}")
print("Then run AIMET QAT (W8A8) — see Track3/README.md Step 7")

# %% [markdown]
# ## Cell 11 — Save Experiment Results

# %%
results_path = Path("/kaggle/working/pipeline2_results.json")
results_path.write_text(json.dumps({
    "pipeline": "pipeline2_dora_cascade",
    "base_vlm": BASE_VLM,
    "cnn_model": CNN_MODEL,
    "use_dora": True,
    "lora_rank": LORA_RANK,
    "curriculum_epochs": NUM_EPOCHS,
    "augmentation": "bfree_jpeg75_gaussian",
    "n_train_samples": len(train_records),
    "cnn_val_accuracy": round(cnn_best_acc, 4),
    "cascade_accuracy": round(acc, 4),
    "json_validity_rate": round(valid_rate, 4),
    "quantization_target": "W8A8_QAT_AIMET",
    "notes": "DoRA + curriculum + B-Free + prefill-guided single-pass inference",
}, indent=2))

print(f"Results saved → {results_path}")
print("\nFiles to download:")
print(f"  {MERGED_DIR}/  (~4.5 GB VLM)")
print(f"  {CNN_SAVE_PATH}  (~100 MB ConvNeXt)")
print(f"  {results_path}")
