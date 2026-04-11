# %% [markdown]
# # Track 3 — Qwen2-VL-2B LoRA Fine-tuning
# **LPCVC 2026 · AI Generated Image Detection**
#
# **How to use:**
# 1. Upload `annotations_train.json` and `annotations_val.json` to Kaggle as a dataset
# 2. Add this file as a notebook script
# 3. Enable GPU T4 x2 (32 GB VRAM total)
# 4. Add secret: `QAI_HUB_API_TOKEN`
# 5. Run all cells top to bottom
#
# **Output:** `/kaggle/working/qwen2vl_merged/` — merged float model ready for AIMET

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
    "datasets==3.1.0",
    "qwen-vl-utils==0.0.8",
    "pillow",
    "torch",
], check=True)

print("Dependencies installed.")

# %% [markdown]
# ## Cell 2 — Config & Constants

# %%
import os
import json
import random
import shutil
from pathlib import Path

import torch

# ── Kaggle secrets ──────────────────────────────────────────────────────────
try:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    os.environ["QAI_HUB_API_TOKEN"] = secrets.get_secret("QAI_HUB_API_TOKEN")
    print("AI Hub token loaded from Kaggle secrets.")
except Exception:
    print("Kaggle secrets not available — ensure QAI_HUB_API_TOKEN is set.")

# ── Paths ────────────────────────────────────────────────────────────────────
# Adjust these to your dataset paths on Kaggle
TRAIN_JSON = "/kaggle/input/track3-annotations/annotations_train.json"
VAL_JSON   = "/kaggle/input/track3-annotations/annotations_val.json"
OUTPUT_DIR = "/kaggle/working/qwen2vl_lora_checkpoints"
MERGED_DIR = "/kaggle/working/qwen2vl_merged"

# ── Model ────────────────────────────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen2-VL-2B-Instruct"

# ── Training hyperparams ─────────────────────────────────────────────────────
LORA_RANK          = 64
LORA_ALPHA         = 128
LORA_DROPOUT       = 0.05
LEARNING_RATE      = 2e-4
NUM_EPOCHS         = 3
BATCH_SIZE         = 1       # per device — T4 memory limit
GRAD_ACCUM_STEPS   = 16      # effective batch = 16
MAX_SEQ_LEN        = 2048
WARMUP_RATIO       = 0.05
SAVE_STEPS         = 200
LOGGING_STEPS      = 10

# ── Inference config ─────────────────────────────────────────────────────────
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 512 * 28 * 28

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} — {torch.cuda.get_device_properties(i).total_memory // 1024**3} GB")

# %% [markdown]
# ## Cell 3 — Prompts

# %%
SYSTEM_PROMPT = (
    "You are a forensic image analyst specializing in detecting AI-generated content. "
    "You examine images with scientific precision, looking for subtle artifacts that differentiate "
    "AI-generated images from authentic photographs."
)

PROMPT_1 = (
    "Carefully examine this image and analyze it across these 8 forensic criteria. "
    "For each criterion, think step by step about what you observe.\n\n"
    "1. **Lighting & Shadows Consistency**: Are light sources consistent? Do shadows fall correctly? "
    "Look for impossible highlights, missing cast shadows, or inconsistent shadow directions.\n\n"
    "2. **Edges & Boundaries**: Examine object boundaries carefully. AI images often show "
    "unnatural smoothing, halos, or blending artifacts at edges between objects.\n\n"
    "3. **Texture & Resolution**: Is texture uniformly sharp or strangely uniform? "
    "AI images often have either over-smooth textures or repetitive texture patterns.\n\n"
    "4. **Perspective & Spatial Relationships**: Do objects maintain correct relative sizes? "
    "Are parallel lines converging correctly? Does depth look physically plausible?\n\n"
    "5. **Physical & Common Sense Logic**: Do all elements make physical sense together? "
    "Look for impossible object combinations or physically implausible configurations.\n\n"
    "6. **Text & Symbols**: Examine any text, logos, signs, numbers, or symbols closely. "
    "AI models frequently corrupt text into illegible or nonsensical characters.\n\n"
    "7. **Human & Biological Structure Integrity**: Check hands (finger count/shape), "
    "eyes (symmetry/reflections), ears, teeth, hair strand coherence, skin texture.\n\n"
    "8. **Material & Object Details**: Are material properties (glass, metal, fabric, skin) "
    "rendered with realistic light interaction? Look for plastic-looking skin or "
    "unrealistic material specularity.\n\n"
    "Analyze each criterion and provide your detailed forensic observations."
)

PROMPT_2 = (
    "Based on your forensic analysis above, provide your final assessment "
    "in the following exact JSON format. Use only the criterion names as specified.\n\n"
    '{\n'
    '  "overall_likelihood": "<AI-Generated or Real>",\n'
    '  "per_criterion": [\n'
    '    {"criterion": "Lighting & Shadows Consistency", "score": <0 or 1>, "evidence": "..."},\n'
    '    {"criterion": "Edges & Boundaries", "score": <0 or 1>, "evidence": "..."},\n'
    '    {"criterion": "Texture & Resolution", "score": <0 or 1>, "evidence": "..."},\n'
    '    {"criterion": "Perspective & Spatial Relationships", "score": <0 or 1>, "evidence": "..."},\n'
    '    {"criterion": "Physical & Common Sense Logic", "score": <0 or 1>, "evidence": "..."},\n'
    '    {"criterion": "Text & Symbols", "score": <0 or 1>, "evidence": "..."},\n'
    '    {"criterion": "Human & Biological Structure Integrity", "score": <0 or 1>, "evidence": "..."},\n'
    '    {"criterion": "Material & Object Details", "score": <0 or 1>, "evidence": "..."}\n'
    '  ]\n'
    '}\n\n'
    "Respond with ONLY the JSON. No markdown fences, no explanation."
)

CRITERIA = [
    "Lighting & Shadows Consistency",
    "Edges & Boundaries",
    "Texture & Resolution",
    "Perspective & Spatial Relationships",
    "Physical & Common Sense Logic",
    "Text & Symbols",
    "Human & Biological Structure Integrity",
    "Material & Object Details",
]

print("Prompts loaded.")

# %% [markdown]
# ## Cell 4 — Load Dataset

# %%
from PIL import Image
from torch.utils.data import Dataset as TorchDataset

with open(TRAIN_JSON) as f:
    train_records = json.load(f)
with open(VAL_JSON) as f:
    val_records = json.load(f)

print(f"Train samples: {len(train_records)}")
print(f"Val samples:   {len(val_records)}")

# Label balance check
for split_name, records in [("Train", train_records), ("Val", val_records)]:
    ai_n   = sum(1 for r in records if r["label"] == "AI-Generated")
    real_n = sum(1 for r in records if r["label"] == "Real")
    print(f"  {split_name}: AI={ai_n}, Real={real_n}")

# %% [markdown]
# ## Cell 5 — Load Model & Processor

# %%
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print(f"Loading {BASE_MODEL} with 4-bit quantization...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2" if torch.cuda.is_bf16_supported() else "eager",
)

processor = AutoProcessor.from_pretrained(
    BASE_MODEL,
    min_pixels=MIN_PIXELS,
    max_pixels=MAX_PIXELS,
)

print(f"Model loaded. Parameters: {model.num_parameters():,}")

# %% [markdown]
# ## Cell 6 — LoRA Configuration

# %%
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

# Prepare for k-bit training (important for stable 4-bit training)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # attention
        "gate_proj", "up_proj", "down_proj",       # MLP
    ],
    bias="none",
    use_rslora=True,  # rank-stabilized LoRA — better for high-rank training
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# %% [markdown]
# ## Cell 7 — Dataset & Data Collator

# %%
def build_messages(record: dict, image: Image.Image) -> list[dict]:
    """Build the full multi-turn conversation for one training sample."""
    reasoning_text = record.get("reasoning_text", "")
    target_json = json.dumps(record["output"], ensure_ascii=False)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT_1},
            ],
        },
        {"role": "assistant", "content": reasoning_text},
        {"role": "user", "content": PROMPT_2},
        {"role": "assistant", "content": target_json},
    ]


def find_assistant_response_spans(input_ids: list[int], tokenizer) -> list[tuple[int, int]]:
    """
    Return list of (start, end) index pairs marking assistant response tokens.
    These are the ONLY positions where loss is computed.
    """
    # Qwen2 chat template uses <|im_start|>assistant\n ... <|im_end|>
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id   = tokenizer.convert_tokens_to_ids("<|im_end|>")

    # "assistant" may tokenize as a single token or multiple; find it after im_start
    assistant_str = "assistant"
    assistant_tokens = tokenizer.encode(assistant_str, add_special_tokens=False)

    spans = []
    i = 0
    while i < len(input_ids) - len(assistant_tokens):
        if input_ids[i] == im_start_id:
            # Check if next tokens spell "assistant"
            if input_ids[i+1 : i+1+len(assistant_tokens)] == assistant_tokens:
                # Skip <|im_start|> assistant \n (3 tokens typically)
                start = i + 1 + len(assistant_tokens) + 1  # +1 for newline
                # Find closing im_end
                end = start
                while end < len(input_ids) and input_ids[end] != im_end_id:
                    end += 1
                if end < len(input_ids):
                    spans.append((start, end + 1))  # include im_end in loss
                i = end + 1
                continue
        i += 1
    return spans


class Track3Dataset(TorchDataset):
    def __init__(self, records: list[dict], processor, max_length: int = MAX_SEQ_LEN):
        self.records = records
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image = Image.open(record["image"]).convert("RGB")
        messages = build_messages(record, image)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = inputs["input_ids"][0].tolist()

        # Build labels: -100 everywhere except assistant responses
        labels = [-100] * len(input_ids)
        spans = find_assistant_response_spans(input_ids, self.processor.tokenizer)
        for start, end in spans:
            for j in range(start, min(end, len(input_ids))):
                labels[j] = input_ids[j]

        result = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": inputs["attention_mask"][0],
            "labels": torch.tensor(labels, dtype=torch.long),
        }

        if "pixel_values" in inputs:
            result["pixel_values"] = inputs["pixel_values"]
        if "image_grid_thw" in inputs:
            result["image_grid_thw"] = inputs["image_grid_thw"]

        return result


def collate_fn(batch: list[dict]) -> dict:
    """Pad sequences to max length in batch. Handle pixel_values concatenation."""
    max_len = max(item["input_ids"].shape[0] for item in batch)
    pad_id  = processor.tokenizer.pad_token_id or 0

    padded_input_ids      = []
    padded_attention_mask = []
    padded_labels         = []
    pixel_values_list     = []
    image_grid_thw_list   = []

    for item in batch:
        seq_len = item["input_ids"].shape[0]
        pad_len = max_len - seq_len

        padded_input_ids.append(
            torch.cat([item["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)])
        )
        padded_attention_mask.append(
            torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        )
        padded_labels.append(
            torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
        )

        if "pixel_values" in item:
            pixel_values_list.append(item["pixel_values"])
        if "image_grid_thw" in item:
            image_grid_thw_list.append(item["image_grid_thw"])

    result = {
        "input_ids":      torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_mask),
        "labels":         torch.stack(padded_labels),
    }

    if pixel_values_list:
        # Qwen2-VL expects concatenated pixel_values (not stacked)
        result["pixel_values"] = torch.cat(pixel_values_list, dim=0)
    if image_grid_thw_list:
        result["image_grid_thw"] = torch.cat(image_grid_thw_list, dim=0)

    return result


train_dataset = Track3Dataset(train_records, processor)
val_dataset   = Track3Dataset(val_records,   processor)

print(f"Train dataset: {len(train_dataset)} samples")
print(f"Val dataset:   {len(val_dataset)} samples")

# Quick sanity check on first sample
sample = train_dataset[0]
n_loss_tokens = (sample["labels"] != -100).sum().item()
print(f"First sample — total tokens: {len(sample['input_ids'])}, loss tokens: {n_loss_tokens}")
assert n_loss_tokens > 0, "No loss tokens found! Check find_assistant_response_spans."

# %% [markdown]
# ## Cell 8 — Training

# %%
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=WARMUP_RATIO,
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    logging_steps=LOGGING_STEPS,
    eval_strategy="steps",
    eval_steps=SAVE_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    dataloader_num_workers=2,
    remove_unused_columns=False,
    report_to="none",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="paged_adamw_8bit",  # memory-efficient optimizer for 4-bit training
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
)

print("Starting training...")
print(f"  Effective batch size: {BATCH_SIZE * GRAD_ACCUM_STEPS}")
print(f"  Steps per epoch: {len(train_dataset) // (BATCH_SIZE * GRAD_ACCUM_STEPS)}")

trainer.train()

# Save LoRA adapter
LORA_SAVE_DIR = "/kaggle/working/qwen2vl_lora_final"
model.save_pretrained(LORA_SAVE_DIR)
processor.save_pretrained(LORA_SAVE_DIR)
print(f"LoRA adapter saved → {LORA_SAVE_DIR}")

# %% [markdown]
# ## Cell 9 — Merge LoRA → Full Float Model (for AIMET)

# %%
from peft import PeftModel

print("Merging LoRA weights into base model (CPU merge)...")

# Reload base in float16, on CPU to avoid OOM
base_model_for_merge = Qwen2VLForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="cpu",
)

merged = PeftModel.from_pretrained(base_model_for_merge, LORA_SAVE_DIR)
merged = merged.merge_and_unload()

merged.save_pretrained(MERGED_DIR, safe_serialization=True)
processor.save_pretrained(MERGED_DIR)

print(f"Merged model saved → {MERGED_DIR}")
print("Next step: download this directory and run AIMET quantization (QPM tutorial).")

# %% [markdown]
# ## Cell 10 — Quick Validation (sample check)

# %%
# Load merged model briefly to verify it produces valid JSON
print("Running quick validation on 5 val samples...")

del merged  # free CPU memory
del base_model_for_merge
import gc; gc.collect()

# Load in 4-bit for quick inference check
val_model = Qwen2VLForConditionalGeneration.from_pretrained(
    MERGED_DIR,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
val_model.eval()

val_processor = AutoProcessor.from_pretrained(
    MERGED_DIR,
    min_pixels=MIN_PIXELS,
    max_pixels=MAX_PIXELS,
)

correct = 0
valid_json = 0
for sample in val_records[:5]:
    image = Image.open(sample["image"]).convert("RGB")

    # Step 1
    msgs_1 = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT_1}]},
    ]
    t1 = val_processor.apply_chat_template(msgs_1, tokenize=False, add_generation_prompt=True)
    i1 = val_processor(text=[t1], images=[image], return_tensors="pt").to("cuda")
    with torch.no_grad():
        o1 = val_model.generate(**i1, max_new_tokens=600, temperature=0.1, do_sample=True)
    reasoning = val_processor.decode(o1[0][i1["input_ids"].shape[1]:], skip_special_tokens=True)

    # Step 2
    msgs_2 = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT_1}]},
        {"role": "assistant", "content": reasoning},
        {"role": "user", "content": PROMPT_2},
    ]
    t2 = val_processor.apply_chat_template(msgs_2, tokenize=False, add_generation_prompt=True)
    i2 = val_processor(text=[t2], images=[image], return_tensors="pt").to("cuda")
    with torch.no_grad():
        o2 = val_model.generate(**i2, max_new_tokens=512, temperature=0.0, do_sample=False)
    raw = val_processor.decode(o2[0][i2["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    # Parse
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()

    try:
        out = json.loads(raw)
        valid_json += 1
        pred = out.get("overall_likelihood", "?")
        if pred == sample["label"]:
            correct += 1
        print(f"  pred={pred:15s} true={sample['label']:15s} {'✓' if pred == sample['label'] else '✗'}")
    except json.JSONDecodeError:
        print(f"  [INVALID JSON] {raw[:80]}")

print(f"\nQuick check: {correct}/5 correct, {valid_json}/5 valid JSON")
print("\nNotebook complete. Download /kaggle/working/qwen2vl_merged/ for AIMET.")
