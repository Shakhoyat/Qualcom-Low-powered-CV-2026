# %% [markdown]
# # Track 2 Pipeline 1 — R(2+1)D-18 Training (88.94% val acc)
#
# **Kaggle settings:**
# - Accelerator: GPU T4 x2
# - Internet: ON
# - Input dataset: your preprocessed QEVD tensors (manifest.jsonl + class_mapping.json)
# - Secret: QAI_HUB_API_TOKEN
#
# **Architecture:** R(2+1)D-18 pretrained on Kinetics-400, fine-tuned on QEVD
# **Input:** NCDHW `(1, 3, 16, 112, 112)` — standard PyTorch layout
# **Head:** Dropout(0.3) + Linear(512 → 91)
# **Training:** 11 epochs, fc+layer4 only (simple freeze strategy)
#
# **Outputs:**
# - `/kaggle/working/best_r2plus1d_qevd.pth`
# - `/kaggle/working/lpcvc_final_unified.onnx`

# %% [markdown]
# ## Cell 1 — Installs

# %%
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "torch", "torchvision", "onnx", "onnxruntime", "qai-hub",
    "scikit-learn", "tqdm", "matplotlib"], check=True)
print("Deps installed.")

# %% [markdown]
# ## Cell 2 — Config

# %%
import os, json, time, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm import tqdm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

try:
    from kaggle_secrets import UserSecretsClient
    os.environ["QAI_HUB_API_TOKEN"] = UserSecretsClient().get_secret("QAI_HUB_API_TOKEN")
    print("AI Hub token loaded.")
except Exception:
    print("Set QAI_HUB_API_TOKEN manually if running locally.")

# ── Paths — update TENSORS_ROOT to your Kaggle input slug ──────────────────
TENSORS_ROOT   = "/kaggle/input/YOUR-DATASET-SLUG/preprocessed_tensors"
MANIFEST_PATH  = os.path.join(TENSORS_ROOT, "manifest.jsonl")
CLASS_MAP_PATH = os.path.join(TENSORS_ROOT, "class_map.json")
CLASS_LBL_PATH = os.path.join(TENSORS_ROOT, "class_labels.json")
SAVE_DIR       = "/kaggle/working"
BEST_PATH      = os.path.join(SAVE_DIR, "best_r2plus1d_qevd.pth")
ONNX_PATH      = os.path.join(SAVE_DIR, "lpcvc_final_unified.onnx")

# ── Hyperparams ──────────────────────────────────────────────────────────────
NUM_CLASSES      = 91
NUM_FRAMES       = 16
FRAME_SIZE       = 112
EPOCHS           = 11
BATCH_SIZE       = 8
INITIAL_LR       = 3e-4
WEIGHT_DECAY     = 1e-2
LABEL_SMOOTHING  = 0.1
GRAD_CLIP        = 1.0
NUM_WORKERS      = 4
LATENCY_LIMIT_MS = 34.0
AIHUB_DEVICE     = "Dragonwing IQ-9075 EVK"

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"
print(f"Device: {device} | AMP: {use_amp}")

# %% [markdown]
# ## Cell 3 — Load Manifest + Class Map

# %%
with open(CLASS_LBL_PATH) as f:
    class_labels = json.load(f)
with open(CLASS_MAP_PATH) as f:
    class_map = json.load(f)

NUM_CLASSES = len(class_labels)
print(f"Classes: {NUM_CLASSES}")

MARKER = "preprocessed_tensors"

def remap(old_path):
    old_path = old_path.replace("\\", "/")
    idx = old_path.find(MARKER)
    if idx == -1:
        return old_path
    return TENSORS_ROOT.rstrip("/") + old_path[idx + len(MARKER):]

all_entries = []
with open(MANIFEST_PATH) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        e = json.loads(line)
        if e["label"] not in class_map:
            continue
        e["tensor_path"] = remap(e["tensor_path"])
        all_entries.append(e)

train_entries = [e for e in all_entries if e.get("split") == "train"]
val_entries   = [e for e in all_entries if e.get("split") == "val"]

if not val_entries:
    print("No split in manifest — using 85/15 random split")
    counts = Counter(e["label"] for e in all_entries)
    entries_aug, labels_aug = [], []
    for e in all_entries:
        if counts[e["label"]] == 1:
            entries_aug.extend([e, e])
            labels_aug.extend([e["label"], e["label"]])
        else:
            entries_aug.append(e)
            labels_aug.append(e["label"])
    train_entries, val_entries = train_test_split(
        entries_aug, test_size=0.15, random_state=42, stratify=labels_aug
    )

print(f"Train: {len(train_entries)} | Val: {len(val_entries)}")

# %% [markdown]
# ## Cell 4 — Dataset (NCDHW)

# %%
class QEVDDataset(Dataset):
    MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
    STD  = torch.tensor([0.22803, 0.22145,  0.216989]).view(3, 1, 1, 1)

    def __init__(self, entries, class_map, augment=False):
        self.entries   = entries
        self.class_map = class_map
        self.augment   = augment

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e   = self.entries[idx]
        arr = np.load(e["tensor_path"])        # (1, 16, 112, 112, 3) NDHWC
        clip = torch.from_numpy(arr).squeeze(0)  # (16, 112, 112, 3)
        clip = clip.permute(3, 0, 1, 2)          # (3, 16, 112, 112) NCTHW
        clip = (clip - self.MEAN) / self.STD

        if self.augment:
            if torch.rand(1).item() < 0.5:
                clip = torch.flip(clip, [-1])
            if torch.rand(1).item() < 0.3:
                s = torch.randint(-2, 3, (1,)).item()
                if s:
                    clip = torch.roll(clip, shifts=s, dims=1)

        return clip, self.class_map[e["label"]]

# %% [markdown]
# ## Cell 5 — Dataloaders

# %%
train_ds = QEVDDataset(train_entries, class_map, augment=True)
val_ds   = QEVDDataset(val_entries,   class_map, augment=False)

train_labels   = [class_map[e["label"]] for e in train_entries]
class_counts   = np.bincount(train_labels, minlength=NUM_CLASSES)
class_weights  = 1.0 / np.clip(class_counts, 1, None)
sample_weights = [class_weights[l] for l in train_labels]
sampler = WeightedRandomSampler(torch.DoubleTensor(sample_weights), len(sample_weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
                          persistent_workers=(NUM_WORKERS > 0))
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=(NUM_WORKERS > 0))
print(f"Train: {len(train_loader)} batches | Val: {len(val_loader)} batches")

# %% [markdown]
# ## Cell 6 — Build Model (NCDHW head with Dropout)

# %%
print("Building R(2+1)D-18 with Kinetics-400 weights...")
model = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
model.fc = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(model.fc.in_features, NUM_CLASSES),
)
print(f"FC replaced: 512 → Dropout(0.3) → {NUM_CLASSES}")

# Freeze all, unfreeze fc + layer4
for p in model.parameters():
    p.requires_grad = False
for p in model.fc.parameters():
    p.requires_grad = True
for p in model.layer4.parameters():
    p.requires_grad = True

n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {n_train:,} / {n_total:,} ({100*n_train/n_total:.1f}%)")
model = model.to(device)

# %% [markdown]
# ## Cell 7 — Training Loop

# %%
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=INITIAL_LR, weight_decay=WEIGHT_DECAY
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
scaler    = torch.amp.GradScaler("cuda") if use_amp else None

best_val_acc = 0.0
history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

print(f"Training {EPOCHS} epochs | batch={BATCH_SIZE} | AMP={use_amp}")
print("=" * 70)

for epoch in range(EPOCHS):
    t0 = time.time()
    model.train()
    run_loss = run_correct = run_total = 0

    for x, y in tqdm(train_loader, desc=f"E{epoch+1:02d} train", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast("cuda"):
                out  = model(x)
                loss = criterion(out, y)
            if torch.isnan(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            out  = model(x)
            loss = criterion(out, y)
            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
        run_loss    += loss.item() * x.size(0)
        run_correct += out.argmax(1).eq(y).sum().item()
        run_total   += x.size(0)

    scheduler.step()
    tr_loss = run_loss / max(run_total, 1)
    tr_acc  = 100.0 * run_correct / max(run_total, 1)

    model.eval()
    vl_loss = vl_correct = vl_total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    out  = model(x)
                    loss = criterion(out, y)
            else:
                out  = model(x)
                loss = criterion(out, y)
            if not torch.isnan(loss):
                vl_loss += loss.item() * x.size(0)
            vl_correct += out.argmax(1).eq(y).sum().item()
            vl_total   += y.size(0)

    vl_loss = vl_loss / max(vl_total, 1)
    vl_acc  = 100.0 * vl_correct / max(vl_total, 1)

    history["train_acc"].append(tr_acc)
    history["val_acc"].append(vl_acc)
    history["train_loss"].append(tr_loss)
    history["val_loss"].append(vl_loss)

    flag = ""
    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        torch.save({
            "model_state": model.state_dict(),
            "epoch": epoch,
            "val_acc": vl_acc,
            "num_classes": NUM_CLASSES,
            "class_labels": class_labels,
            "class_map": class_map,
        }, BEST_PATH)
        flag = "  BEST"

    print(f"E{epoch+1:02d}/{EPOCHS} | Train {tr_acc:.2f}% loss={tr_loss:.4f} | "
          f"Val {vl_acc:.2f}% loss={vl_loss:.4f} | "
          f"LR={optimizer.param_groups[0]['lr']:.1e} | {time.time()-t0:.0f}s{flag}")

print(f"\nBest val acc: {best_val_acc:.2f}%")

# %% [markdown]
# ## Cell 8 — Export ONNX (NCDHW, single self-contained file)

# %%
import onnx
import onnxruntime as ort

print("Rebuilding model from best checkpoint for export...")
export_model = r2plus1d_18(weights=None)
export_model.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(export_model.fc.in_features, NUM_CLASSES))

ckpt = torch.load(BEST_PATH, map_location="cpu", weights_only=False)
state_dict = ckpt.get("model_state", ckpt)
state_dict = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
              for k, v in state_dict.items()}
export_model.load_state_dict(state_dict, strict=True)
export_model.eval()

dummy = torch.randn(1, 3, NUM_FRAMES, FRAME_SIZE, FRAME_SIZE)
print(f"Export input shape: {dummy.shape}  (NCDHW)")

torch.onnx.export(
    export_model,
    dummy,
    ONNX_PATH,
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes=None,
    dynamo=False,
)

# Inline any external data
m = onnx.load(ONNX_PATH, load_external_data=True)
onnx.save(m, ONNX_PATH, save_as_external_data=False)
onnx.checker.check_model(ONNX_PATH)

size_mb = os.path.getsize(ONNX_PATH) / 1024 / 1024
print(f"ONNX saved: {ONNX_PATH} ({size_mb:.1f} MB)")

# Parity check
sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
with torch.no_grad():
    torch_out = export_model(dummy).numpy()
ort_out = sess.run(None, {"input": dummy.numpy()})[0]
max_diff = float(np.abs(torch_out - ort_out).max())
print(f"Max PyTorch vs ONNX diff: {max_diff:.6f}  ({'OK' if max_diff < 1e-3 else 'CHECK THIS'})")

# %% [markdown]
# ## Cell 9 — Save Training Curves

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
ep = range(1, len(history["train_acc"]) + 1)
ax1.plot(ep, history["train_acc"], "b-o", ms=4, label="Train")
ax1.plot(ep, history["val_acc"],   "r-o", ms=4, label="Val")
ax1.axhline(90, color="g", ls="--", alpha=0.6, label="90% target")
ax1.set(xlabel="Epoch", ylabel="Accuracy (%)", title="Accuracy")
ax1.legend(); ax1.grid(alpha=0.3)
ax2.plot(ep, history["train_loss"], "b-o", ms=4, label="Train")
ax2.plot(ep, history["val_loss"],   "r-o", ms=4, label="Val")
ax2.set(xlabel="Epoch", ylabel="Loss", title="Loss")
ax2.legend(); ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "training_curves.png"), dpi=150)
print("Training curves saved.")

print("\n=== Files to download from Kaggle ===")
print(f"  {BEST_PATH}")
print(f"  {ONNX_PATH}")
print("Then run: python aihub_deploy.py")
