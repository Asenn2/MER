# %% [markdown]
# # 🎙️ IEMOCAP - Version Finale Optimisée
# 
# Target: 60% accuracy (State of the Art)
# - wav2vec2-base (comme le papier)
# - Audio 10 secondes
# - Data augmentation légère
# - Weighted layer pooling
# - Early stopping + LR scheduling

# %%
!pip install -q transformers datasets librosa torch torchaudio scikit-learn

# %%
import os
import torch
import torch.nn as nn
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings('ignore')

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# %% [markdown]
# ## Configuration Optimisée

# %%
DATA_DIR = "/kaggle/input/iemocap-subset"
OUTPUT_DIR = "/kaggle/working/wav2vec2-final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model - BASE comme le papier
MODEL_NAME = "facebook/wav2vec2-base"
SAMPLE_RATE = 16000
MAX_DURATION = 10  # 10 secondes

# Training optimisé
BATCH_SIZE = 16
EPOCHS = 100  # Beaucoup d'epochs avec early stopping
LEARNING_RATE = 3e-4  # Optimisé pour base
WEIGHT_DECAY = 0.01
PATIENCE = 15  # Early stopping
HIDDEN_SIZE = 256

# Augmentation - légère
AUGMENT_PROB = 0.3  # 30% de chance d'augmenter

EMOTION_LABELS = ["angry", "excited", "frustrated", "happy", "neutral", "sad"]
LABEL2ID = {label: i for i, label in enumerate(EMOTION_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(EMOTION_LABELS)}
NUM_LABELS = len(EMOTION_LABELS)

# %% [markdown]
# ## Data Augmentation Légère

# %%
def augment_audio(waveform, sr=16000):
    """Augmentation légère - pas trop aggressive"""
    augmented = waveform.copy()
    
    # 1. Add slight noise (very subtle)
    if random.random() < 0.3:
        noise = np.random.randn(len(augmented)) * 0.002
        augmented = augmented + noise
    
    # 2. Volume change
    if random.random() < 0.3:
        volume_factor = random.uniform(0.8, 1.2)
        augmented = augmented * volume_factor
    
    # 3. Time shift (small)
    if random.random() < 0.2:
        shift = int(sr * random.uniform(-0.1, 0.1))  # max 100ms shift
        augmented = np.roll(augmented, shift)
    
    return augmented

# %% [markdown]
# ## Load Data

# %%
df_metadata = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))
df_split = pd.read_csv(os.path.join(DATA_DIR, "split.csv"))
df = df_metadata.merge(df_split, on="utterance_id")

train_df = df[df["split"] == "train"].reset_index(drop=True)
test_df = df[df["split"] == "test"].reset_index(drop=True)

print(f"Train: {len(train_df)}, Test: {len(test_df)}")
print(f"Emotions distribution:\n{train_df['emotion'].value_counts()}")

# %% [markdown]
# ## Dataset

# %%
class EmotionDataset(Dataset):
    def __init__(self, df, data_dir, max_duration=10, augment=False):
        self.df = df
        self.data_dir = data_dir
        self.max_length = SAMPLE_RATE * max_duration
        self.augment = augment
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = os.path.join(self.data_dir, row["audio_path"])
        
        # Load audio - full duration
        waveform, _ = librosa.load(audio_path, sr=SAMPLE_RATE, duration=MAX_DURATION)
        
        # Augment if training
        if self.augment and random.random() < AUGMENT_PROB:
            waveform = augment_audio(waveform, SAMPLE_RATE)
        
        # Pad/truncate to fixed length
        if len(waveform) < self.max_length:
            # Pad with zeros
            waveform = np.pad(waveform, (0, self.max_length - len(waveform)))
        else:
            waveform = waveform[:self.max_length]
        
        # Normalize
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val
        
        label = LABEL2ID[row["emotion"]]
        
        return {
            "waveform": torch.tensor(waveform, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long)
        }

def collate_fn(batch):
    waveforms = torch.stack([x["waveform"] for x in batch])
    labels = torch.stack([x["label"] for x in batch])
    return {"waveform": waveforms, "labels": labels}

# %%
train_dataset = EmotionDataset(train_df, DATA_DIR, MAX_DURATION, augment=True)
test_dataset = EmotionDataset(test_df, DATA_DIR, MAX_DURATION, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                          collate_fn=collate_fn, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         collate_fn=collate_fn, num_workers=2, pin_memory=True)

print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

# %% [markdown]
# ## Model - Weighted Layer Pooling (comme le papier)

# %%
class Wav2Vec2EmotionClassifier(nn.Module):
    """
    Approche du papier:
    - Weighted sum des 13 couches de wav2vec2-base
    - Mean pooling sur le temps
    - MLP classifier
    """
    def __init__(self, model_name, num_labels, hidden_size=256):
        super().__init__()
        
        # Load wav2vec2 frozen
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(
            model_name,
            output_hidden_states=True
        )
        
        # FREEZE wav2vec2
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
        
        # wav2vec2-base: 13 layers (1 CNN + 12 transformer), 768 dim
        self.num_layers = 13
        self.hidden_dim = 768
        
        # Learnable layer weights (key innovation from paper)
        self.layer_weights = nn.Parameter(torch.zeros(self.num_layers))
        
        # Classifier with LayerNorm for stability
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),  # GELU often better than ReLU
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_labels)
        )
        
    def forward(self, waveforms):
        # Get all hidden states (frozen, no grad)
        with torch.no_grad():
            outputs = self.wav2vec2(waveforms)
        
        # Stack all layers: (num_layers, batch, seq, hidden)
        hidden_states = outputs.hidden_states
        stacked = torch.stack(hidden_states, dim=0)
        
        # Softmax weights
        weights = torch.softmax(self.layer_weights, dim=0)
        
        # Weighted sum
        weighted = torch.sum(stacked * weights.view(-1, 1, 1, 1), dim=0)
        
        # Mean pooling over time
        pooled = weighted.mean(dim=1)  # (batch, hidden)
        
        # Classify
        logits = self.classifier(pooled)
        
        return logits

# %%
model = Wav2Vec2EmotionClassifier(MODEL_NAME, NUM_LABELS, HIDDEN_SIZE).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total_params:,}")
print(f"Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

# %% [markdown]
# ## Training Setup

# %%
# Loss with label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=LEARNING_RATE, 
    weight_decay=WEIGHT_DECAY
)

# Scheduler - reduce LR on plateau
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
)

# %%
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        waveforms = batch["waveform"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        logits = model(waveforms)
        loss = criterion(logits, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({"loss": f"{loss.item():.3f}", "acc": f"{correct/total:.3f}"})
    
    return total_loss / len(loader), correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            waveforms = batch["waveform"].to(device)
            labels = batch["labels"].to(device)
            
            logits = model(waveforms)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=-1)
            
            total_loss += loss.item()
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    all_probs = np.vstack(all_probs)
    return total_loss / len(loader), accuracy, all_preds, all_labels, all_probs

# %% [markdown]
# ## Training Loop

# %%
print("=" * 60)
print("🚀 Starting training - Target: 60% accuracy")
print("=" * 60)

best_accuracy = 0
patience_counter = 0
history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "lr": []}

for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, _, _, _ = evaluate(model, test_loader, criterion, device)
    
    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step(val_acc)
    
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_acc)
    history["lr"].append(current_lr)
    
    print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val Acc: {val_acc:.4f} | "
          f"LR: {current_lr:.2e}")
    
    # Save best
    if val_acc > best_accuracy:
        best_accuracy = val_acc
        patience_counter = 0
        torch.save({
            'model_state_dict': model.state_dict(),
            'accuracy': val_acc,
            'epoch': epoch
        }, os.path.join(OUTPUT_DIR, "best_model.pt"))
        print(f"         ✅ New best model! Accuracy: {val_acc:.4f}")
    else:
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= PATIENCE:
        print(f"\n⏹️ Early stopping at epoch {epoch+1}")
        break
    
    # Target reached?
    if val_acc >= 0.60:
        print(f"\n🎯 Target 60% reached!")

print(f"\n{'='*60}")
print(f"🏆 Best accuracy: {best_accuracy:.4f} ({best_accuracy*100:.1f}%)")
print(f"{'='*60}")

# %% [markdown]
# ## Evaluation

# %%
# Load best model
checkpoint = torch.load(os.path.join(OUTPUT_DIR, "best_model.pt"))
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded best model from epoch {checkpoint['epoch']+1}")

# Final evaluation
_, accuracy, y_pred, y_true, all_probs = evaluate(model, test_loader, criterion, device)

print(f"\n📊 Final Test Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=EMOTION_LABELS, digits=3))

# %%
# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix - Final Model (Accuracy: {accuracy:.1%})")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
plt.show()

# %%
# Training curves
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(history["train_loss"], label="Train")
axes[0].plot(history["val_loss"], label="Val")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Loss")
axes[0].legend()

axes[1].plot(history["val_accuracy"])
axes[1].axhline(y=0.6, color='r', linestyle='--', label='Target 60%')
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Validation Accuracy")
axes[1].legend()

axes[2].plot(history["lr"])
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Learning Rate")
axes[2].set_title("Learning Rate Schedule")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150)
plt.show()

# %%
# Layer weights analysis
weights = torch.softmax(model.layer_weights, dim=0).detach().cpu().numpy()
plt.figure(figsize=(10, 4))
bars = plt.bar(range(len(weights)), weights, color='steelblue')
plt.xlabel("Layer")
plt.ylabel("Weight")
plt.title("Learned Layer Importance")
layer_names = ["CNN"] + [f"T{i}" for i in range(1, 13)]
plt.xticks(range(len(weights)), layer_names)

# Highlight top 3 layers
top3 = np.argsort(weights)[-3:]
for i in top3:
    bars[i].set_color('orange')
plt.savefig(os.path.join(OUTPUT_DIR, "layer_weights.png"), dpi=150)
plt.show()

print("\nTop 3 most important layers:")
for i in sorted(top3, key=lambda x: weights[x], reverse=True):
    print(f"  Layer {i} ({layer_names[i]}): {weights[i]:.4f}")

# %% [markdown]
# ## Save Predictions for Fusion

# %%
predictions_df = pd.DataFrame({
    "utterance_id": test_df["utterance_id"],
    **{f"prob_{emotion}": all_probs[:, i] for i, emotion in enumerate(EMOTION_LABELS)},
    "predicted_emotion": [ID2LABEL[p] for p in y_pred],
    "true_emotion": test_df["emotion"]
})

predictions_df.to_parquet(os.path.join(OUTPUT_DIR, "audio_predictions.parquet"))
predictions_df.to_csv(os.path.join(OUTPUT_DIR, "audio_predictions.csv"), index=False)

print("✅ Predictions saved!")
print(predictions_df.head())

# %%
# Save complete model
torch.save({
    'model_state_dict': model.state_dict(),
    'layer_weights': model.layer_weights.data,
    'accuracy': accuracy,
    'config': {
        'model_name': MODEL_NAME,
        'num_labels': NUM_LABELS,
        'hidden_size': HIDDEN_SIZE,
        'label2id': LABEL2ID,
        'id2label': ID2LABEL
    }
}, os.path.join(OUTPUT_DIR, "emotion_model_complete.pt"))

# %%
print("\n" + "="*60)
print("📁 Output files:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    path = os.path.join(OUTPUT_DIR, f)
    if os.path.isfile(path):
        size = os.path.getsize(path) / 1024 / 1024
        print(f"  {f}: {size:.2f} MB")
print("="*60)
print(f"\n🎉 Done! Final accuracy: {accuracy:.1%}")
print("Download 'audio_predictions.parquet' for fusion with your teammates!")
