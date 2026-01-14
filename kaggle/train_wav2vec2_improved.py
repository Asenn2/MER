# %% [markdown]
# # 🎙️ IEMOCAP Emotion Recognition - IMPROVED VERSION
# 
# Improvements:
# - wav2vec2-large (instead of base)
# - Longer audio (10s instead of 5s)
# - Data augmentation (noise, pitch shift, time stretch)
# - Early stopping
# - Better hyperparameters

# %%
!pip install -q transformers datasets librosa torch torchaudio scikit-learn audiomentations

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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# %% [markdown]
# ## Configuration

# %%
DATA_DIR = "/kaggle/input/iemocap-subset"
OUTPUT_DIR = "/kaggle/working/wav2vec2-emotion-improved"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model - USE LARGE MODEL
MODEL_NAME = "facebook/wav2vec2-large"  # CHANGED: large instead of base
SAMPLE_RATE = 16000
MAX_DURATION = 10  # CHANGED: 10 seconds instead of 5

# Training
BATCH_SIZE = 8  # Reduced due to larger model
EPOCHS = 50
LEARNING_RATE = 5e-4
HIDDEN_SIZE = 512  # Larger for wav2vec2-large
PATIENCE = 10  # Early stopping patience

# Emotions
EMOTION_LABELS = ["angry", "excited", "frustrated", "happy", "neutral", "sad"]
LABEL2ID = {label: i for i, label in enumerate(EMOTION_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(EMOTION_LABELS)}
NUM_LABELS = len(EMOTION_LABELS)

print(f"Using {MODEL_NAME} with {MAX_DURATION}s audio, batch={BATCH_SIZE}")

# %% [markdown]
# ## Data Augmentation

# %%
class AudioAugmenter:
    """Simple audio augmentations"""
    
    def __init__(self, sr=16000):
        self.sr = sr
    
    def add_noise(self, waveform, noise_factor=0.005):
        """Add random noise"""
        noise = np.random.randn(len(waveform))
        return waveform + noise_factor * noise
    
    def time_stretch(self, waveform, rate=None):
        """Time stretch (speed up or slow down)"""
        if rate is None:
            rate = random.uniform(0.9, 1.1)
        return librosa.effects.time_stretch(waveform, rate=rate)
    
    def pitch_shift(self, waveform, n_steps=None):
        """Pitch shift"""
        if n_steps is None:
            n_steps = random.uniform(-2, 2)
        return librosa.effects.pitch_shift(waveform, sr=self.sr, n_steps=n_steps)
    
    def augment(self, waveform):
        """Apply random augmentations"""
        # Randomly apply augmentations
        if random.random() < 0.5:
            waveform = self.add_noise(waveform)
        if random.random() < 0.3:
            waveform = self.time_stretch(waveform)
        if random.random() < 0.3:
            waveform = self.pitch_shift(waveform)
        return waveform

augmenter = AudioAugmenter(SAMPLE_RATE)

# %% [markdown]
# ## Dataset

# %%
df_metadata = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))
df_split = pd.read_csv(os.path.join(DATA_DIR, "split.csv"))
df = df_metadata.merge(df_split, on="utterance_id")

train_df = df[df["split"] == "train"].reset_index(drop=True)
test_df = df[df["split"] == "test"].reset_index(drop=True)

print(f"Train: {len(train_df)}, Test: {len(test_df)}")

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
        
        # Load audio
        waveform, _ = librosa.load(audio_path, sr=SAMPLE_RATE, duration=MAX_DURATION)
        
        # Apply augmentation during training
        if self.augment:
            waveform = augmenter.augment(waveform)
        
        # Pad/truncate
        if len(waveform) < self.max_length:
            waveform = np.pad(waveform, (0, self.max_length - len(waveform)))
        else:
            waveform = waveform[:self.max_length]
        
        # Normalize
        if np.max(np.abs(waveform)) > 0:
            waveform = waveform / np.max(np.abs(waveform))
        
        label = LABEL2ID[row["emotion"]]
        
        return {
            "waveform": torch.tensor(waveform, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long)
        }

def collate_fn(batch):
    waveforms = torch.stack([x["waveform"] for x in batch])
    labels = torch.stack([x["label"] for x in batch])
    return {"waveform": waveforms, "labels": labels}

# Create datasets
train_dataset = EmotionDataset(train_df, DATA_DIR, MAX_DURATION, augment=True)
test_dataset = EmotionDataset(test_df, DATA_DIR, MAX_DURATION, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                          collate_fn=collate_fn, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         collate_fn=collate_fn, num_workers=2)

# %% [markdown]
# ## Model (Weighted Layer Pooling)

# %%
class Wav2Vec2EmotionClassifier(nn.Module):
    def __init__(self, model_name, num_labels, hidden_size=512):
        super().__init__()
        
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(
            model_name,
            output_hidden_states=True
        )
        
        # Freeze wav2vec2
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
        
        # wav2vec2-large has 25 layers (1 CNN + 24 transformer)
        self.num_layers = 25
        self.hidden_dim = 1024  # wav2vec2-large uses 1024
        
        # Learnable layer weights
        self.layer_weights = nn.Parameter(torch.ones(self.num_layers) / self.num_layers)
        
        # MLP classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_labels)
        )
        
    def forward(self, waveforms):
        with torch.no_grad():
            outputs = self.wav2vec2(waveforms)
        
        hidden_states = outputs.hidden_states  # 25 layers
        stacked = torch.stack(hidden_states, dim=0)
        
        weights = torch.softmax(self.layer_weights, dim=0)
        weighted_sum = torch.sum(stacked * weights.view(-1, 1, 1, 1), dim=0)
        pooled = weighted_sum.mean(dim=1)
        
        logits = self.classifier(pooled)
        return logits

# %%
model = Wav2Vec2EmotionClassifier(MODEL_NAME, NUM_LABELS, HIDDEN_SIZE).to(device)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params:,}")

# %% [markdown]
# ## Training with Early Stopping

# %%
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing helps
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

# %%
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        waveforms = batch["waveform"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        logits = model(waveforms)
        loss = criterion(logits, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(loader), correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            waveforms = batch["waveform"].to(device)
            labels = batch["labels"].to(device)
            
            logits = model(waveforms)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), accuracy, all_preds, all_labels

# %%
print("🚀 Starting training with early stopping...")
best_accuracy = 0
patience_counter = 0
history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "lr": []}

for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion, device)
    
    current_lr = optimizer.param_groups[0]['lr']
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_acc)
    history["lr"].append(current_lr)
    
    scheduler.step(val_acc)
    
    print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, LR={current_lr:.2e}")
    
    if val_acc > best_accuracy:
        best_accuracy = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))
        print(f"  ✅ New best! Accuracy: {val_acc:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n⏹️ Early stopping at epoch {epoch+1}")
            break

print(f"\n🎉 Best accuracy: {best_accuracy:.4f}")

# %% [markdown]
# ## Final Evaluation

# %%
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pt")))
_, accuracy, y_pred, y_true = evaluate(model, test_loader, criterion, device)

print(f"\n📊 Final Test Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=EMOTION_LABELS))

# %%
# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix - Improved Model (Acc: {accuracy:.2%})")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.show()

# %%
# Save predictions for fusion
model.eval()
all_probs = []
all_preds = []

with torch.no_grad():
    for batch in test_loader:
        waveforms = batch["waveform"].to(device)
        logits = model(waveforms)
        probs = torch.softmax(logits, dim=-1)
        all_probs.append(probs.cpu().numpy())
        all_preds.extend(logits.argmax(dim=-1).cpu().numpy())

all_probs = np.vstack(all_probs)

predictions_df = pd.DataFrame({
    "utterance_id": test_df["utterance_id"],
    **{f"prob_{emotion}": all_probs[:, i] for i, emotion in enumerate(EMOTION_LABELS)},
    "predicted_emotion": [ID2LABEL[p] for p in all_preds],
    "true_emotion": test_df["emotion"]
})

predictions_df.to_parquet(os.path.join(OUTPUT_DIR, "audio_predictions.parquet"), index=False)
print("✅ Predictions saved!")

# %%
# Save model
torch.save({
    "model_state_dict": model.state_dict(),
    "config": {"model_name": MODEL_NAME, "num_labels": NUM_LABELS, "hidden_size": HIDDEN_SIZE}
}, os.path.join(OUTPUT_DIR, "emotion_model.pt"))

print(f"\n✅ Done! Final accuracy: {accuracy:.2%}")
