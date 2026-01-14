# %% [markdown]
# # 🎙️ IEMOCAP Emotion Recognition - Paper Approach
# 
# Using Wav2Vec2 embeddings with **Weighted Layer Pooling + MLP**
# Based on: "Emotion Recognition from Speech Using Wav2vec 2.0 Embeddings" (Pepino et al., 2021)

# %% [markdown]
# ## 1. Setup

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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# %% [markdown]
# ## 2. Configuration

# %%
# Paths - MODIFY FOR YOUR KAGGLE DATASET
DATA_DIR = "/kaggle/input/iemocap-subset"
OUTPUT_DIR = "/kaggle/working/wav2vec2-emotion-paper"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model
MODEL_NAME = "facebook/wav2vec2-base"  # Base model for embeddings
SAMPLE_RATE = 16000
MAX_DURATION = 5

# Training
BATCH_SIZE = 16  # Can be larger since we're not fine-tuning wav2vec2
EPOCHS = 30  # More epochs since training is fast
LEARNING_RATE = 1e-3  # Higher LR for MLP
HIDDEN_SIZE = 256

# Emotions
EMOTION_LABELS = ["angry", "excited", "frustrated", "happy", "neutral", "sad"]
LABEL2ID = {label: i for i, label in enumerate(EMOTION_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(EMOTION_LABELS)}
NUM_LABELS = len(EMOTION_LABELS)

print(f"Config: {EPOCHS} epochs, batch={BATCH_SIZE}, lr={LEARNING_RATE}")

# %% [markdown]
# ## 3. Load Data

# %%
df_metadata = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))
df_split = pd.read_csv(os.path.join(DATA_DIR, "split.csv"))
df = df_metadata.merge(df_split, on="utterance_id")

train_df = df[df["split"] == "train"].reset_index(drop=True)
test_df = df[df["split"] == "test"].reset_index(drop=True)

print(f"Train: {len(train_df)}, Test: {len(test_df)}")

# %% [markdown]
# ## 4. Dataset Class

# %%
class EmotionDataset(Dataset):
    def __init__(self, df, data_dir, feature_extractor, max_duration=5):
        self.df = df
        self.data_dir = data_dir
        self.feature_extractor = feature_extractor
        self.max_length = SAMPLE_RATE * max_duration
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = os.path.join(self.data_dir, row["audio_path"])
        
        # Load audio
        waveform, _ = librosa.load(audio_path, sr=SAMPLE_RATE, duration=MAX_DURATION)
        
        # Pad/truncate
        if len(waveform) < self.max_length:
            waveform = np.pad(waveform, (0, self.max_length - len(waveform)))
        else:
            waveform = waveform[:self.max_length]
        
        # Get label
        label = LABEL2ID[row["emotion"]]
        
        return {
            "waveform": torch.tensor(waveform, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long)
        }

def collate_fn(batch):
    waveforms = torch.stack([x["waveform"] for x in batch])
    labels = torch.stack([x["label"] for x in batch])
    return {"waveform": waveforms, "labels": labels}

# %% [markdown]
# ## 5. Weighted Layer Pooling + MLP Model

# %%
class Wav2Vec2EmotionClassifier(nn.Module):
    """
    Paper approach: Weighted sum of all wav2vec2 layers + MLP classifier
    """
    def __init__(self, model_name, num_labels, hidden_size=256):
        super().__init__()
        
        # Load wav2vec2 and FREEZE it
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(
            model_name,
            output_hidden_states=True  # Get all layer outputs
        )
        
        # Freeze wav2vec2 completely
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
        
        # Number of layers (base model has 13 layers: 1 CNN + 12 transformer)
        self.num_layers = 13
        
        # Learnable weights for layer pooling
        self.layer_weights = nn.Parameter(torch.ones(self.num_layers) / self.num_layers)
        
        # MLP classifier
        self.classifier = nn.Sequential(
            nn.Linear(768, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_labels)
        )
        
    def forward(self, waveforms):
        # Get all hidden states from wav2vec2
        with torch.no_grad():
            outputs = self.wav2vec2(waveforms)
        
        # hidden_states: tuple of 13 tensors, each (batch, seq_len, 768)
        hidden_states = outputs.hidden_states
        
        # Stack and apply weighted sum
        # Shape: (num_layers, batch, seq_len, 768)
        stacked = torch.stack(hidden_states, dim=0)
        
        # Normalize weights with softmax
        weights = torch.softmax(self.layer_weights, dim=0)
        
        # Weighted sum: (batch, seq_len, 768)
        weighted_sum = torch.sum(stacked * weights.view(-1, 1, 1, 1), dim=0)
        
        # Mean pooling over time: (batch, 768)
        pooled = weighted_sum.mean(dim=1)
        
        # Classify
        logits = self.classifier(pooled)
        
        return logits

# %%
# Create model
model = Wav2Vec2EmotionClassifier(MODEL_NAME, NUM_LABELS, HIDDEN_SIZE).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

# %% [markdown]
# ## 6. Create DataLoaders

# %%
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

train_dataset = EmotionDataset(train_df, DATA_DIR, feature_extractor, MAX_DURATION)
test_dataset = EmotionDataset(test_df, DATA_DIR, feature_extractor, MAX_DURATION)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                          collate_fn=collate_fn, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         collate_fn=collate_fn, num_workers=2)

print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

# %% [markdown]
# ## 7. Training Loop

# %%
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# %%
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(loader, desc="Training"):
        waveforms = batch["waveform"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        logits = model(waveforms)
        loss = criterion(logits, labels)
        loss.backward()
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
        for batch in tqdm(loader, desc="Evaluating"):
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
print("Starting training...")
best_accuracy = 0
history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion, device)
    scheduler.step()
    
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_acc)
    
    print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
    
    # Save best model
    if val_acc > best_accuracy:
        best_accuracy = val_acc
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))
        print(f"  ✅ New best model saved! Accuracy: {val_acc:.4f}")

print(f"\n🎉 Training complete! Best accuracy: {best_accuracy:.4f}")

# %% [markdown]
# ## 8. Evaluation

# %%
# Load best model
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pt")))

# Final evaluation
_, accuracy, y_pred, y_true = evaluate(model, test_loader, criterion, device)

print(f"\nTest Accuracy: {accuracy:.4f}")
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
plt.title("Confusion Matrix - Paper Approach")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.show()

# %%
# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history["train_loss"], label="Train")
axes[0].plot(history["val_loss"], label="Validation")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Loss over epochs")
axes[0].legend()

axes[1].plot(history["val_accuracy"])
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Validation Accuracy")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_history.png"))
plt.show()

# %% [markdown]
# ## 9. Save Predictions for Fusion

# %%
# Generate predictions with probabilities
model.eval()
all_probs = []
all_preds = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Generating predictions"):
        waveforms = batch["waveform"].to(device)
        logits = model(waveforms)
        probs = torch.softmax(logits, dim=-1)
        all_probs.append(probs.cpu().numpy())
        all_preds.extend(logits.argmax(dim=-1).cpu().numpy())

all_probs = np.vstack(all_probs)

# Create predictions DataFrame
predictions_df = pd.DataFrame({
    "utterance_id": test_df["utterance_id"],
    **{f"prob_{emotion}": all_probs[:, i] for i, emotion in enumerate(EMOTION_LABELS)},
    "predicted_emotion": [ID2LABEL[p] for p in all_preds],
    "true_emotion": test_df["emotion"]
})

# Save
predictions_df.to_csv(os.path.join(OUTPUT_DIR, "audio_predictions.csv"), index=False)
predictions_df.to_parquet(os.path.join(OUTPUT_DIR, "audio_predictions.parquet"), index=False)

print("✅ Predictions saved!")
print(predictions_df.head())

# %% [markdown]
# ## 10. Save Model for Inference

# %%
# Save complete model info
torch.save({
    "model_state_dict": model.state_dict(),
    "layer_weights": model.layer_weights.data,
    "config": {
        "model_name": MODEL_NAME,
        "num_labels": NUM_LABELS,
        "hidden_size": HIDDEN_SIZE,
        "label2id": LABEL2ID,
        "id2label": ID2LABEL
    }
}, os.path.join(OUTPUT_DIR, "emotion_model_complete.pt"))

# Save layer weights analysis
weights = torch.softmax(model.layer_weights, dim=0).detach().cpu().numpy()
print("\nLearned layer weights:")
for i, w in enumerate(weights):
    layer_name = "CNN" if i == 0 else f"Transformer {i}"
    print(f"  Layer {i} ({layer_name}): {w:.4f}")

# Plot layer weights
plt.figure(figsize=(10, 4))
plt.bar(range(len(weights)), weights)
plt.xlabel("Layer")
plt.ylabel("Weight")
plt.title("Learned Layer Weights")
plt.xticks(range(len(weights)), ["CNN"] + [f"T{i}" for i in range(1, len(weights))])
plt.savefig(os.path.join(OUTPUT_DIR, "layer_weights.png"))
plt.show()

# %%
print("\n📁 Output files:")
for f in os.listdir(OUTPUT_DIR):
    path = os.path.join(OUTPUT_DIR, f)
    if os.path.isfile(path):
        size = os.path.getsize(path) / 1024 / 1024
        print(f"  {f}: {size:.2f} MB")
    else:
        print(f"  {f}/ (directory)")

print("\n✅ Done! Download files for Spark inference.")
