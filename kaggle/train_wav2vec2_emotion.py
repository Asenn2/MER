# %% [markdown]
# # 🎙️ IEMOCAP Emotion Recognition with Wav2Vec2
# 
# Fine-tuning `superb/wav2vec2-base-superb-er` on IEMOCAP audio dataset
# 
# **Dataset**: IEMOCAP subset (6 emotions: angry, excited, frustrated, happy, neutral, sad)

# %% [markdown]
# ## 1. Setup & Installation

# %%
# Install required packages
!pip install -q transformers datasets librosa torch torchaudio accelerate evaluate scikit-learn

# %%
import os
import torch
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset, DatasetDict

# Check GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# %% [markdown]
# ## 2. Configuration

# %%
# ============================================================
# CONFIGURATION - Modify these paths for your Kaggle dataset
# ============================================================

# Path to your IEMOCAP dataset on Kaggle
# Upload your iemocap_subset folder as a Kaggle dataset
DATA_DIR = "/kaggle/input/iemocap-subset"  # Modify this!

# Model configuration
MODEL_NAME = "superb/wav2vec2-base-superb-er"
SAMPLE_RATE = 16000
MAX_DURATION = 5  # seconds

# Training configuration
BATCH_SIZE = 8  # Reduce if OOM
EPOCHS = 5
LEARNING_RATE = 1e-5
WARMUP_RATIO = 0.1

# Output directory
OUTPUT_DIR = "/kaggle/working/wav2vec2-emotion"

# Emotion labels
EMOTION_LABELS = ["angry", "excited", "frustrated", "happy", "neutral", "sad"]
LABEL2ID = {label: i for i, label in enumerate(EMOTION_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(EMOTION_LABELS)}

print(f"Emotions: {EMOTION_LABELS}")
print(f"Label mapping: {LABEL2ID}")

# %% [markdown]
# ## 3. Load Dataset

# %%
# Load metadata and split
metadata_path = os.path.join(DATA_DIR, "metadata.csv")
split_path = os.path.join(DATA_DIR, "split.csv")

df_metadata = pd.read_csv(metadata_path)
df_split = pd.read_csv(split_path)

# Merge
df = df_metadata.merge(df_split, on="utterance_id")

print(f"Total samples: {len(df)}")
print(f"\nSamples per emotion:")
print(df["emotion"].value_counts())

print(f"\nSplit distribution:")
print(df["split"].value_counts())

# %%
# Separate train and test
train_df = df[df["split"] == "train"].reset_index(drop=True)
test_df = df[df["split"] == "test"].reset_index(drop=True)

print(f"Train samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")

# %% [markdown]
# ## 4. Audio Loading & Preprocessing

# %%
def load_audio(audio_path, max_duration=MAX_DURATION):
    """Load and preprocess audio file."""
    full_path = os.path.join(DATA_DIR, audio_path)
    
    # Load audio
    waveform, sr = librosa.load(full_path, sr=SAMPLE_RATE, duration=max_duration)
    
    # Pad if too short
    max_length = SAMPLE_RATE * max_duration
    if len(waveform) < max_length:
        waveform = np.pad(waveform, (0, max_length - len(waveform)))
    
    return waveform

# Test loading
sample_audio = load_audio(train_df.iloc[0]["audio_path"])
print(f"Sample audio shape: {sample_audio.shape}")
print(f"Duration: {len(sample_audio) / SAMPLE_RATE:.2f}s")

# %% [markdown]
# ## 5. Create HuggingFace Dataset

# %%
# Load feature extractor
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

def prepare_dataset(batch):
    """Prepare a single sample for the model."""
    audio_path = batch["audio_path"]
    
    # Load audio
    waveform = load_audio(audio_path)
    
    # Extract features using wav2vec2 feature extractor
    inputs = feature_extractor(
        waveform,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True
    )
    
    batch["input_values"] = inputs.input_values[0]
    batch["label"] = LABEL2ID[batch["emotion"]]
    
    return batch

# %%
# Create datasets
print("Creating train dataset...")
train_dataset = Dataset.from_pandas(train_df[["utterance_id", "audio_path", "emotion"]])
train_dataset = train_dataset.map(prepare_dataset, remove_columns=["audio_path", "emotion"])

print("Creating test dataset...")
test_dataset = Dataset.from_pandas(test_df[["utterance_id", "audio_path", "emotion"]])
test_dataset = test_dataset.map(prepare_dataset, remove_columns=["audio_path", "emotion"])

# Combine into DatasetDict
dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

print(f"\nDataset created:")
print(dataset)

# %% [markdown]
# ## 6. Load Model

# %%
# Load pre-trained model with custom number of labels
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(EMOTION_LABELS),
    label2id=LABEL2ID,
    id2label=ID2LABEL,
    ignore_mismatched_sizes=True  # Important for different num_labels
)

# Freeze feature extractor (optional - saves memory and training time)
model.freeze_feature_extractor()

print(f"Model loaded: {MODEL_NAME}")
print(f"Number of parameters: {model.num_parameters():,}")

# %% [markdown]
# ## 7. Training Setup

# %%
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import torch

@dataclass
class DataCollatorCTCWithPadding:
    """Data collator that pads inputs to the same length."""
    
    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Separate input values and labels
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        labels = [feature["label"] for feature in features]
        
        # Pad input values
        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        
        return batch

data_collator = DataCollatorCTCWithPadding(feature_extractor=feature_extractor)

# %%
def compute_metrics(eval_pred):
    """Compute accuracy and other metrics."""
    predictions = eval_pred.predictions
    
    # Handle case where model returns tuple (logits, hidden_states)
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    predictions = np.argmax(predictions, axis=-1)
    labels = eval_pred.label_ids
    
    accuracy = accuracy_score(labels, predictions)
    
    return {"accuracy": accuracy}

# %%
# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    warmup_ratio=WARMUP_RATIO,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    push_to_hub=False,
    fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
    dataloader_num_workers=2,
    remove_unused_columns=False,
    report_to="none",  # Disable wandb logging
)

# %%
# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# %% [markdown]
# ## 8. Train! 🚀

# %%
# Train the model
print("Starting training...")
trainer.train()

# %% [markdown]
# ## 9. Evaluation

# %%
# Evaluate on test set
results = trainer.evaluate()
print(f"\nTest Results:")
print(f"  Accuracy: {results['eval_accuracy']:.4f}")

# %%
# Detailed evaluation
predictions = trainer.predict(dataset["test"])
y_pred = np.argmax(predictions.predictions, axis=-1)
y_true = predictions.label_ids

# Classification report
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
plt.title("Confusion Matrix - Wav2Vec2 Emotion Recognition")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.show()

# %% [markdown]
# ## 10. Save Model

# %%
# Save the final model
FINAL_MODEL_PATH = os.path.join(OUTPUT_DIR, "wav2vec2-emotion-final")

trainer.save_model(FINAL_MODEL_PATH)
feature_extractor.save_pretrained(FINAL_MODEL_PATH)

print(f"\n✅ Model saved to: {FINAL_MODEL_PATH}")

# %%
# Create predictions for fusion with other modalities
print("\nGenerating predictions for fusion...")

# Get probabilities
probs = torch.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()

# Create predictions DataFrame
predictions_df = pd.DataFrame({
    "utterance_id": test_df["utterance_id"],
    **{f"prob_{emotion}": probs[:, i] for i, emotion in enumerate(EMOTION_LABELS)},
    "predicted_emotion": [ID2LABEL[p] for p in y_pred],
    "true_emotion": test_df["emotion"]
})

# Save predictions
predictions_df.to_csv(os.path.join(OUTPUT_DIR, "audio_predictions.csv"), index=False)
predictions_df.to_parquet(os.path.join(OUTPUT_DIR, "audio_predictions.parquet"), index=False)

print(f"✅ Predictions saved!")
print(f"\nSample predictions:")
print(predictions_df.head(10))

# %% [markdown]
# ## 11. Test Inference

# %%
def predict_emotion(audio_path, model, feature_extractor):
    """Predict emotion for a single audio file."""
    # Load audio
    waveform = load_audio(audio_path)
    
    # Extract features
    inputs = feature_extractor(
        waveform,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True
    )
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        predicted_id = torch.argmax(probs, dim=-1).item()
    
    return {
        "emotion": ID2LABEL[predicted_id],
        "confidence": probs[0, predicted_id].item(),
        "probabilities": {ID2LABEL[i]: p.item() for i, p in enumerate(probs[0])}
    }

# Test inference
model.to(device)
sample_path = test_df.iloc[0]["audio_path"]
result = predict_emotion(sample_path, model, feature_extractor)

print(f"\nSample prediction:")
print(f"  Audio: {sample_path}")
print(f"  Predicted: {result['emotion']} (confidence: {result['confidence']:.2%})")
print(f"  True: {test_df.iloc[0]['emotion']}")

# %% [markdown]
# ## 12. Download Files
# 
# Download these files from Kaggle:
# - `wav2vec2-emotion-final/` - The trained model
# - `audio_predictions.parquet` - Predictions for fusion with video/text

# %%
# List output files
print("\n📁 Output files:")
for f in os.listdir(OUTPUT_DIR):
    path = os.path.join(OUTPUT_DIR, f)
    if os.path.isfile(path):
        size = os.path.getsize(path) / 1024 / 1024
        print(f"  {f}: {size:.2f} MB")
    else:
        print(f"  {f}/ (directory)")

print("\n✅ Training complete! Download the model and predictions for your Spark inference pipeline.")
