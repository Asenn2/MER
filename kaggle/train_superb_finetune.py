# %% [markdown]
# # 🎙️ IEMOCAP - SUPERB Fine-tuning
# 
# Using `superb/wav2vec2-base-superb-er` (pré-entraîné sur Emotion Recognition)
# avec fine-tuning classique

# %%
!pip install -q transformers datasets librosa torch torchaudio scikit-learn accelerate

# %%
import os
import torch
import librosa
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
import warnings
warnings.filterwarnings('ignore')

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# %% [markdown]
# ## Configuration

# %%
DATA_DIR = "/kaggle/input/iemocap-subset"
OUTPUT_DIR = "/kaggle/working/superb-finetuned"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# SUPERB model - déjà entraîné sur les émotions!
MODEL_NAME = "superb/wav2vec2-base-superb-er"
SAMPLE_RATE = 16000
MAX_DURATION = 10

# Training
BATCH_SIZE = 4
EPOCHS = 15
LEARNING_RATE = 5e-6  # Très bas car modèle déjà bon

EMOTION_LABELS = ["angry", "excited", "frustrated", "happy", "neutral", "sad"]
LABEL2ID = {label: i for i, label in enumerate(EMOTION_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(EMOTION_LABELS)}

# %% [markdown]
# ## Load Data

# %%
df_metadata = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))
df_split = pd.read_csv(os.path.join(DATA_DIR, "split.csv"))
df = df_metadata.merge(df_split, on="utterance_id")

train_df = df[df["split"] == "train"].reset_index(drop=True)
test_df = df[df["split"] == "test"].reset_index(drop=True)

print(f"Train: {len(train_df)}, Test: {len(test_df)}")

# %% [markdown]
# ## Prepare Dataset

# %%
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

def load_audio(audio_path):
    full_path = os.path.join(DATA_DIR, audio_path)
    waveform, _ = librosa.load(full_path, sr=SAMPLE_RATE, duration=MAX_DURATION)
    
    # Pad to fixed length
    max_length = SAMPLE_RATE * MAX_DURATION
    if len(waveform) < max_length:
        waveform = np.pad(waveform, (0, max_length - len(waveform)))
    else:
        waveform = waveform[:max_length]
    
    return waveform

def prepare_dataset(batch):
    waveform = load_audio(batch["audio_path"])
    
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
print("Preparing train dataset...")
train_dataset = Dataset.from_pandas(train_df[["utterance_id", "audio_path", "emotion"]])
train_dataset = train_dataset.map(prepare_dataset, remove_columns=["audio_path", "emotion"])

print("Preparing test dataset...")
test_dataset = Dataset.from_pandas(test_df[["utterance_id", "audio_path", "emotion"]])
test_dataset = test_dataset.map(prepare_dataset, remove_columns=["audio_path", "emotion"])

print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

# %% [markdown]
# ## Load SUPERB Model

# %%
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(EMOTION_LABELS),
    label2id=LABEL2ID,
    id2label=ID2LABEL,
    ignore_mismatched_sizes=True
)

# Fine-tune tout le modèle (pas de freeze)
print(f"Model loaded: {MODEL_NAME}")
print(f"Total params: {model.num_parameters():,}")

# %% [markdown]
# ## Training Setup

# %%
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

@dataclass
class DataCollatorWithPadding:
    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = True
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        labels = [f["label"] for f in features]
        
        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch

data_collator = DataCollatorWithPadding(feature_extractor=feature_extractor)

# %%
def compute_metrics(eval_pred):
    predictions = eval_pred.predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predictions = np.argmax(predictions, axis=-1)
    labels = eval_pred.label_ids
    return {"accuracy": accuracy_score(labels, predictions)}

# %%
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    num_train_epochs=EPOCHS,
    warmup_ratio=0.1,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    fp16=True,
    dataloader_num_workers=2,
    remove_unused_columns=False,
    report_to="none",
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# %% [markdown]
# ## Train

# %%
import gc
torch.cuda.empty_cache()
gc.collect()

print("🚀 Starting fine-tuning...")
trainer.train()

# %% [markdown]
# ## Evaluation

# %%
results = trainer.evaluate()
print(f"\n📊 Test Accuracy: {results['eval_accuracy']:.4f} ({results['eval_accuracy']*100:.1f}%)")

# %%
predictions = trainer.predict(test_dataset)
preds = predictions.predictions
if isinstance(preds, tuple):
    preds = preds[0]

y_pred = np.argmax(preds, axis=-1)
y_true = predictions.label_ids

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=EMOTION_LABELS, digits=3))

# %%
# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"SUPERB Fine-tuned (Acc: {results['eval_accuracy']:.1%})")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.show()

# %% [markdown]
# ## Save

# %%
# Save model
trainer.save_model(os.path.join(OUTPUT_DIR, "superb-emotion-final"))
feature_extractor.save_pretrained(os.path.join(OUTPUT_DIR, "superb-emotion-final"))

# Save predictions
probs = torch.softmax(torch.tensor(preds), dim=-1).numpy()

predictions_df = pd.DataFrame({
    "utterance_id": test_df["utterance_id"],
    **{f"prob_{e}": probs[:, i] for i, e in enumerate(EMOTION_LABELS)},
    "predicted_emotion": [ID2LABEL[p] for p in y_pred],
    "true_emotion": test_df["emotion"]
})

predictions_df.to_parquet(os.path.join(OUTPUT_DIR, "audio_predictions.parquet"))
print("✅ Saved!")

# %%
print("\n" + "="*50)
print(f"🎉 Final Accuracy: {results['eval_accuracy']:.1%}")
print("="*50)
