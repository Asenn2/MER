# %% [markdown]
# # Text Emotion Recognition - FIXED VERSION
# 
# Modèle: `bhadresh-savani/distilbert-base-uncased-emotion`
# Dataset: IEMOCAP transcriptions

# %%
# Nettoyer l'espace disque au début
!rm -rf /kaggle/working/outputs/*
!rm -rf /kaggle/working/saved_models/*
!rm -rf /root/.cache/huggingface/hub/*

!pip -q install --upgrade transformers datasets accelerate evaluate scikit-learn matplotlib

# %%
import os
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    DataCollatorWithPadding, 
    Trainer, 
    TrainingArguments
)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import torch
import gc

SEED = 42
np.random.seed(SEED)

# %% [markdown]
# ## Configuration

# %%
# Paths - MODIFIER SELON TON DATASET KAGGLE
DATA_DIR = "/kaggle/input/iemocap-subset"  # Ton dataset IEMOCAP

# Créer les dossiers
OUTPUT_DIR = "/kaggle/working/text_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model
MODEL_NAME = 'bhadresh-savani/distilbert-base-uncased-emotion'

# Training - OPTIMISÉ
EPOCHS = 15  # Pas 100 !
LR = 2e-5
BATCH = 16

# Emotions IEMOCAP
EMOTION_LABELS = ["angry", "excited", "frustrated", "happy", "neutral", "sad"]
label2id = {lbl: i for i, lbl in enumerate(EMOTION_LABELS)}
id2label = {i: lbl for lbl, i in label2id.items()}
num_labels = len(EMOTION_LABELS)

print(f"Labels: {EMOTION_LABELS}")

# %% [markdown]
# ## Load Data

# %%
# Charger metadata et split
df_metadata = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))
df_split = pd.read_csv(os.path.join(DATA_DIR, "split.csv"))
df = df_metadata.merge(df_split, on="utterance_id")

train_df = df[df["split"] == "train"][["transcription", "emotion"]].reset_index(drop=True)
test_df = df[df["split"] == "test"][["transcription", "emotion"]].reset_index(drop=True)

# Map labels
train_df["label"] = train_df["emotion"].map(label2id)
test_df["label"] = test_df["emotion"].map(label2id)

print(f"Train: {len(train_df)}, Test: {len(test_df)}")
print(f"\nDistribution:\n{train_df['emotion'].value_counts()}")

# %%
# HuggingFace datasets
datasets = DatasetDict({
    'train': Dataset.from_pandas(train_df[['transcription', 'label']]),
    'test':  Dataset.from_pandas(test_df[['transcription', 'label']])
})

# %% [markdown]
# ## Tokenization

# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_batch(batch):
    return tokenizer(batch['transcription'], truncation=True, max_length=128)

encoded = datasets.map(tokenize_batch, batched=True)
collator = DataCollatorWithPadding(tokenizer=tokenizer)

encoded = encoded.remove_columns(['transcription'])
encoded = encoded.rename_column('label', 'labels')
encoded.set_format('torch')

print(f"Tokenized: Train={len(encoded['train'])}, Test={len(encoded['test'])}")

# %% [markdown]
# ## Model

# %%
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

# Option: Freeze backbone (entraîner seulement la tête)
# Décommenter pour transfer learning plus rapide
# for name, param in model.named_parameters():
#     if "classifier" not in name and "pre_classifier" not in name:
#         param.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {trainable:,}")

# %% [markdown]
# ## Training

# %%
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    eval_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=2,  # IMPORTANT: Limiter les checkpoints !
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    greater_is_better=True,
    logging_steps=25,
    seed=SEED,
    report_to='none',
    fp16=True,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return {'accuracy': acc, 'f1': f1}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded['train'],
    eval_dataset=encoded['test'],
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

# %%
# Libérer mémoire
gc.collect()
torch.cuda.empty_cache()

print("🚀 Starting training...")
trainer.train()

# %% [markdown]
# ## Evaluation

# %%
eval_metrics = trainer.evaluate()
print(f"\n📊 Test Accuracy: {eval_metrics['eval_accuracy']:.4f}")
print(f"📊 Test F1: {eval_metrics['eval_f1']:.4f}")

# %%
# Predictions
preds_output = trainer.predict(encoded['test'])
preds = np.argmax(preds_output.predictions, axis=1)
true = preds_output.label_ids

print("\nClassification Report:")
print(classification_report(true, preds, target_names=EMOTION_LABELS, digits=4))

# %%
# Confusion Matrix
cm = confusion_matrix(true, preds)
plt.figure(figsize=(10, 8))
import seaborn as sns
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f"Text Emotion - Accuracy: {eval_metrics['eval_accuracy']:.1%}")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.show()

# %% [markdown]
# ## Save for Fusion

# %%
# Predictions avec probabilités
probs = torch.softmax(torch.tensor(preds_output.predictions), dim=-1).numpy()

# Récupérer utterance_ids
test_utterance_ids = df[df["split"] == "test"]["utterance_id"].values

predictions_df = pd.DataFrame({
    "utterance_id": test_utterance_ids,
    **{f"prob_{e}": probs[:, i] for i, e in enumerate(EMOTION_LABELS)},
    "predicted_emotion": [id2label[p] for p in preds],
    "true_emotion": test_df["emotion"]
})

predictions_df.to_parquet(os.path.join(OUTPUT_DIR, "text_predictions.parquet"))
predictions_df.to_csv(os.path.join(OUTPUT_DIR, "text_predictions.csv"), index=False)

print("✅ Predictions saved!")
print(predictions_df.head())

# %%
# Save model
SAVE_DIR = os.path.join(OUTPUT_DIR, "text_emotion_model")
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print(f"✅ Model saved to: {SAVE_DIR}")

# %%
print("\n" + "="*50)
print(f"🎉 Final Accuracy: {eval_metrics['eval_accuracy']:.1%}")
print(f"🎉 Final F1: {eval_metrics['eval_f1']:.4f}")
print("="*50)
