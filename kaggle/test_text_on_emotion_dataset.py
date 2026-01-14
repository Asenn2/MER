# Test du modèle texte IEMOCAP sur Emotion Dataset de Kaggle
# Dataset: https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp

import os
import torch
import pandas as pd
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ================= CONFIGURATION =================

# Chemin du modèle IEMOCAP entraîné
MODEL_PATH = "/kaggle/input/your-text-model"  # Modifier selon ton modèle

# Chemin Emotion Dataset (vérifier le nom exact sur Kaggle)
EMOTION_DATASET_DIR = "/kaggle/input/emotions-dataset-for-nlp"

# Vos 4 classes IEMOCAP (l'ordre doit être identique à l'entraînement !)
IEMOCAP_LABELS = ["angry", "happy", "neutral", "sad"]
label2id = {label: i for i, label in enumerate(IEMOCAP_LABELS)}
id2label = {i: label for i, label in enumerate(IEMOCAP_LABELS)}

# Mapping Emotion Dataset → IEMOCAP
# Emotion Dataset a: anger, fear, joy, love, sadness, surprise
EMOTION_TO_IEMOCAP = {
    "anger": "angry",
    "joy": "happy",
    "sadness": "sad",
    # Note: pas de "neutral" direct dans Emotion Dataset
    # On peut considérer "love" comme neutral ou l'exclure
}

# ================= CHARGEMENT DU MODÈLE =================
print("📥 Chargement du modèle...")

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"✅ Modèle chargé sur {device}")

# ================= CHARGEMENT DES DONNÉES =================
print("📂 Chargement des données...")

# Emotion Dataset a généralement train.txt, test.txt, val.txt
# Format: texte;label

test_file = None
for root, dirs, files in os.walk(EMOTION_DATASET_DIR):
    for f in files:
        if "test" in f.lower() and f.endswith(".txt"):
            test_file = os.path.join(root, f)
            break
    if test_file:
        break

if not test_file:
    # Essayer avec val.txt
    for root, dirs, files in os.walk(EMOTION_DATASET_DIR):
        for f in files:
            if "val" in f.lower() and f.endswith(".txt"):
                test_file = os.path.join(root, f)
                break
        if test_file:
            break

if not test_file:
    raise FileNotFoundError("❌ Fichier test non trouvé dans le dataset!")

print(f"✅ Fichier trouvé: {test_file}")

# Charger les données
data = []
with open(test_file, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split(';')
        if len(parts) == 2:
            text, label = parts
            data.append({"text": text, "label": label})

df = pd.DataFrame(data)
print(f"📊 Total échantillons: {len(df)}")
print(f"\n📈 Distribution des émotions:")
print(df["label"].value_counts())

# Filtrer pour garder seulement les émotions mappables
df_filtered = df[df["label"].isin(EMOTION_TO_IEMOCAP.keys())].copy()
df_filtered["iemocap_label"] = df_filtered["label"].map(EMOTION_TO_IEMOCAP)

print(f"\n✅ Après filtrage (3 classes): {len(df_filtered)} échantillons")
print(df_filtered["iemocap_label"].value_counts())

# ================= INFÉRENCE =================
print("\n🚀 Lancement de l'inférence...")

predictions = []
true_labels = []
processed_texts = []

MAX_LENGTH = 128

for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Inference"):
    text = row["text"]
    true_label = label2id[row["iemocap_label"]]
    
    try:
        # Tokenization
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Prédiction
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            pred_id = torch.argmax(logits, dim=-1).item()
        
        predictions.append(pred_id)
        true_labels.append(true_label)
        processed_texts.append(text)
        
    except Exception as e:
        print(f"⚠️ Erreur: {e}")

# ================= RÉSULTATS =================
print(f"\n✅ Inférence terminée: {len(predictions)} échantillons")

# Note: On teste sur 3 classes car Emotion Dataset n'a pas "neutral"
TEST_LABELS = ["angry", "happy", "sad"]
test_label2id = {label: i for i, label in enumerate(IEMOCAP_LABELS) if label in TEST_LABELS}

accuracy = accuracy_score(true_labels, predictions)
print(f"\n📊 Accuracy sur Emotion Dataset (Cross-Corpus): {accuracy:.4f} ({accuracy*100:.1f}%)")

print("\n📝 Classification Report:")
print(classification_report(true_labels, predictions, target_names=IEMOCAP_LABELS, labels=range(len(IEMOCAP_LABELS))))

# Confusion Matrix
cm = confusion_matrix(true_labels, predictions, labels=range(len(IEMOCAP_LABELS)))
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=IEMOCAP_LABELS, yticklabels=IEMOCAP_LABELS)
plt.title(f"IEMOCAP Text Model on Emotion Dataset - Accuracy: {accuracy:.1%}")
plt.xlabel("Predicted")
plt.ylabel("True (Emotion Dataset)")
plt.tight_layout()
plt.savefig("/kaggle/working/emotion_dataset_confusion_matrix.png")
plt.show()

# ================= EXEMPLES DE PRÉDICTIONS =================
print("\n📋 Exemples de prédictions:")
for i in range(min(10, len(processed_texts))):
    true_emotion = id2label[true_labels[i]]
    pred_emotion = id2label[predictions[i]]
    status = "✅" if true_labels[i] == predictions[i] else "❌"
    print(f"{status} True: {true_emotion:10} | Pred: {pred_emotion:10} | Text: {processed_texts[i][:50]}...")

print("\n✅ Test terminé!")
