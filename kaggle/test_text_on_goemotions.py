# Test du modèle texte IEMOCAP sur GoEmotions Dataset
# Dataset: https://www.kaggle.com/datasets/debarshichanda/goemotions

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

# Chemin GoEmotions Dataset
GOEMOTIONS_DIR = "/kaggle/input/goemotions"

# Vos 4 classes IEMOCAP (l'ordre doit être identique à l'entraînement !)
IEMOCAP_LABELS = ["angry", "happy", "neutral", "sad"]
label2id = {label: i for i, label in enumerate(IEMOCAP_LABELS)}
id2label = {i: label for i, label in enumerate(IEMOCAP_LABELS)}

# Mapping GoEmotions → IEMOCAP
# GoEmotions a 27+ émotions, on les mappe vers nos 4 classes
GOEMOTIONS_TO_IEMOCAP = {
    # Angry
    "anger": "angry",
    "annoyance": "angry",
    "disapproval": "angry",
    "disgust": "angry",
    
    # Happy
    "joy": "happy",
    "amusement": "happy",
    "excitement": "happy",
    "love": "happy",
    "optimism": "happy",
    "pride": "happy",
    "admiration": "happy",
    "gratitude": "happy",
    "relief": "happy",
    "approval": "happy",
    "caring": "happy",
    "desire": "happy",
    
    # Neutral
    "neutral": "neutral",
    "realization": "neutral",
    "curiosity": "neutral",
    "surprise": "neutral",
    "confusion": "neutral",
    
    # Sad
    "sadness": "sad",
    "grief": "sad",
    "disappointment": "sad",
    "remorse": "sad",
    "embarrassment": "sad",
    "fear": "sad",
    "nervousness": "sad",
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
print("📂 Chargement des données GoEmotions...")

# Recherche du fichier test
test_file = None
for root, dirs, files in os.walk(GOEMOTIONS_DIR):
    for f in files:
        if "test" in f.lower() and f.endswith(".csv"):
            test_file = os.path.join(root, f)
            break
    if test_file:
        break

if not test_file:
    # Essayer avec goemotions_1.csv, goemotions_2.csv, etc.
    for root, dirs, files in os.walk(GOEMOTIONS_DIR):
        for f in files:
            if f.endswith(".csv"):
                test_file = os.path.join(root, f)
                break
        if test_file:
            break

if not test_file:
    raise FileNotFoundError("❌ Fichier CSV non trouvé dans le dataset!")

print(f"✅ Fichier trouvé: {test_file}")

# Charger les données
df = pd.read_csv(test_file)
print(f"📊 Colonnes disponibles: {df.columns.tolist()}")
print(f"📊 Total échantillons: {len(df)}")

# GoEmotions peut avoir différents formats
# Format 1: colonnes pour chaque émotion (one-hot)
# Format 2: colonne "label" ou "emotion"

# Détecter le format
text_col = None
for col in ['text', 'comment_text', 'sentence']:
    if col in df.columns:
        text_col = col
        break

if text_col is None:
    # Prendre la première colonne non-numérique
    for col in df.columns:
        if df[col].dtype == 'object':
            text_col = col
            break

print(f"✅ Colonne texte: {text_col}")

# Trouver les colonnes d'émotions
emotion_cols = [col for col in df.columns if col in GOEMOTIONS_TO_IEMOCAP.keys()]
print(f"✅ Colonnes d'émotions trouvées: {emotion_cols}")

# Si format one-hot, extraire l'émotion avec la valeur max
if len(emotion_cols) > 0:
    # Format one-hot
    df_samples = []
    for idx, row in df.iterrows():
        # Trouver l'émotion avec la plus haute valeur
        max_emotion = None
        max_val = 0
        for col in emotion_cols:
            if row[col] > max_val:
                max_val = row[col]
                max_emotion = col
        if max_emotion and max_val > 0:
            df_samples.append({
                "text": row[text_col],
                "goemotions_label": max_emotion,
                "iemocap_label": GOEMOTIONS_TO_IEMOCAP.get(max_emotion, None)
            })
    df_filtered = pd.DataFrame(df_samples)
    df_filtered = df_filtered[df_filtered["iemocap_label"].notna()]
else:
    # Essayer de trouver une colonne label
    label_col = None
    for col in ['label', 'emotion', 'labels']:
        if col in df.columns:
            label_col = col
            break
    
    if label_col:
        df["goemotions_label"] = df[label_col]
        df["iemocap_label"] = df[label_col].map(GOEMOTIONS_TO_IEMOCAP)
        df_filtered = df[df["iemocap_label"].notna()].copy()
        df_filtered["text"] = df_filtered[text_col]
    else:
        raise ValueError("Impossible de trouver le format des émotions!")

print(f"\n✅ Après filtrage et mapping: {len(df_filtered)} échantillons")
print(f"\n📈 Distribution des émotions mappées:")
print(df_filtered["iemocap_label"].value_counts())

# Limiter à un échantillon raisonnable si trop grand
MAX_SAMPLES = 5000
if len(df_filtered) > MAX_SAMPLES:
    print(f"⚠️ Échantillonnage à {MAX_SAMPLES} pour accélérer...")
    df_filtered = df_filtered.sample(n=MAX_SAMPLES, random_state=42)

# ================= INFÉRENCE =================
print("\n🚀 Lancement de l'inférence...")

predictions = []
true_labels = []
processed_texts = []

MAX_LENGTH = 128

for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Inference"):
    text = str(row["text"])
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
        pass

# ================= RÉSULTATS =================
print(f"\n✅ Inférence terminée: {len(predictions)} échantillons")

accuracy = accuracy_score(true_labels, predictions)
print(f"\n📊 Accuracy sur GoEmotions (Cross-Corpus): {accuracy:.4f} ({accuracy*100:.1f}%)")

print("\n📝 Classification Report:")
print(classification_report(true_labels, predictions, target_names=IEMOCAP_LABELS, labels=range(len(IEMOCAP_LABELS))))

# Confusion Matrix
cm = confusion_matrix(true_labels, predictions, labels=range(len(IEMOCAP_LABELS)))
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=IEMOCAP_LABELS, yticklabels=IEMOCAP_LABELS)
plt.title(f"IEMOCAP Text Model on GoEmotions - Accuracy: {accuracy:.1%}")
plt.xlabel("Predicted")
plt.ylabel("True (GoEmotions)")
plt.tight_layout()
plt.savefig("/kaggle/working/goemotions_confusion_matrix.png")
plt.show()

# ================= EXEMPLES DE PRÉDICTIONS =================
print("\n📋 Exemples de prédictions:")
for i in range(min(10, len(processed_texts))):
    true_emotion = id2label[true_labels[i]]
    pred_emotion = id2label[predictions[i]]
    status = "✅" if true_labels[i] == predictions[i] else "❌"
    print(f"{status} True: {true_emotion:10} | Pred: {pred_emotion:10} | Text: {processed_texts[i][:60]}...")

print("\n✅ Test terminé!")
