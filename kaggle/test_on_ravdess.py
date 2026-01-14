# Test du modèle IEMOCAP sur RAVDESS
# Dataset Kaggle: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

import os
import torch
import librosa
import numpy as np
import pandas as pd
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ================= CONFIGURATION =================

# Chemin du modèle IEMOCAP entraîné
MODEL_PATH = "/kaggle/input/wav2vec2"  # Modifier selon ton modèle

# Chemin RAVDESS (vérifier le nom exact sur Kaggle)
RAVDESS_DIR = "/kaggle/input/ravdess-emotional-speech-audio"

# Mapping des classes IEMOCAP (mêmes que ton entraînement)
IEMOCAP_LABELS = ["angry", "happy", "neutral", "sad"]
label2id = {label: i for i, label in enumerate(IEMOCAP_LABELS)}
id2label = {i: label for i, label in enumerate(IEMOCAP_LABELS)}

# Mapping RAVDESS → IEMOCAP
# RAVDESS: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
RAVDESS_TO_IEMOCAP = {
    "01": "neutral",   # neutral → neutral
    "03": "happy",     # happy → happy
    "04": "sad",       # sad → sad
    "05": "angry",     # angry → angry
}

# ================= CHARGEMENT DU MODÈLE =================
print("📥 Chargement du modèle...")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_PATH)
model = AutoModelForAudioClassification.from_pretrained(MODEL_PATH)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"✅ Modèle chargé sur {device}")

# ================= PARSING DES FICHIERS RAVDESS =================
def parse_ravdess_filename(filename):
    """
    Parse RAVDESS filename to extract emotion.
    Format: XX-XX-XX-XX-XX-XX-XX.wav
            ^^ emotion code (position 2)
    """
    parts = filename.replace(".wav", "").split("-")
    if len(parts) >= 3:
        emotion_code = parts[2]  # 3ème position = émotion
        return emotion_code
    return None

# Trouver tous les fichiers RAVDESS
print("🔍 Recherche des fichiers RAVDESS...")
audio_files = []
true_labels = []

# RAVDESS peut avoir différentes structures, on cherche récursivement
for root, dirs, files in os.walk(RAVDESS_DIR):
    for f in files:
        if f.endswith(".wav"):
            emotion_code = parse_ravdess_filename(f)
            if emotion_code in RAVDESS_TO_IEMOCAP:
                iemocap_label = RAVDESS_TO_IEMOCAP[emotion_code]
                audio_files.append(os.path.join(root, f))
                true_labels.append(label2id[iemocap_label])

print(f"✅ Trouvé {len(audio_files)} fichiers audio (4 classes)")

# Distribution des classes
label_counts = pd.Series(true_labels).value_counts().sort_index()
print("\n📊 Distribution des classes:")
for idx, count in label_counts.items():
    print(f"   {id2label[idx]}: {count}")

# ================= INFÉRENCE =================
print("\n🚀 Lancement de l'inférence...")
predictions = []
processed_labels = []

for path, label in tqdm(zip(audio_files, true_labels), total=len(audio_files), desc="Inference"):
    try:
        # Charger l'audio
        audio, sr = librosa.load(path, sr=16000, duration=5.0)
        
        # Feature extraction
        inputs = feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            max_length=16000 * 5,
            truncation=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Prédiction
        with torch.no_grad():
            logits = model(**inputs).logits
            pred_id = torch.argmax(logits, dim=-1).item()
        
        predictions.append(pred_id)
        processed_labels.append(label)
        
    except Exception as e:
        # print(f"Erreur: {e}")
        pass

# ================= RÉSULTATS =================
print(f"\n✅ Inférence terminée: {len(predictions)} / {len(audio_files)} fichiers")

accuracy = accuracy_score(processed_labels, predictions)
print(f"\n📊 Accuracy sur RAVDESS: {accuracy:.4f} ({accuracy*100:.1f}%)")

print("\n📝 Classification Report:")
print(classification_report(processed_labels, predictions, target_names=IEMOCAP_LABELS))

# Confusion Matrix
cm = confusion_matrix(processed_labels, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=IEMOCAP_LABELS, yticklabels=IEMOCAP_LABELS)
plt.title(f"IEMOCAP Model on RAVDESS - Accuracy: {accuracy:.1%}")
plt.xlabel("Predicted")
plt.ylabel("True (RAVDESS)")
plt.tight_layout()
plt.savefig("/kaggle/working/ravdess_confusion_matrix.png")
plt.show()

print("\n✅ Test terminé!")
