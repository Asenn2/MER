# Test du modèle vidéo IEMOCAP sur FER2013 (Images)
# Dataset: https://www.kaggle.com/datasets/msambare/fer2013

import os
import torch
import numpy as np
import pandas as pd
import timm
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ================= CONFIGURATION =================

# Chemin du modèle IEMOCAP entraîné
MODEL_PATH = "/kaggle/input/your-video-model/efficientnet_v2_emotion.pth"  # Modifier!

# Chemin FER2013 Dataset
FER2013_DIR = "/kaggle/input/fer2013"

# Vos 4 classes IEMOCAP
IEMOCAP_LABELS = ["angry", "happy", "neutral", "sad"]
label2id = {label: i for i, label in enumerate(IEMOCAP_LABELS)}
id2label = {i: label for i, label in enumerate(IEMOCAP_LABELS)}

# Mapping FER2013 → IEMOCAP
# FER2013: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
FER2013_TO_IEMOCAP = {
    "angry": "angry",
    "Angry": "angry",
    "0": "angry",
    
    "happy": "happy",
    "Happy": "happy",
    "3": "happy",
    
    "neutral": "neutral",
    "Neutral": "neutral",
    "6": "neutral",
    
    "sad": "sad",
    "Sad": "sad",
    "4": "sad",
}

IMAGE_SIZE = 224

# ================= CHARGEMENT DU MODÈLE =================
print("📥 Chargement du modèle...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(MODEL_PATH, map_location=device)
num_classes = len(checkpoint.get("classes", IEMOCAP_LABELS))
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()

print(f"✅ Modèle chargé sur {device}")

# ================= TRANSFORMS =================
transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ================= RECHERCHE DES IMAGES FER2013 =================
print("\n📂 Recherche des images FER2013...")

image_files = []
true_labels = []

# FER2013 structure: test/angry/*.jpg, test/happy/*.jpg, etc.
test_dir = None
for d in ["test", "Test", "validation", "val"]:
    potential_dir = os.path.join(FER2013_DIR, d)
    if os.path.exists(potential_dir):
        test_dir = potential_dir
        break

if test_dir is None:
    # Chercher récursivement
    for root, dirs, files in os.walk(FER2013_DIR):
        if any(f.endswith(('.jpg', '.png', '.jpeg')) for f in files):
            test_dir = FER2013_DIR
            break

if test_dir is None:
    raise FileNotFoundError("❌ Dossier test non trouvé!")

print(f"✅ Dossier trouvé: {test_dir}")

# Parcourir les dossiers d'émotions
for emotion_folder in os.listdir(test_dir):
    emotion_path = os.path.join(test_dir, emotion_folder)
    if not os.path.isdir(emotion_path):
        continue
    
    # Mapper vers IEMOCAP
    iemocap_label = FER2013_TO_IEMOCAP.get(emotion_folder, None)
    if iemocap_label is None:
        continue
    
    # Ajouter les images
    for img_file in os.listdir(emotion_path):
        if img_file.endswith(('.jpg', '.png', '.jpeg')):
            image_files.append(os.path.join(emotion_path, img_file))
            true_labels.append(label2id[iemocap_label])

print(f"✅ Trouvé {len(image_files)} images (4 classes)")

if len(image_files) == 0:
    print("❌ Aucune image trouvée!")
else:
    # Distribution
    label_counts = pd.Series(true_labels).value_counts().sort_index()
    print("\n📊 Distribution:")
    for idx, count in label_counts.items():
        print(f"   {id2label[idx]}: {count}")

# Limiter si trop d'images
MAX_SAMPLES = 3000
if len(image_files) > MAX_SAMPLES:
    print(f"\n⚠️ Échantillonnage à {MAX_SAMPLES} images...")
    indices = np.random.choice(len(image_files), MAX_SAMPLES, replace=False)
    image_files = [image_files[i] for i in indices]
    true_labels = [true_labels[i] for i in indices]

# ================= INFÉRENCE =================
print("\n🚀 Lancement de l'inférence...")

predictions = []
processed_labels = []

for path, label in tqdm(zip(image_files, true_labels), total=len(image_files), desc="Inference"):
    try:
        # Charger l'image
        img = Image.open(path).convert("RGB")
        
        # Transformer
        img_tensor = transform(img).unsqueeze(0).to(device)  # (1, 3, 224, 224)
        
        # Prédiction (directement sur l'image, sans temporal averaging)
        with torch.no_grad():
            logits = model(img_tensor)
            pred_id = torch.argmax(logits, dim=-1).item()
        
        predictions.append(pred_id)
        processed_labels.append(label)
        
    except Exception as e:
        pass

# ================= RÉSULTATS =================
print(f"\n✅ Inférence terminée: {len(predictions)} images")

if len(predictions) > 0:
    accuracy = accuracy_score(processed_labels, predictions)
    print(f"\n📊 Accuracy sur FER2013 (Cross-Corpus): {accuracy:.4f} ({accuracy*100:.1f}%)")
    
    print("\n📝 Classification Report:")
    print(classification_report(processed_labels, predictions, target_names=IEMOCAP_LABELS, labels=range(len(IEMOCAP_LABELS))))
    
    # Confusion Matrix
    cm = confusion_matrix(processed_labels, predictions, labels=range(len(IEMOCAP_LABELS)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=IEMOCAP_LABELS, yticklabels=IEMOCAP_LABELS)
    plt.title(f"IEMOCAP Video Model on FER2013 - Accuracy: {accuracy:.1%}")
    plt.xlabel("Predicted")
    plt.ylabel("True (FER2013)")
    plt.tight_layout()
    plt.savefig("/kaggle/working/fer2013_confusion_matrix.png")
    plt.show()
    
    print("\n✅ Test terminé!")
