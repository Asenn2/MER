# Test du modèle vidéo IEMOCAP sur RAVDESS Video Dataset
# Dataset: https://www.kaggle.com/datasets/orvile/ravdess-dataset

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import timm
import torchvision.transforms as T
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# --- TRY IMPORTING DECORD FOR SPEED ---
try:
    from decord import VideoReader, cpu
    USE_DECORD = True
except ImportError:
    import cv2
    USE_DECORD = False
    print("WARNING: 'decord' not found. Run '!pip install decord'")

# ================= CONFIGURATION =================

# Chemin du modèle IEMOCAP entraîné
MODEL_PATH = "/kaggle/input/your-video-model/efficientnet_v2_emotion.pth"  # Modifier!

# Chemin RAVDESS Dataset
RAVDESS_DIR = "/kaggle/input/ravdess-dataset"

# Vos 4 classes IEMOCAP
IEMOCAP_LABELS = ["angry", "happy", "neutral", "sad"]
label2id = {label: i for i, label in enumerate(IEMOCAP_LABELS)}
id2label = {i: label for i, label in enumerate(IEMOCAP_LABELS)}

# Mapping RAVDESS → IEMOCAP
# RAVDESS: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
RAVDESS_TO_IEMOCAP = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
}

# Video parameters (must match training)
IMAGE_SIZE = 224
NUM_FRAMES = 8

# ================= HELPER FUNCTIONS =================

def sample_frames_decord(path, num_frames):
    """Sample frames using Decord."""
    try:
        vr = VideoReader(str(path), ctx=cpu(0))
        total_frames = len(vr)
        if total_frames == 0:
            return None
        indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
        frames = vr.get_batch(indices).asnumpy()
        return [frames[i] for i in range(num_frames)]
    except Exception as e:
        return None

def sample_frames_cv2(path, num_frames):
    """Sample frames using OpenCV."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames > 0:
        indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    
    if len(frames) < num_frames:
        return None
    return frames

def build_transforms(image_size=224):
    return T.Compose([
        T.ToPILImage(),
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def parse_ravdess_filename(filename):
    """Parse RAVDESS filename to extract emotion code."""
    # Format: XX-XX-XX-XX-XX-XX-XX.mp4
    parts = Path(filename).stem.split("-")
    if len(parts) >= 3:
        return parts[2]  # 3rd position = emotion
    return None

def forward_video(model, frames):
    """Forward pass avec temporal averaging."""
    b, t, c, h, w = frames.shape
    flat = frames.view(b * t, c, h, w)
    logits = model(flat)
    logits = logits.view(b, t, -1).mean(dim=1)
    return logits

# ================= CHARGEMENT DU MODÈLE =================
print("📥 Chargement du modèle...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charger le checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=device)

# Recréer le modèle (doit correspondre à l'architecture d'entraînement)
num_classes = len(checkpoint.get("classes", IEMOCAP_LABELS))
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()

print(f"✅ Modèle chargé sur {device}")
print(f"   Classes: {checkpoint.get('classes', IEMOCAP_LABELS)}")

# ================= RECHERCHE DES VIDÉOS RAVDESS =================
print("\n📂 Recherche des vidéos RAVDESS...")

video_files = []
true_labels = []

# RAVDESS peut avoir différentes structures
for root, dirs, files in os.walk(RAVDESS_DIR):
    for f in files:
        if f.endswith((".mp4", ".avi", ".mov")):
            emotion_code = parse_ravdess_filename(f)
            if emotion_code in RAVDESS_TO_IEMOCAP:
                iemocap_label = RAVDESS_TO_IEMOCAP[emotion_code]
                video_files.append(os.path.join(root, f))
                true_labels.append(label2id[iemocap_label])

print(f"✅ Trouvé {len(video_files)} vidéos (4 classes)")

if len(video_files) == 0:
    print("❌ Aucune vidéo trouvée! Vérifiez le chemin RAVDESS_DIR")
else:
    # Distribution
    label_counts = pd.Series(true_labels).value_counts().sort_index()
    print("\n📊 Distribution:")
    for idx, count in label_counts.items():
        print(f"   {id2label[idx]}: {count}")

# ================= INFÉRENCE =================
print("\n🚀 Lancement de l'inférence...")

transform = build_transforms(IMAGE_SIZE)
predictions = []
processed_labels = []
processed_paths = []

for path, label in tqdm(zip(video_files, true_labels), total=len(video_files), desc="Inference"):
    try:
        # Extraire les frames
        if USE_DECORD:
            frames = sample_frames_decord(path, NUM_FRAMES)
        else:
            frames = sample_frames_cv2(path, NUM_FRAMES)
        
        if frames is None or len(frames) < NUM_FRAMES:
            continue
        
        # Transformer et empiler
        imgs = []
        for f in frames:
            if f is None or f.size == 0:
                continue
            imgs.append(transform(f))
        
        if len(imgs) != NUM_FRAMES:
            continue
            
        video_tensor = torch.stack(imgs, dim=0).unsqueeze(0).to(device)  # (1, T, C, H, W)
        
        # Prédiction
        with torch.no_grad():
            logits = forward_video(model, video_tensor)
            pred_id = torch.argmax(logits, dim=-1).item()
        
        predictions.append(pred_id)
        processed_labels.append(label)
        processed_paths.append(path)
        
    except Exception as e:
        pass

# ================= RÉSULTATS =================
print(f"\n✅ Inférence terminée: {len(predictions)} / {len(video_files)} vidéos")

if len(predictions) > 0:
    accuracy = accuracy_score(processed_labels, predictions)
    print(f"\n📊 Accuracy sur RAVDESS (Cross-Corpus): {accuracy:.4f} ({accuracy*100:.1f}%)")
    
    print("\n📝 Classification Report:")
    print(classification_report(processed_labels, predictions, target_names=IEMOCAP_LABELS, labels=range(len(IEMOCAP_LABELS))))
    
    # Confusion Matrix
    cm = confusion_matrix(processed_labels, predictions, labels=range(len(IEMOCAP_LABELS)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=IEMOCAP_LABELS, yticklabels=IEMOCAP_LABELS)
    plt.title(f"IEMOCAP Video Model on RAVDESS - Accuracy: {accuracy:.1%}")
    plt.xlabel("Predicted")
    plt.ylabel("True (RAVDESS)")
    plt.tight_layout()
    plt.savefig("/kaggle/working/ravdess_video_confusion_matrix.png")
    plt.show()
    
    print("\n✅ Test terminé!")
else:
    print("❌ Aucune prédiction réussie. Vérifiez les chemins et formats.")
