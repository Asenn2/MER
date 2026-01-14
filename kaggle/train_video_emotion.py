import sys
import zipfile
import logging
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm
import torchvision.transforms as T

# --- TRY IMPORTING DECORD FOR SPEED ---
try:
    from decord import VideoReader, cpu
    USE_DECORD = True
except ImportError:
    import cv2
    USE_DECORD = False
    print("WARNING: 'decord' not found. Fallback to OpenCV (Slower). Run '!pip install decord' for speed.")

# --- CONFIGURATION (EDIT THIS) ---
class Config:
    # Path to your dataset in Kaggle input (Read-Only)
    data_root = Path("/kaggle/input/processedvideo/processed_output_splited") 
    
    # Path to zip if you need to extract it first (leave None if dataset is unzipped)
    zip_path = None 
    
    # Where to save the model (Kaggle Writable Directory)
    output_path = Path("/kaggle/working/efficientnet_v2_emotion.pth")
    predictions_dir = Path("/kaggle/working")  # Where to save predictions
    
    epochs = 20
    batch_size = 16
    lr = 1e-3
    image_size = 224
    num_frames = 8  # Frames per video
    num_workers = 2 
    
    # --- AJOUT EARLY STOPPING ---
    patience = 5 # Si l'accuracy ne monte pas pendant 5 époques, on arrête

# Setup simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def list_videos(data_root: Path, split: str) -> Tuple[List[Path], List[int], List[str]]:
    """Return video paths and labels."""
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
        
    classes = sorted([d.name for d in data_root.iterdir() if d.is_dir()])
    if not classes:
        raise RuntimeError(f"No class folders found in {data_root}")
        
    class_to_idx = {c: i for i, c in enumerate(classes)}
    video_paths = []
    labels = []
    
    for cls in classes:
        split_dir = data_root / cls / split
        if not split_dir.exists():
            continue
            
        for vid in split_dir.glob("*.mp4"):
            video_paths.append(vid)
            labels.append(class_to_idx[cls])
            
    logger.info(f"Found {len(video_paths)} videos for split '{split}' across {len(classes)} classes.")
    return video_paths, labels, classes

def sample_frames_decord(path: Path, num_frames: int) -> List[np.ndarray]:
    """HIGH SPEED: Sample frames using Decord."""
    try:
        vr = VideoReader(str(path), ctx=cpu(0))
        total_frames = len(vr)
        indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
        frames = vr.get_batch(indices).asnumpy() # (T, H, W, C)
        return [frames[i] for i in range(num_frames)]
    except Exception as e:
        logger.warning(f"Decord failed for {path}: {e}. Returning zeros.")
        return [np.zeros((224, 224, 3), dtype=np.uint8)] * num_frames

def sample_frames_cv2(path: Path, num_frames: int) -> List[np.ndarray]:
    """FALLBACK: Sample frames using OpenCV (Slow)."""
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
            else:
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
    cap.release()
    
    while len(frames) < num_frames:
        frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
    return frames

class VideoFrameDataset(Dataset):
    def __init__(self, video_paths: List[Path], labels: List[int], transform, num_frames: int = 8):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.num_frames = num_frames

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        label = self.labels[idx]
        
        if USE_DECORD:
            frames = sample_frames_decord(path, self.num_frames)
        else:
            frames = sample_frames_cv2(path, self.num_frames)
            
        imgs = []
        for f in frames:
            if f is None or f.size == 0:
                f = np.zeros((224, 224, 3), dtype=np.uint8)
            imgs.append(self.transform(f))
            
        return torch.stack(imgs, dim=0), label

def build_transforms(image_size: int = 224):
    return T.Compose([
        T.ToPILImage(),
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def forward_video(model, frames: torch.Tensor):
    # frames: (B, T, 3, H, W)
    b, t, c, h, w = frames.shape
    flat = frames.view(b * t, c, h, w)
    logits = model(flat)  # (B*T, Num_Classes)
    logits = logits.view(b, t, -1).mean(dim=1)
    return logits


def generate_predictions(model, test_loader, test_videos, classes, device):
    """Generate predictions with probabilities for fusion."""
    logger.info("Generating predictions for fusion...")
    
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = forward_video(model, images)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            all_probs.append(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_probs = np.vstack(all_probs)
    
    # Extract utterance_id from video paths (filename without extension)
    utterance_ids = [p.stem for p in test_videos]
    
    # Create id2label mapping
    id2label = {i: c for i, c in enumerate(classes)}
    
    # Create DataFrame
    predictions_df = pd.DataFrame({
        "utterance_id": utterance_ids,
        **{f"prob_{cls}": all_probs[:, i] for i, cls in enumerate(classes)},
        "predicted_emotion": [id2label[p] for p in all_preds],
        "true_emotion": [id2label[l] for l in all_labels]
    })
    
    return predictions_df


def train_engine():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Handle Zip Extraction
    data_root = Config.data_root
    if Config.zip_path and not data_root.exists():
        extract_path = Path("/kaggle/working/temp_data")
        extract_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Extracting zip from {Config.zip_path}...")
        with zipfile.ZipFile(Config.zip_path, "r") as zf:
            zf.extractall(extract_path)
        data_root = extract_path / "processed_output"
        logger.info("Extraction complete.")

    # Load Data
    try:
        train_videos, train_labels, classes = list_videos(data_root, "train")
        test_videos, test_labels, _ = list_videos(data_root, "test")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    transform = build_transforms(Config.image_size)
    train_ds = VideoFrameDataset(train_videos, train_labels, transform, num_frames=Config.num_frames)
    test_ds = VideoFrameDataset(test_videos, test_labels, transform, num_frames=Config.num_frames)

    train_loader = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True, num_workers=Config.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=Config.batch_size, shuffle=False, num_workers=Config.num_workers, pin_memory=True)

    # Model Setup (Utilisation de efficientnet_b0 ou tf_efficientnetv2_s pour éviter les erreurs)
    logger.info(f"Creating Model for {len(classes)} classes...")
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=len(classes))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # --- VARIABLES EARLY STOPPING ---
    best_acc = 0.0
    patience_counter = 0

    # Training Loop
    for epoch in range(1, Config.epochs + 1):
        # 1. TRAIN
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = forward_video(model, images)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"\rEpoch {epoch} [Batch {batch_idx}/{len(train_loader)}] Train Loss: {loss.item():.4f}", end="")

        avg_train_loss = train_loss / len(train_loader)

        # 2. VALIDATION
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0 # Ajout du calcul de val_loss
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                logits = forward_video(model, images)
                
                # Calcul loss validation
                v_loss = criterion(logits, labels)
                val_loss += v_loss.item()
                
                # Calcul accuracy
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        avg_val_loss = val_loss / len(test_loader)
        acc = correct / total if total > 0 else 0
        
        print(f"\n✅ Epoch {epoch} Summary:")
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   Val Loss:   {avg_val_loss:.4f}")
        print(f"   Val Acc:    {acc:.2%} (Best: {best_acc:.2%})")

        # --- LOGIQUE EARLY STOPPING & SAUVEGARDE ---
        # On sauvegarde UNIQUEMENT si le modèle est meilleur
        if acc > best_acc:
            best_acc = acc
            patience_counter = 0 # On reset le compteur
            
            # Sauvegarde du Champion
            Config.output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": model.state_dict(), "classes": classes}, Config.output_path)
            logger.info(f"🏆 New Best Model Saved! Accuracy: {acc:.2%}")
            
        else:
            patience_counter += 1
            print(f"⏳ No improvement. Patience: {patience_counter}/{Config.patience}")
            
            if patience_counter >= Config.patience:
                print(f"\n🛑 EARLY STOPPING TRIGGERED at Epoch {epoch}")
                print(f"Best Accuracy was: {best_acc:.2%}")
                break # On sort de la boucle d'entraînement

    logger.info("Training Finished.")
    
    # =====================================================
    # GENERATE AND SAVE PREDICTIONS FOR FUSION
    # =====================================================
    
    # Reload best model
    logger.info("Loading best model for predictions...")
    checkpoint = torch.load(Config.output_path)
    model.load_state_dict(checkpoint["model_state"])
    
    # Generate predictions
    predictions_df = generate_predictions(model, test_loader, test_videos, classes, device)
    
    # Save predictions
    csv_path = Config.predictions_dir / "video_predictions.csv"
    parquet_path = Config.predictions_dir / "video_predictions.parquet"
    
    predictions_df.to_csv(csv_path, index=False)
    predictions_df.to_parquet(parquet_path, index=False)
    
    logger.info(f"✅ Predictions saved!")
    logger.info(f"   CSV: {csv_path}")
    logger.info(f"   Parquet: {parquet_path}")
    
    # Print sample predictions
    print("\n📊 Sample predictions:")
    print(predictions_df.head(10))
    
    # Calculate accuracy from predictions
    correct = (predictions_df["predicted_emotion"] == predictions_df["true_emotion"]).sum()
    total = len(predictions_df)
    print(f"\n🎯 Final Accuracy: {correct/total:.2%}")
    
    # Print classification report
    from sklearn.metrics import classification_report
    print("\n📋 Classification Report:")
    print(classification_report(
        predictions_df["true_emotion"], 
        predictions_df["predicted_emotion"],
        target_names=classes
    ))

if __name__ == "__main__":
    train_engine()
