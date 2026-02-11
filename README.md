# MER - Multimodal Emotion Recognition

Projet de reconnaissance des émotions multimodales utilisant les modalités audio, texte et vidéo sur le dataset IEMOCAP.

## 📋 Description

Ce projet implémente un système de reconnaissance d'émotions multimodales qui combine trois modalités différentes :
- **Audio** : Utilisation de modèles wav2vec2 et WavLM pour l'analyse acoustique
- **Texte** : Analyse des transcriptions avec DistilBERT
- **Vidéo** : Extraction de caractéristiques visuelles pour la reconnaissance d'émotions faciales

Le projet permet de :
- Extraire et préparer des sous-ensembles équilibrés du dataset IEMOCAP
- Entraîner des modèles unimodaux pour chaque modalité
- Fusionner les prédictions multimodales pour améliorer les performances
- Évaluer les modèles sur différents datasets (RAVDESS, FER2013, GoEmotions)

### Émotions supportées

Le projet reconnaît 6 émotions principales :
- Angry (Colère)
- Happy (Joie)
- Sad (Tristesse)
- Neutral (Neutre)
- Frustrated (Frustration)
- Excited/Fearful (Excitation/Peur - sélection automatique)

## 🔧 Prérequis

### Logiciels requis
- Python 3.8+
- FFmpeg (pour l'extraction de segments vidéo)
- CUDA (optionnel, pour l'accélération GPU)

### Bibliothèques Python
Les principales dépendances incluent :
- `torch` (PyTorch)
- `transformers` (Hugging Face)
- `datasets`
- `librosa` ou `torchaudio` (traitement audio)
- `opencv-python` (traitement vidéo)
- `pandas`, `numpy`
- `scikit-learn`
- `matplotlib`, `seaborn` (visualisation)

## 📦 Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/Asenn2/MER.git
cd MER
```

### 2. Créer un environnement virtuel

```bash
python -m venv venv
source venv/bin/activate  # Sur Linux/Mac
# ou
venv\Scripts\activate  # Sur Windows
```

### 3. Installer les dépendances

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate
pip install librosa soundfile
pip install opencv-python pillow
pip install pandas numpy scikit-learn
pip install matplotlib seaborn
pip install evaluate
```

### 4. Installer FFmpeg

**Ubuntu/Debian :**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

**macOS :**
```bash
brew install ffmpeg
```

**Windows :**
Télécharger depuis [ffmpeg.org](https://ffmpeg.org/download.html)

## 📊 Dataset

### Obtention du dataset IEMOCAP

1. Le dataset IEMOCAP doit être obtenu depuis [IEMOCAP](https://sail.usc.edu/iemocap/)
2. Placer l'archive `IEMOCAP_full_release.tar.gz` dans un répertoire accessible
3. Les modèles pré-entraînés peuvent être téléchargés depuis Hugging Face

### Extraction d'un sous-ensemble

Le script `subset.py` permet d'extraire un sous-ensemble équilibré du dataset :

```bash
python subset.py \
    --input /path/to/IEMOCAP_full_release.tar.gz \
    --output /path/to/iemocap_subset \
    --target-size 5.0 \
    --dry-run  # Pour tester sans extraire
```

**Options :**
- `--input, -i` : Chemin vers l'archive IEMOCAP
- `--output, -o` : Répertoire de sortie
- `--target-size, -s` : Taille cible en GB (défaut: 5.0)
- `--dry-run` : Mode simulation sans extraction
- `--force-emotion` : Forcer la sélection de la 6ème émotion (excited ou fearful)

## 🚀 Utilisation

### 1. Extraction de segments vidéo

Pour extraire des segments vidéo individuels pour chaque utterance :

```bash
python extract_video_segments.py
```

**Note :** Modifier le `base_path` dans le script selon votre configuration.

### 2. Entraînement des modèles unimodaux

#### Modèle Audio (Wav2Vec2)

```python
# Dans Kaggle ou Jupyter
python kaggle/train_wav2vec2_emotion.py
```

Options disponibles :
- `train_wav2vec2_emotion.py` : Version de base
- `train_wav2vec2_improved.py` : Version améliorée
- `train_wav2vec2_final.py` : Version finale optimisée
- `train_wavlm_emotion.py` : Utilise WavLM au lieu de Wav2Vec2

#### Modèle Texte (DistilBERT)

```python
python kaggle/train_text_emotion.py
```

Utilise le modèle `bhadresh-savani/distilbert-base-uncased-emotion`.

#### Modèle Vidéo

```python
python kaggle/train_video_emotion.py
```

### 3. Fusion multimodale

Pour combiner les prédictions des trois modalités :

```python
python kaggle/fusion_multimodal.py
```

Le script effectue une fusion tardive pondérée des prédictions audio, texte et vidéo.

### 4. Évaluation sur d'autres datasets

```python
# Test sur RAVDESS (audio)
python kaggle/test_on_ravdess.py

# Test sur FER2013 (vidéo)
python kaggle/test_video_on_fer2013.py

# Test sur GoEmotions (texte)
python kaggle/test_text_on_goemotions.py
```

## 📁 Structure du projet

```
MER/
├── README.md                           # Ce fichier
├── subset.py                           # Extraction de sous-ensembles IEMOCAP
├── extract_video_segments.py           # Extraction de segments vidéo
└── kaggle/                             # Scripts d'entraînement et test
    ├── train_wav2vec2_emotion.py       # Entraînement audio (Wav2Vec2)
    ├── train_wav2vec2_improved.py      # Version améliorée
    ├── train_wav2vec2_final.py         # Version finale
    ├── train_wavlm_emotion.py          # Entraînement audio (WavLM)
    ├── train_text_emotion.py           # Entraînement texte
    ├── train_video_emotion.py          # Entraînement vidéo
    ├── fusion_multimodal.py            # Fusion des modalités
    ├── test_on_ravdess.py              # Test sur RAVDESS
    ├── test_video_on_fer2013.py        # Test vidéo sur FER2013
    ├── test_video_on_ravdess.py        # Test vidéo sur RAVDESS
    ├── test_text_on_emotion_dataset.py # Test texte
    └── test_text_on_goemotions.py      # Test sur GoEmotions
```

## 🔬 Exemple d'utilisation complet

### Workflow typique

```bash
# 1. Extraire un sous-ensemble du dataset
python subset.py \
    --input /data/IEMOCAP_full_release.tar.gz \
    --output /data/iemocap_subset \
    --target-size 5.0

# 2. Extraire les segments vidéo
python extract_video_segments.py

# 3. Entraîner les modèles (dans Kaggle/Jupyter)
# - Téléverser le sous-ensemble sur Kaggle
# - Exécuter les notebooks d'entraînement

# 4. Fusionner les prédictions
python kaggle/fusion_multimodal.py

# 5. Évaluer sur d'autres datasets
python kaggle/test_on_ravdess.py
```

### Utilisation sur Kaggle

1. Téléverser le dataset IEMOCAP subset sur Kaggle
2. Créer un nouveau notebook
3. Copier le contenu d'un script d'entraînement
4. Ajuster les chemins (`/kaggle/input/...`)
5. Lancer l'entraînement avec GPU

## 📈 Performances

Le système multimodal combine les forces de chaque modalité :
- **Audio** : Capture les indices prosodiques et paralinguistiques
- **Texte** : Analyse le contenu sémantique
- **Vidéo** : Détecte les expressions faciales

La fusion tardée pondérée améliore généralement les performances par rapport aux modèles unimodaux.

## Disponibilité des models et dataset

Les modèles utilisés sont déja présent sur huggingface : Asenn/MultimodalEmotionRecognition. Ainsi que le dataset utilisé : Asenn/Iemocap_Subset
