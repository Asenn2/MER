# Weighted Late Fusion pour les 3 modalités
# Combine les prédictions Audio, Texte, Vidéo

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

# ================= CONFIGURATION =================

# Chemins des fichiers de prédictions
AUDIO_PREDICTIONS = "/kaggle/working/audio_predictions.csv"  # ou .parquet
TEXT_PREDICTIONS = "/kaggle/working/text_predictions.csv"
VIDEO_PREDICTIONS = "/kaggle/working/video_predictions.csv"

# Classes
CLASSES = ["angry", "happy", "neutral", "sad"]
label2id = {c: i for i, c in enumerate(CLASSES)}
id2label = {i: c for i, c in enumerate(CLASSES)}

# ================= CHARGEMENT DES DONNÉES =================
print("📂 Chargement des prédictions...")

# Charger les fichiers (CSV ou Parquet)
def load_predictions(path):
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_csv(path)

audio_df = load_predictions(AUDIO_PREDICTIONS)
text_df = load_predictions(TEXT_PREDICTIONS)
video_df = load_predictions(VIDEO_PREDICTIONS)

print(f"✅ Audio: {len(audio_df)} prédictions")
print(f"✅ Texte: {len(text_df)} prédictions")
print(f"✅ Vidéo: {len(video_df)} prédictions")

# ================= FUSION DES DATAFRAMES =================
print("\n🔗 Fusion des prédictions...")

# Renommer les colonnes pour éviter les conflits
audio_df = audio_df.rename(columns={
    'prob_angry': 'prob_angry_audio',
    'prob_happy': 'prob_happy_audio',
    'prob_neutral': 'prob_neutral_audio',
    'prob_sad': 'prob_sad_audio',
    'predicted_emotion': 'pred_audio'
})

text_df = text_df.rename(columns={
    'prob_angry': 'prob_angry_text',
    'prob_happy': 'prob_happy_text',
    'prob_neutral': 'prob_neutral_text',
    'prob_sad': 'prob_sad_text',
    'predicted_emotion': 'pred_text'
})

video_df = video_df.rename(columns={
    'prob_angry': 'prob_angry_video',
    'prob_happy': 'prob_happy_video',
    'prob_neutral': 'prob_neutral_video',
    'prob_sad': 'prob_sad_video',
    'predicted_emotion': 'pred_video'
})

# Fusionner sur utterance_id
merged = audio_df.merge(
    text_df[['utterance_id', 'prob_angry_text', 'prob_happy_text', 'prob_neutral_text', 'prob_sad_text', 'pred_text']], 
    on='utterance_id', 
    how='inner'
)
merged = merged.merge(
    video_df[['utterance_id', 'prob_angry_video', 'prob_happy_video', 'prob_neutral_video', 'prob_sad_video', 'pred_video']], 
    on='utterance_id', 
    how='inner'
)

print(f"✅ {len(merged)} échantillons après fusion")

# ================= MÉTHODE 1: WEIGHTED AVERAGE (Manuel) =================
print("\n" + "="*60)
print("📊 MÉTHODE 1: Weighted Average (Poids manuels)")
print("="*60)

# Poids basés sur les performances in-domain
# Audio: 74%, Texte: 85%, Vidéo: 73%
WEIGHT_AUDIO = 0.30
WEIGHT_TEXT = 0.45  # Le texte a la meilleure accuracy
WEIGHT_VIDEO = 0.25

print(f"\nPoids utilisés:")
print(f"  Audio: {WEIGHT_AUDIO:.0%}")
print(f"  Texte: {WEIGHT_TEXT:.0%}")
print(f"  Vidéo: {WEIGHT_VIDEO:.0%}")

# Calculer les probabilités fusionnées
for emotion in CLASSES:
    merged[f'prob_{emotion}_fused'] = (
        WEIGHT_AUDIO * merged[f'prob_{emotion}_audio'] +
        WEIGHT_TEXT * merged[f'prob_{emotion}_text'] +
        WEIGHT_VIDEO * merged[f'prob_{emotion}_video']
    )

# Prédiction finale = argmax des probabilités fusionnées
prob_cols = [f'prob_{e}_fused' for e in CLASSES]
merged['pred_fused_weighted'] = merged[prob_cols].idxmax(axis=1).str.replace('prob_', '').str.replace('_fused', '')

# Évaluation
true_labels = merged['true_emotion']
pred_weighted = merged['pred_fused_weighted']

acc_weighted = accuracy_score(true_labels, pred_weighted)
print(f"\n📊 Accuracy Weighted Fusion: {acc_weighted:.4f} ({acc_weighted*100:.1f}%)")

# ================= MÉTHODE 2: LEARNED WEIGHTS (Méta-classifieur) =================
print("\n" + "="*60)
print("📊 MÉTHODE 2: Learned Weights (Régression Logistique)")
print("="*60)

# Préparer les features (12 probabilités)
feature_cols = (
    [f'prob_{e}_audio' for e in CLASSES] +
    [f'prob_{e}_text' for e in CLASSES] +
    [f'prob_{e}_video' for e in CLASSES]
)

X = merged[feature_cols].values
y = merged['true_emotion'].map(label2id).values

# Entraîner un méta-classifieur
meta_clf = LogisticRegression(max_iter=1000, random_state=42)
meta_clf.fit(X, y)

# Prédictions
pred_learned = meta_clf.predict(X)
merged['pred_fused_learned'] = [id2label[p] for p in pred_learned]

acc_learned = accuracy_score(y, pred_learned)
print(f"\n📊 Accuracy Learned Fusion: {acc_learned:.4f} ({acc_learned*100:.1f}%)")

# Cross-validation pour éviter l'overfitting
cv_scores = cross_val_score(meta_clf, X, y, cv=5)
print(f"📊 Cross-Validation (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ================= MÉTHODE 3: VOTING (Majoritaire) =================
print("\n" + "="*60)
print("📊 MÉTHODE 3: Majority Voting")
print("="*60)

def majority_vote(row):
    votes = [row['pred_audio'], row['pred_text'], row['pred_video']]
    # Si majorité claire
    for v in votes:
        if votes.count(v) >= 2:
            return v
    # Sinon, prendre le texte (meilleure accuracy)
    return row['pred_text']

merged['pred_fused_voting'] = merged.apply(majority_vote, axis=1)

acc_voting = accuracy_score(true_labels, merged['pred_fused_voting'])
print(f"\n📊 Accuracy Majority Voting: {acc_voting:.4f} ({acc_voting*100:.1f}%)")

# ================= COMPARAISON DES MÉTHODES =================
print("\n" + "="*60)
print("📊 COMPARAISON DES MÉTHODES")
print("="*60)

# Accuracy par modalité seule
acc_audio = accuracy_score(true_labels, merged['pred_audio'])
acc_text = accuracy_score(true_labels, merged['pred_text'])
acc_video = accuracy_score(true_labels, merged['pred_video'])

results = {
    'Audio seul': acc_audio,
    'Texte seul': acc_text,
    'Vidéo seul': acc_video,
    'Weighted Fusion': acc_weighted,
    'Learned Fusion': acc_learned,
    'Majority Voting': acc_voting
}

print("\n| Méthode | Accuracy |")
print("|---------|----------|")
for method, acc in results.items():
    marker = "🏆" if acc == max(results.values()) else "  "
    print(f"| {marker} {method:20} | {acc:.2%} |")

# ================= CLASSIFICATION REPORT (Meilleure méthode) =================
best_method = max(results, key=results.get)
print(f"\n📝 Classification Report ({best_method}):")

if best_method == 'Weighted Fusion':
    best_pred = pred_weighted
elif best_method == 'Learned Fusion':
    best_pred = [id2label[p] for p in pred_learned]
else:
    best_pred = merged['pred_fused_voting']

print(classification_report(true_labels, best_pred, target_names=CLASSES))

# ================= CONFUSION MATRIX =================
cm = confusion_matrix(true_labels, best_pred, labels=CLASSES)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.title(f'Fusion Multimodale - {best_method} ({results[best_method]:.1%})')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('/kaggle/working/fusion_confusion_matrix.png')
plt.show()

# ================= SAUVEGARDE DES RÉSULTATS =================
print("\n💾 Sauvegarde des résultats...")

output_df = merged[['utterance_id', 'true_emotion', 
                     'pred_audio', 'pred_text', 'pred_video',
                     'pred_fused_weighted', 'pred_fused_learned', 'pred_fused_voting']]

output_df.to_csv('/kaggle/working/fusion_predictions.csv', index=False)
output_df.to_parquet('/kaggle/working/fusion_predictions.parquet', index=False)

print("✅ Résultats sauvegardés!")
print("   - fusion_predictions.csv")
print("   - fusion_predictions.parquet")
print("   - fusion_confusion_matrix.png")

# ================= POIDS APPRIS (pour production) =================
print("\n🔧 Coefficients appris par le méta-classifieur:")
print("(Utiles pour la production)")

coef_df = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': meta_clf.coef_.mean(axis=0)  # Moyenne sur les classes
})
coef_df = coef_df.sort_values('Coefficient', ascending=False)
print(coef_df.to_string(index=False))

print("\n✅ Fusion terminée!")
