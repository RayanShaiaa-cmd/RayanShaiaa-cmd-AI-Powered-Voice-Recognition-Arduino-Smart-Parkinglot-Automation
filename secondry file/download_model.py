import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA

# =============================
# الإعدادات
# =============================
TRAIN_DIR = "commands"
TEST_DIR = "Test"
FS = 16000
N_MFCC = 26

# =============================
# استخراج ميزات متقدمة (عالية الدقة)
# =============================
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=FS)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = []

    for feature in [mfcc, delta, delta2]:
        features.extend(np.mean(feature, axis=1))
        features.extend(np.std(feature, axis=1))
        features.extend(np.min(feature, axis=1))
        features.extend(np.max(feature, axis=1))

    return np.array(features)

# =============================
# تحميل البيانات
# =============================
def load_dataset(folder):
    X, y = [], []
    labels = sorted(os.listdir(folder))

    for label in labels:
        path = os.path.join(folder, label)
        if not os.path.isdir(path):
            continue

        for file in os.listdir(path):
            if file.endswith(".wav"):
                full_path = os.path.join(path, file)
                X.append(extract_features(full_path))
                y.append(label)

    return np.array(X), np.array(y), labels

# =============================
# تحميل بيانات التدريب
# =============================
X_train, y_train, labels = load_dataset(TRAIN_DIR)
print("🔹 Training samples:", X_train.shape)
print("🔹 Number of classes:", len(labels))

# =============================
# رسم توازن البيانات
# =============================
plt.figure(figsize=(6,4))
sns.countplot(x=y_train)
plt.title("عدد العينات لكل أمر")
plt.xlabel("الأمر")
plt.ylabel("عدد التسجيلات")
plt.show()

# =============================
# تقييس البيانات
# =============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# =============================
# PCA Visualization
# =============================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

plt.figure(figsize=(8,6))
sns.scatterplot(
    x=X_pca[:,0],
    y=X_pca[:,1],
    hue=y_train,
    palette="tab10"
)
plt.title("تمثيل الميزات باستخدام PCA")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()

# =============================
# تدريب نموذج عالي الدقة
# =============================
model = RandomForestClassifier(
    n_estimators=1000,
    oob_score=True,
    random_state=50
)

model.fit(X_train_scaled, y_train)
print("✅ تم تدريب النموذج")

# =============================
# تحميل بيانات الاختبار
# =============================

X_test, y_test, _ = load_dataset(TEST_DIR)
X_test_scaled = scaler.transform(X_test)

# =============================
# التنبؤ والتقييم
# =============================
y_pred = model.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
print(f"\n🎯 دقة النموذج: {acc*100:.2f}%\n")

print("📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# =============================
# Confusion Matrix
# =============================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# =============================
# حفظ النموذج
# =============================
'''joblib.dump(model, "model14.joblib")
joblib.dump(scaler, "model14_scalar.joblib")'''

print("💾 تم حفظ النموذج والمقياس")