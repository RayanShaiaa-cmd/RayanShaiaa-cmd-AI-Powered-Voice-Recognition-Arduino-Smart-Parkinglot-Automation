import os
from datetime import datetime
import numpy as np
import pandas as pd
import librosa
import librosa.display
import sounddevice as sd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
import pyttsx3 as py

# --------------------- CONFIG ---------------------
TRAIN_DIR = r"C:\Users\hp\AppData\Local\Programs\Python\Python312\مشروع الاكوع\commands"
TEST_DIR = r"C:\Users\hp\AppData\Local\Programs\Python\Python312\مشروع الاكوع\Test"
FS = 16000
N_MFCC = 26
DURATION = 2.0
MODEL_PATH = r"C:\Users\hp\AppData\Local\Programs\Python\Python312\مشروع الاكوع\model14.joblib"
SCALER_PATH = r"C:\Users\hp\AppData\Local\Programs\Python\Python312\مشروع الاكوع\model14_scalar.joblib"
SERIAL_PORT = "COM3"
BAUDRATE = 115200

# --------------------- APP SETUP ---------------------
st.set_page_config(page_title="Voice Command Platform", layout="wide")
st.title("🎤 منصة التعرف على الأوامر الصوتية - Streamlit Monolith")

# ==================================================================================================
# Utility functions and repeated explanatory blocks (to increase file length and clarity)
# ==================================================================================================
def ensure_dir(path):
    """Ensure a directory exists; if not create it."""
    if not os.path.exists(path):
        os.makedirs(path)

# =============================
# استخراج ميزات متقدمة (عالية الدقة)
# =============================
def extract_features(file_path):
    """
    Input: path to a wav file
    Output: numpy array of features (312 dims)
    Logic: identical to provided script
    """
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

def extract_features_from_signal(y, sr=FS):
    """
    Same logic as extract_features but accepts raw audio array and sampling rate.
    This keeps feature-extraction consistent between training and live prediction.
    """
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

# ==================================================================================================
# Data loading functions - preserve original behavior
# ==================================================================================================
def load_dataset(folder):
    X, y = [], []
    if not os.path.exists(folder):
        return None, None
    labels = sorted(os.listdir(folder))

    for label in labels:
        path = os.path.join(folder, label)
        if not os.path.isdir(path):
            continue

        for file in os.listdir(path):
            if file.endswith('.wav'):
                full_path = os.path.join(path, file)
                feats = extract_features(full_path)
                X.append(feats)
                y.append(label)
    if len(X) == 0:
        return None, None
    return np.array(X), np.array(y)

# ==================================================================================================
# Training pipeline - RandomForest as in original
# ==================================================================================================
def train_and_save_model(force_retrain=False):
    """
    Train RandomForest on TRAIN_DIR and save model + scaler if training is successful.
    Returns model, scaler, X_train, y_train
    """
    X_train, y_train = load_dataset(TRAIN_DIR)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(n_estimators=1000, oob_score=True, random_state=50)
    model.fit(X_scaled, y_train)
    # Save
    try:
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
    except Exception as e:
        st.error(f"Failed to save model or scaler: {e}")

    return model, scaler, X_train, y_train

# ==================================================================================================
# Load model if exists
# ==================================================================================================
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ==================================================================================================
# Evaluate on test dataset - preserve original logic
# ==================================================================================================
def evaluate_on_test(model, scaler):
    X_test, y_test = load_dataset(TEST_DIR)
    X_test_scaled = scaler.transform(X_test)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    st.write(f"🎯 Test Accuracy: {acc*100:.2f}%")
    st.text_area("Classification Report", classification_report(y_test, y_pred), height=200)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    st.pyplot(fig)

# ==================================================================================================
# Exploratory Data Analysis (EDA) visualizations from the third script
# ==================================================================================================

def do_eda():
    durations = []
    energies = []
    labels = []
    signals = []
    mfcc_features = []

    command_names = sorted(os.listdir(TRAIN_DIR))

    for command in command_names:
        path = os.path.join(TRAIN_DIR, command)
        if not os.path.isdir(path):
            continue

        for file in os.listdir(path):
            if file.endswith('.wav'):
                full_path = os.path.join(path, file)
                y, sr = librosa.load(full_path, sr=FS)

                durations.append(len(y) / sr)
                energies.append(np.sum(y**2))
                labels.append(command)
                signals.append(y)

                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
                mfcc_features.append(np.mean(mfcc, axis=1))

    if len(labels) == 0:
        st.warning("No wav files found for EDA")
        return

    mfcc_features = np.array(mfcc_features)

    # 1. Count per command
    df_labels = pd.DataFrame({"Command": labels})
    fig1 = px.histogram(df_labels, x='Command', title='عدد العينات لكل أمر')
    st.plotly_chart(fig1, use_container_width=True)

    # 2. Duration distribution
    fig2, ax2 = plt.subplots(figsize=(8,4))
    sns.boxplot(x=labels, y=durations, ax=ax2)
    ax2.set_title('توزيع مدة التسجيلات')
    st.pyplot(fig2)

    # 3. Energy distribution
    fig3, ax3 = plt.subplots(figsize=(8,4))
    sns.boxplot(x=labels, y=energies, ax=ax3)
    ax3.set_title('توزيع طاقة الصوت')
    st.pyplot(fig3)

    # 4. MFCC spectrogram of first sample
    example_signal = signals[0]
    mfcc_example = librosa.feature.mfcc(y=example_signal, sr=FS, n_mfcc=N_MFCC)
    fig4, ax4 = plt.subplots(figsize=(10,4))
    librosa.display.specshow(mfcc_example, x_axis='time', sr=FS, ax=ax4)
    img = librosa.display.specshow(
        mfcc_example,
        sr=FS,
        x_axis="time",
        ax=ax4
    )
    fig4.colorbar(img, ax=ax4)

    ax4.set_title('MFCC Spectrogram (عينة واحدة)')
    st.pyplot(fig4)

    # 5. Compare mean MFCC across commands
    fig5, ax5 = plt.subplots(figsize=(10,5))
    # For safety, find first index per command
    df_labels_pd = pd.DataFrame({'Command': labels})
    for command in command_names:
        idxs = df_labels_pd[df_labels_pd['Command']==command].index
        if len(idxs) == 0:
            continue
        idx = idxs[0]
        ax5.plot(mfcc_features[idx], label=command)
    ax5.set_title('مقارنة متوسط MFCC بين الأوامر')
    ax5.set_xlabel('MFCC Index')
    ax5.set_ylabel('القيمة')
    ax5.legend()
    st.pyplot(fig5)

    # 6. Heatmap of mean features
    mean_features = []
    for command in command_names:
        mask = np.array(labels) == command
        if np.sum(mask) == 0:
            mean_features.append(np.zeros(mfcc_features.shape[1]))
        else:
            mean_features.append(np.mean(mfcc_features[mask], axis=0))

    fig6, ax6 = plt.subplots(figsize=(12,6))
    sns.heatmap(mean_features, yticklabels=command_names, cmap='coolwarm', ax=ax6)
    ax6.set_title('Heatmap لمتوسط MFCC لكل أمر')
    ax6.set_xlabel('MFCC Index')
    ax6.set_ylabel('الأمر')
    st.pyplot(fig6)

    # 7. PCA on mfcc_features
    scaler_local = StandardScaler()
    X_scaled = scaler_local.fit_transform(mfcc_features)
    pca_local = PCA(n_components=2)
    X_pca_local = pca_local.fit_transform(X_scaled)

    fig7 = px.scatter(x=X_pca_local[:,0], y=X_pca_local[:,1], color=labels, title='PCA (MFCC only)')
    st.plotly_chart(fig7, use_container_width=True)

    # 8. Boxplot for MFCC[0]
    fig8, ax8 = plt.subplots(figsize=(8,5))
    sns.boxplot(x=labels, y=mfcc_features[:,0], ax=ax8)
    ax8.set_title('توزيع MFCC[0] بين الأوامر')
    st.pyplot(fig8)

    st.success('✅ تم تحليل البيانات بصريًا من جميع النواحي')

# ==================================================================================================
# Live microphone recording and prediction (preserve original interactive loop but adapted for Streamlit)
# ==================================================================================================
commands = ["on","off","unlock","alarm","number"]
masseges = [
    "Ok , i will turn on the lamp right now",
    "Ok , i will turn off the lamp right now",
    "The door will unlock , welcome to our parkinglot",
    "The Siren will start on right now",
    "Ok , i will write the number of cars"
            ]
def voice(text):
    for index in range(len(commands)):
        if text == commands[index]:
            engine = py.init()
            engine.setProperty('rate', 150)
            engine.say(masseges[index])
            engine.runAndWait()
def record_audio_streamlit(duration=DURATION):
    """
    Record from microphone using sounddevice; return audio array.
    In Streamlit, long blocking calls are acceptable but we show spinner.
    """
    with st.spinner('🎤 تسجيل الصوت... الرجاء التكلم الآن'):
        audio = sd.rec(int(duration * FS), samplerate=FS, channels=1, dtype='float32')
        sd.wait()
    return audio.flatten()

def predict_live(model, scaler):
    if model is None or scaler is None:
        st.error('Model or scaler missing. Train or load model first.')
        return

    audio = record_audio_streamlit()
    features = extract_features_from_signal(audio, FS)
    features = features.reshape(1, -1)
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    confidence = float(np.max(model.predict_proba(features_scaled)))
    voice(prediction)
    get_arduino()
    send_to_arduino(prediction)
    st.write(f"🔮 الأمر المتوقع: **{prediction}**")
    st.write(f"📊 نسبة الثقة: **{confidence*100:.2f}%**")

    return prediction, confidence, features

import serial
import time
import streamlit as st

SERIAL_PORT = "COM7"   # غيّر المنفذ حسب جهازك
BAUDRATE = 115200

# ------------------ دالة الاتصال بالأردوينو (مرة واحدة فقط) ------------------
def get_arduino():
    """
    تتأكد من وجود اتصال بالأردوينو أول مرة فقط وتخزنه في session_state
    """
    if "arduino" not in st.session_state:
        try:
            st.session_state.arduino = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
            time.sleep(2)  # وقت للتأكد من اتصال الأردوينو
            st.success(f"✅ Arduino متصل على {SERIAL_PORT}")
        except serial.SerialException as e:
            st.session_state.arduino = None
            st.error(f"⚠️ فشل الاتصال بالأردوينو: {e}")
    return st.session_state.arduino

# ------------------ دالة إرسال الأمر للأردوينو ------------------
def send_to_arduino(command):
    """
    ترسل أي أمر للأردوينو باستخدام الاتصال الموجود
    """
    arduino = get_arduino()
    if arduino is None:
        st.warning("⚠️ لم يتم الاتصال بالأردوينو")
        return False

    try:
        arduino.write((command + "\n").encode("utf-8"))
        return True
    except Exception as e:
        st.error(f"⚠️ فشل إرسال الأمر: {e}")
        return False

# ==================================================================================================
# PCA visualization across full training features (using same features as training)
# ==================================================================================================

def prepare_pca_for_training(X_train):
    scaler_local = StandardScaler()
    X_scaled = scaler_local.fit_transform(X_train)
    pca_local = PCA(n_components=2)
    X_pca = pca_local.fit_transform(X_scaled)
    return scaler_local, pca_local, X_pca

# ==================================================================================================
# UI: Sidebar controls
# ==================================================================================================
st.sidebar.header('الإعدادات')
if st.sidebar.button('إنشاء المجلدات الافتراضية'):
    ensure_dir(TRAIN_DIR)
    ensure_dir(TEST_DIR)
    st.sidebar.success('تم إنشاء المجلدات (إن لم تكن موجودة)')

action = st.sidebar.selectbox('اختر جزءًا', ['Dataset EDA', 'Train Model', 'Evaluate Test', 'Live Predict', 'PCA Visualize', 'Download Model'])

# ==================================================================================================
# Main app behavior depending on chosen action
# ==================================================================================================
if 'history' not in st.session_state:
    st.session_state.history = []

if action == 'Dataset EDA':
    st.header('📊 تحليل بيانات التدريب (EDA)')
    do_eda()

elif action == 'Train Model':
    st.header('🧠 تدريب النموذج (RandomForest)')
    if st.button('Start Training'):
        with st.spinner('Training...'):
            model, scaler, X_train, y_train = train_and_save_model(force_retrain=True)
            if model is not None:
                st.success('✅ تم تدريب النموذج وحفظه')
                st.write(f'عدد العينات: {X_train.shape[0]}')
                st.write(f'عدد الميزات: {X_train.shape[1]}')
            else:
                st.error('فشل التدريب')

elif action == 'Evaluate Test':
    st.header('🔍 تقييم على مجموعة الاختبار')
    if model is None or scaler is None:
        st.warning('لا يوجد نموذج محمّل. درّب النموذج أولاً أو حمّله.')
    else:
        if st.button('Run Evaluation'):
            evaluate_on_test(model, scaler)

elif action == 'Live Predict':
    st.header('🎙️ التنبؤ من الميكروفون (Live)')
    st.write('اضغط على زر التسجيل ثم تحدّث لأمر من أوامر التدريب.')
    col1, col2 = st.columns(2)
    with col1:
        if st.button('Record & Predict'):
            prediction, confidence, features = predict_live(model, scaler)
            st.session_state.history.append((datetime.now(), prediction, confidence))

    with col2:
        st.write('History:')
        if len(st.session_state.history) > 0:
            df_hist = pd.DataFrame(st.session_state.history, columns=['Time','Command','Confidence'])
            st.dataframe(df_hist)
        else:
            st.write('لا توجد سجلات بعد')

elif action == 'PCA Visualize':
    st.header('📈 PCA Visualization (Live point vs training)')
    X_train_local, y_train_local = load_dataset(TRAIN_DIR)
    scaler_local, pca_local, X_pca_local = prepare_pca_for_training(X_train_local)
    df_pca = pd.DataFrame(X_pca_local, columns=['PC1','PC2'])
    df_pca['Label'] = y_train_local
    fig = px.scatter(df_pca, x='PC1', y='PC2', color='Label', title='PCA of Training Data')
    st.plotly_chart(fig, use_container_width=True)

    if st.button('Record sample and show red dot'):
        if model is None or scaler is None:
            st.error('Model/scaler missing')
        else:
            audio = record_audio_streamlit()
            feat = extract_features_from_signal(audio).reshape(1,-1)
            # To map to training PCA space we must use the same pipeline: scaler_local then pca_local
            if feat.shape[1] == X_train_local.shape[1]:
                feat_scaled = scaler_local.transform(feat)
                feat_pca = pca_local.transform(feat_scaled)
                fig.add_scatter(x=[feat_pca[0,0]], y=[feat_pca[0,1]], mode='markers', marker=dict(size=12, color='red'), name='Live Sample')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error('Mismatch in feature dimensionality between training and live sample')

elif action == 'Download Model':
    st.header('⬇️ تحميل النموذج والمقياس')
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        with open(MODEL_PATH, 'rb') as f:
            st.download_button('Download model14.joblib', f, file_name='model14.joblib')
        with open(SCALER_PATH, 'rb') as f:
            st.download_button('Download scaler', f, file_name='model14_scalar.joblib')
    else:
        st.info('Model files not found. Train the model first.')

# ==================================================================================================
# Footer - tips and dependency list
# ==================================================================================================

st.markdown('---')
st.markdown('### تعليمات التشغيل:')
st.markdown('1. تأكد من تثبيت الحزم: `pip install streamlit librosa sounddevice scikit-learn seaborn matplotlib joblib pyserial plotly`')
st.markdown('2. ضع ملفات wav في مجلد `commands/<label>/*.wav` و `Test/<label>/*.wav`')
st.markdown('3. شغّل التطبيق: `streamlit run Streamlit_Voice_Command_Platform_Monolith.py`')
