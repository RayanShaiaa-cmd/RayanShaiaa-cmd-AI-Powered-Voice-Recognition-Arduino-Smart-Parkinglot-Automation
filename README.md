# 🎙️ Smart Parking Voice Control System

> AI-Powered Voice Recognition + Arduino Smart Parking Automation

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![Arduino](https://img.shields.io/badge/Arduino-UNO-green?style=for-the-badge&logo=arduino)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge&logo=streamlit)
![Status](https://img.shields.io/badge/Project-Completed-success?style=for-the-badge)

---

# 📌 Overview

An intelligent smart parking system that combines:

- Artificial Intelligence
- Machine Learning
- Voice Recognition
- Arduino Automation
- Real-Time Audio Processing
- IoT Integration

The system recognizes voice commands using a Machine Learning model and controls a smart parking environment through Arduino hardware.

---

# 🚀 Features

## 🎤 Real-Time Voice Recognition
Recognizes voice commands directly from the microphone.

## 🧠 Machine Learning Pipeline
- MFCC Feature Extraction
- Delta & Delta-Delta Features
- StandardScaler
- Random Forest Classifier

## 🔌 Arduino Integration
Controls:
- LEDs
- Servo Motor
- Buzzer
- LCD Display
- Ultrasonic Sensor

## 🚗 Smart Parking Automation
- Automatic gate control
- Vehicle counting
- Parking monitoring
- Alarm system

## 📊 Interactive Dashboard
Built using Streamlit for:
- Model training
- Live prediction
- Visualization
- Testing

---

# 🏗️ System Architecture

```text
Voice Input
     ↓
Feature Extraction (MFCC)
     ↓
Machine Learning Model
     ↓
Prediction
     ↓
Serial Communication
     ↓
Arduino UNO
     ↓
Hardware Execution
```

---

# 🛠️ Technologies Used

## Programming & AI
- Python
- Scikit-learn
- Librosa
- NumPy
- Pandas

## Dashboard & Visualization
- Streamlit
- Plotly
- Matplotlib
- Seaborn

## Embedded Systems
- Arduino UNO
- PySerial

## Audio Processing
- SoundDevice
- pyttsx3

---

# 📂 Project Structure

```text
Smart-Parking-Voice-Control/
│
├── commands/
│   ├── on/
│   ├── off/
│   ├── unlock/
│   ├── alarm/
│   └── number/
│
├── Test/
│
├── model14.joblib
├── model14_scalar.joblib
│
├── Streamlit_Voice_Command_Platform_Monolith.py
│
├── Arduino_Code/
│   └── SmartParking.ino
│
└── README.md
```

---

# 🎧 Supported Voice Commands

| Command | Function |
|---|---|
| `on` | Turn ON LEDs |
| `off` | Turn OFF LEDs |
| `unlock` | Open parking gate |
| `alarm` | Activate alarm |
| `number` | Display number of cars |

---

# 🔌 Hardware Components

- Arduino UNO
- HC-SR04 Ultrasonic Sensor
- Servo Motor
- LCD 16x2 I2C
- LEDs
- Buzzer
- Resistors
- Jumper Wires

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/smart-parking-voice-control.git

cd smart-parking-voice-control
```

---

## 2️⃣ Install Requirements

```bash
pip install streamlit librosa sounddevice scikit-learn seaborn matplotlib joblib pyserial plotly numpy pandas pyttsx3
```

---

## 3️⃣ Run Streamlit App

```bash
streamlit run Streamlit_Voice_Command_Platform_Monolith.py
```

---

# 📊 Machine Learning Pipeline

## 1. Data Collection
Custom voice commands dataset in WAV format.

## 2. Preprocessing
- Normalization
- 16kHz sampling rate

## 3. Feature Extraction
Extract:
- MFCC
- Delta
- Delta-Delta
- Statistical Features

## 4. Feature Scaling
Using:
```python
StandardScaler()
```

## 5. Model Training
Using:
```python
RandomForestClassifier(
    n_estimators=1000,
    random_state=50
)
```

## 6. Evaluation
- Accuracy
- Confusion Matrix
- Classification Report
- PCA Visualization

---

# 📈 Dashboard Features

The Streamlit dashboard contains:

- Dataset EDA
- Train Model
- Evaluate Test
- Live Prediction
- PCA Visualization
- Download Model

---

# 🚗 Smart Parking Workflow

```text
User Speaks Command
        ↓
AI Predicts Intent
        ↓
Python Sends Serial Command
        ↓
Arduino Receives Instruction
        ↓
Hardware Executes Action
```

---

# 🔮 Future Improvements

- Arabic Voice Recognition
- Deep Learning Models (CNN/LSTM)
- Noise Reduction
- Wake Word Detection
- Mobile Application
- Cloud Deployment
- License Plate Recognition

---

# 🧪 Applications

- Smart Parking Systems
- Smart Cities
- Home Automation
- Security Systems
- AI IoT Platforms

---

# 👨‍💻 Author

## Rayan Shaiaa

Artificial Intelligence Student  
AI & Embedded Systems Developer

Specialized in:
- Machine Learning
- Smart Systems
- Computer Vision
- Embedded AI
- IoT Automation

---

# 📄 License

This project is licensed under the MIT License.

---

# ⭐ Final Note

This project demonstrates a complete integration between:

- Artificial Intelligence
- Machine Learning
- Audio Signal Processing
- Embedded Systems
- IoT Automation

Creating a real-world intelligent smart parking solution controlled entirely by voice commands.