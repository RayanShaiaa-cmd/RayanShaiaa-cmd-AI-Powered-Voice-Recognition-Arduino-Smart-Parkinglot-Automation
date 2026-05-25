# 🎙️ Smart Parking Voice Control System

> AI-Powered Voice Recognition + Arduino Smart Parking Automation

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge\&logo=python)
![Arduino](https://img.shields.io/badge/Arduino-UNO-green?style=for-the-badge\&logo=arduino)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge\&logo=streamlit)
![IoT](https://img.shields.io/badge/IoT-Smart%20Automation-purple?style=for-the-badge)
![Status](https://img.shields.io/badge/Project-Completed-success?style=for-the-badge)

---

# 📌 Overview

An AI-powered smart parking system that integrates:

* Machine Learning
* Real-Time Voice Recognition
* Embedded Systems
* IoT Automation
* Audio Signal Processing
* Arduino Hardware Control

The project enables users to control a smart parking environment entirely through voice commands.
A Machine Learning model processes spoken commands, predicts user intent, and sends instructions directly to an Arduino-based hardware system.

---

# 🚀 Core Features

## 🎤 Real-Time Voice Recognition

* Live microphone input processing
* Real-time command prediction
* Confidence score estimation
* Interactive voice response system

---

## 🧠 Advanced Machine Learning Pipeline

### Feature Engineering

* MFCC Extraction
* Delta Features
* Delta-Delta Features
* Statistical Audio Features

### Data Processing

* Audio Normalization
* Feature Scaling using `StandardScaler`
* WAV Dataset Handling

### Classification Model

* Random Forest Classifier
* High-dimensional feature representation
* Multi-command voice classification

---

## 🔌 Arduino Smart Automation

The Arduino subsystem controls:

* Servo Motor (Parking Gate)
* LEDs
* Buzzer Alarm
* LCD Display
* Ultrasonic Sensor

Through serial communication between Python and Arduino UNO.

---

## 🚗 Smart Parking Functionalities

* Automatic parking gate control
* Parking occupancy monitoring
* Vehicle counting system
* Alarm activation
* Smart command execution

---

## 📊 Interactive AI Dashboard

Built using Streamlit.

### Dashboard Modules

* Dataset EDA
* Model Training
* Model Evaluation
* Live Voice Prediction
* PCA Visualization
* Model Download System

---

# 🏗️ System Architecture

```text
User Voice Input
        ↓
Audio Recording
        ↓
MFCC Feature Extraction
        ↓
Feature Scaling
        ↓
Random Forest Model
        ↓
Command Prediction
        ↓
Serial Communication
        ↓
Arduino UNO
        ↓
Hardware Execution
```

---

# 🛠️ Technologies Used

## Artificial Intelligence & Machine Learning

* Python
* Scikit-learn
* Librosa
* NumPy
* Pandas

---

## Data Visualization & Dashboard

* Streamlit
* Plotly
* Matplotlib
* Seaborn

---

## Audio Processing

* SoundDevice
* pyttsx3

---

## Embedded Systems & IoT

* Arduino UNO
* PySerial

---

# 📂 Project Structure

```text
Smart-Parking-Voice-Control/
│
├── project_images/
│   ├── electrical_connection/
│   ├── brochure/
│   ├── infographic/
│   ├── real_project_audio_1.jpg
│   └── real_project_audio_2.jpg
│
├── secondary_files/
│   ├── voice_generator.py
│   ├── model_loader.py
│   └── arduino_codes/
│       └── SmartParking.ino
│
├── training_voice/
│   ├── on/
│   ├── off/
│   ├── unlock/
│   ├── alarm/
│   └── number/
│
├── Test/
│   ├── on/
│   ├── off/
│   ├── unlock/
│   ├── alarm/
│   └── number/
│
├── models/
│   ├── model14.joblib
│   └── model14_scalar.joblib
│
├── web.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

# 🎧 Supported Voice Commands

| Voice Command | System Action              |
| ------------- | -------------------------- |
| `on`          | Turn ON LEDs               |
| `off`         | Turn OFF LEDs              |
| `unlock`      | Open parking gate          |
| `alarm`       | Activate security alarm    |
| `number`      | Display number of vehicles |

---

# 🔌 Hardware Components

* Arduino UNO
* HC-SR04 Ultrasonic Sensor
* Servo Motor
* LCD 16x2 I2C
* LEDs
* Buzzer
* Resistors
* Jumper Wires

---

# ⚙️ Installation & Setup

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/Smart-Parking-Voice-Control.git

cd Smart-Parking-Voice-Control
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install streamlit librosa sounddevice scikit-learn seaborn matplotlib joblib pyserial plotly numpy pandas pyttsx3
```

---

## 3️⃣ Run Streamlit Application

```bash
streamlit run web.py
```

---

# 🧠 Machine Learning Workflow

## 1. Dataset Collection

Custom WAV voice command dataset collected for each command category.

---

## 2. Audio Preprocessing

* Sampling Rate: 16kHz
* Audio normalization
* Signal preparation

---

## 3. Feature Extraction

The system extracts:

* MFCC Features
* Delta Features
* Delta-Delta Features
* Mean / Std / Min / Max Statistics

---

## 4. Feature Scaling

The extracted features are normalized using:

```python
StandardScaler()
```

---

## 5. Model Training

The classification model uses:

```python
RandomForestClassifier(
    n_estimators=1000,
    random_state=50
)
```

---

## 6. Model Evaluation

Evaluation Metrics:

* Accuracy Score
* Confusion Matrix
* Classification Report
* PCA Visualization

---

# 📊 Streamlit Dashboard Modules

## 📈 Dataset EDA

Visual analysis of:

* Audio duration
* Energy distribution
* MFCC heatmaps
* Dataset balance

---

## 🧠 Train Model

* Train RandomForest model
* Save scaler & classifier
* Real-time training status

---

## 🔍 Evaluate Test Dataset

* Accuracy calculation
* Confusion matrix visualization
* Classification performance report

---

## 🎙️ Live Voice Prediction

* Real-time microphone recording
* Command prediction
* Confidence estimation
* Arduino command execution

---

## 📉 PCA Visualization

* Feature-space visualization
* Live sample mapping
* Command clustering analysis

---

## ⬇️ Download Model

Download:

* Trained model
* Feature scaler

---

# 🔄 Smart Parking Workflow

```text
User Speaks Command
        ↓
Microphone Captures Audio
        ↓
AI Extracts Audio Features
        ↓
Machine Learning Predicts Command
        ↓
Python Sends Serial Instruction
        ↓
Arduino Receives Command
        ↓
Hardware Executes Action
```

---

# 📁 About Project Folders

## 📂 project_images

Contains:

* Electrical wiring diagrams
* Project brochure designs
* Infographics
* Real recorded project audio demonstrations

---

## 📂 secondary_files

Contains:

* Voice generation scripts
* Model loading utilities
* Arduino source codes

---

## 📂 training_voice

Contains:

* Voice datasets used for training
* Organized by command labels

---

# 🔮 Future Improvements

* Arabic Voice Recognition
* CNN/LSTM Deep Learning Models
* Noise Reduction Pipeline
* Wake Word Detection
* Mobile Application
* Cloud Deployment
* License Plate Recognition
* Edge AI Deployment

---

# 🧪 Real-World Applications

* Smart Parking Systems
* Smart City Infrastructure
* Voice-Controlled Automation
* Embedded AI Systems
* Security & Surveillance
* IoT Control Platforms

---

# 👨‍💻 Author

## Rayan Shaiaa

Artificial Intelligence Student
AI & Embedded Systems Developer

### Specializations

* Machine Learning
* Embedded AI
* Computer Vision
* IoT Automation
* Smart Systems Engineering

---

# 📄 License

This project is licensed under the MIT License.

---

# ⭐ Final Statement

This project demonstrates a complete real-world integration between:

* Artificial Intelligence
* Machine Learning
* Audio Signal Processing
* Embedded Systems
* IoT Automation

Delivering a fully interactive AI-powered smart parking solution controlled entirely through voice commands.