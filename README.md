# 🎙️ VoiceCare AI  
### Intelligent Telecom Service Agent & Analytics Platform 📡🤖

**VoiceCare AI** is a production-grade, AI-powered telecom customer service platform designed to **automate complaint handling**, **enhance customer experience**, and **deliver real-time operational intelligence** for telecom operators.

The system combines **Voice AI**, **Large Language Models (LLMs)**, **Retrieval-Augmented Generation (RAG)**, **behavioral Machine Learning**, **CRM integration**, and a **live analytics dashboard** into a unified, intelligent service agent.

---

## 🚀 Key Capabilities

### 🎤 Voice-First Customer Interaction
- **Speech-to-Text using Whisper (large-v2)**
- Supports **live microphone recording** and **audio file uploads**
- Optimized for **Arabic telecom terminology and real customer language**
- Seamless conversion of voice complaints into structured cases

---

### 🤖 AI Service Agent (LLM + RAG)
- Uses **Gemma LLM** to format and present solutions professionally
- **Retrieval-Augmented Generation (RAG)** with FAISS + semantic embeddings
- Step-by-step troubleshooting guidance
- Strict formatting rules to avoid hallucinations and preserve technical accuracy

---

### 🧠 Intelligent Issue Classification
Automatically categorizes complaints into telecom-specific domains:
- Internet Down  
- Slow Internet  
- Router Issues  
- Billing Issues  
- Landline Problems  
- Roaming & Offers  

Each classification includes:
- **Confidence score**
- **Decision source** (intent-based / context-based / fallback)

---

### 🔮 Behavioral Issue Prediction Engine
- Predicts the **next likely customer issue** based on historical behavior
- Learns from **customer issue sequences**
- Ensemble ML models:
  - Random Forest
  - XGBoost
  - LightGBM
- Confidence-based filtering with intelligent fallback logic

---

### 📞 CRM Integration (Safe & Robust)
- Normalizes Egyptian phone numbers reliably
- Fetches customer profile:
  - Name
  - Subscription type
  - Bundle price
- Works safely with **registered and unregistered customers**
- Designed to handle Excel / Google Sheets data inconsistencies

---

### 📊 Live Analytics Dashboard
A fully integrated **management dashboard** built with **Gradio + Plotly**.

**Key KPIs**
- Total cases
- Resolution rate
- Average resolution time
- Customer satisfaction rating
- Prediction usage & acceptance rate

**Interactive Visuals**
- Weekly case trends
- Case status distribution
- Rating analysis
- Detailed case table with filters

---

### 🔁 Self-Learning & Model Retraining
- Collects **high-quality feedback samples**
- Supports **admin-triggered retraining**
- Retrains behavioral models using real operational data
- Built following **MLOps-ready principles**

---

### 🐳 Production-Ready Deployment
- Fully **Dockerized**
- Deployed on **Hugging Face Spaces**
- Secure configuration via environment variables
- No secrets, credentials, or customer data stored in the repository

---

## 🏗️ System Architecture

![VoiceCare AI Architecture](assets/architecture.png)

**Architecture Layers**
- **Frontend Layer**:  
  AI Service Agent UI + Analytics Dashboard (Gradio)
- **Voice Processing Layer**:  
  Whisper Speech-to-Text
- **AI & Reasoning Layer**:  
  Issue Classification + RAG + LLM Formatter
- **Machine Learning Layer**:  
  Behavioral Prediction Engine
- **Data Layer**:  
  Google Sheets (Cases, Feedback, KPIs) + CRM Dataset
- **Deployment Layer**:  
  Docker + Hugging Face Spaces

---

## 🛠️ Tech Stack

### Backend & AI
- Python 3.10
- PyTorch
- Transformers
- Sentence-Transformers
- FAISS
- faster-whisper
- Scikit-learn
- XGBoost
- LightGBM

### Data & Analytics
- Pandas
- NumPy
- Plotly
- Google Sheets API

### UI & Deployment
- Gradio
- Docker
- Hugging Face Spaces

---

## 📂 Project Structure

```text
VoiceCare-AI/
├── app.py
├── Dockerfile
├── requirements.txt
├── README.md
├── .gitignore
│
├── assets/
│   ├── logo.png
│   ├── architecture.png
│   ├── agent_ui.png
│   └── dashboard.png
│
├── data/
│   └── README.md
│
├── models/
│   └── README.md
│
└── vector_store/
    └── README.md
```
---

## 📋 Prerequisites

-Python 3.10+
-Docker
-FFmpeg (for audio processing)
-Google Sheets API access
-Hugging Face account

---

## ⚙️ Environment Variables

```env
HF_TOKEN=huggingface_token
GOOGLE_SERVICE_ACCOUNT='{}'
ADMIN_PASSWORD=******
```
⚠️ All secrets are managed securely and excluded from version control.

---

## ⚡ Run with Docker

```bash
docker build -t voicecare-ai .
docker run -p 7860:7860 voicecare-ai
```
Access the application:
👉 http://localhost:7860

---

## 🌐 Live Demo
🔗 Hugging Face Space – VoiceCare AI
https://huggingface.co/spaces/youssefmoustafa172/VoiceCare-AI

---

## 👥 Meet the Team

| Name           | Role        |
| -------------- | ----------- |
| Eman Taha      | AI Engineer |
| Menna Osama    | AI Engineer |
| Mariam Maged   | AI Engineer |
| Shorok Mohamed | AI Engineer |

---

## 🔐 Security & Privacy

- No credentials stored in code
- No customer data exposed publicly
- Models and vector indexes excluded from GitHub
- Privacy-first system design

---

## 📄 Project Statement

- This project demonstrates a **production-ready AI-powered telecom service platform**, built to address real-world customer service and operational challenges through intelligent automation, predictive analytics, and scalable system design.











