# 🎙️ VoiceCare AI – Intelligent Telecom Service Agent 📡🤖

**VoiceCare AI** is a production-grade AI-powered telecom customer service system designed to automate complaint handling, enhance customer experience, and provide actionable analytics for telecom operators.

The platform combines **Voice AI**, **LLMs**, **Retrieval-Augmented Generation (RAG)**, **Machine Learning behavioral prediction**, **CRM integration**, and a **real-time analytics dashboard** into a unified intelligent service agent.

---

## 🚀 Key Features

### 1. 🎤 Voice-Based Complaint Handling
- **Speech-to-Text (Whisper):** Customers can submit complaints via voice or audio files.
- **Arabic-Optimized:** Tuned for Arabic telecom terminology and real customer expressions.
- **Hands-Free Interaction:** Enables faster complaint intake without manual typing.

---

### 2. 🤖 AI Service Agent (LLM + RAG)
- **LLM-Powered Responses:** Uses a large language model to format and present troubleshooting steps clearly.
- **RAG Pipeline:** Retrieves the most relevant historical solutions using **FAISS + semantic embeddings**.
- **Step-by-Step Resolution:** Guides customers through structured troubleshooting flows.

---

### 3. 🧠 Intelligent Issue Classification
Automatically classifies customer complaints into telecom categories such as:
- Internet Down  
- Slow Internet  
- Router Issues  
- Billing Issues  
- Landline Problems  
- Roaming & Offers  

**Confidence Scoring:** Each classification includes a confidence score and source (intent/context).

---

### 4. 🔮 Behavioral Issue Prediction
- **Next-Issue Prediction:** Predicts the customer’s next likely issue based on historical behavior.
- **Pattern Learning:** Leverages customer issue sequences and resolution patterns.
- **ML Models:** Trained using ensemble techniques (**Random Forest, XGBoost, LightGBM**).

---

### 5. 📞 CRM Integration
- Fetches customer data using normalized phone numbers.
- Displays:
  - Customer name  
  - Subscription type  
  - Bundle price  
- Works safely with both **registered and unregistered customers**.

---

### 6. 📊 Live Analytics Dashboard

**Real-time KPIs**
- Total cases  
- Resolution rate  
- Average resolution time  
- Customer satisfaction rating  
- Prediction usage & accuracy  

**Interactive Charts**
- Weekly trends  
- Case status distribution  
- Rating analysis  

Powered by **Plotly + Gradio**.

---

### 7. 🔁 Self-Learning & Retraining
- Collects high-quality feedback samples.
- Supports **admin-triggered retraining** of behavioral models.
- Designed with **MLOps best practices** in mind.

---

### 8. 🐳 Production-Ready Deployment
- Fully **Dockerized**
- Deployed on **Hugging Face Spaces**
- Secure environment-variable-based configuration
- No secrets or customer data stored in the repository

---

## 🏗️ System Architecture

VoiceCare AI follows a **modular, scalable architecture**:

### Frontend
- Gradio (AI Agent UI + Analytics Dashboard)

### Voice Layer
- Whisper (Speech-to-Text)

### AI Layer
- LLM (Gemma)
- RAG (Sentence Transformers + FAISS)

### ML Layer
- Issue Classification
- Behavioral Prediction Models

### Data Layer
- Google Sheets (Cases, Feedback, Analytics)
- CRM Dataset (Offline, Secure)

### Deployment
- Docker
- Hugging Face Spaces

📌 Architecture diagram and UI screenshots are available in `/assets`.

---

## 🛠️ Tech Stack

### Backend & AI
- Python 3.10  
- PyTorch  
- Transformers  
- Sentence-Transformers  
- FAISS  
- Whisper (faster-whisper)  
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
---
##📋 Prerequisites

-Python 3.10+

-Docker

-FFmpeg (for audio processing)

-Google Sheets API access

-Hugging Face account
---
##⚙️ Environment Variables
-HF_TOKEN=your_huggingface_token
-GOOGLE_SERVICE_ACCOUNT='{}'
-ADMIN_PASSWORD=******
-⚠️ All secrets are excluded from the repository.
---
##⚡ Installation & Run (Docker)
docker build -t voicecare-ai .
docker run -p 7860:7860 voicecare-ai
---


