# 🎙️ VoiceCare AI – Intelligent Telecom Service Agent 📡🤖

**VoiceCare AI** is a production-grade, AI-powered telecom customer service platform designed to automate complaint handling, improve customer experience, and deliver actionable operational insights for telecom operators.

The system unifies **Voice AI**, **Large Language Models (LLMs)**, **Retrieval-Augmented Generation (RAG)**, **Machine Learning behavioral prediction**, **CRM integration**, and a **real-time analytics dashboard** into a single intelligent service agent.

---

## 🚀 Key Features

### 1. 🎤 Voice-Based Complaint Handling
- **Speech-to-Text (Whisper):** Customers can submit complaints via live voice recording or audio files.
- **Arabic-Optimized Processing:** Tuned for Arabic telecom terminology and real customer language.
- **Hands-Free Interaction:** Reduces friction and accelerates complaint intake.

---

### 2. 🤖 AI Service Agent (LLM + RAG)
- **LLM-Powered Responses:** Uses a large language model to format and present troubleshooting steps clearly and professionally.
- **RAG Pipeline:** Retrieves the most relevant historical solutions using **FAISS + semantic embeddings**.
- **Step-by-Step Resolution Flow:** Guides customers through structured troubleshooting processes.

---

### 3. 🧠 Intelligent Issue Classification
Automatically classifies customer complaints into core telecom categories:
- Internet Down  
- Slow Internet  
- Router Issues  
- Billing Issues  
- Landline Problems  
- Roaming & Offers  

Each classification includes:
- **Confidence score**
- **Decision source** (intent-based / context-based)

---

### 4. 🔮 Behavioral Issue Prediction
- **Next-Issue Prediction:** Predicts the most likely next issue based on historical customer behavior.
- **Pattern Learning:** Learns from customer issue sequences and resolution history.
- **ML Models:** Ensemble-based approach using **Random Forest, XGBoost, and LightGBM**.

---

### 5. 📞 CRM Integration
- Customer lookup via **normalized phone numbers**.
- Displays:
  - Customer name  
  - Subscription type  
  - Bundle price  
- Supports both **registered and unregistered customers** safely.

---

### 6. 📊 Live Analytics Dashboard
Provides real-time operational insights:

**Key Performance Indicators**
- Total cases  
- Resolution rate  
- Average resolution time  
- Customer satisfaction rating  
- Prediction usage & accuracy  

**Interactive Visualizations**
- Weekly case trends  
- Case status distribution  
- Rating analysis  

Built using **Plotly + Gradio**.

---

### 7. 🔁 Self-Learning & Retraining
- Collects **high-quality feedback samples**.
- Supports **admin-triggered retraining** of behavioral ML models.
- Designed following **MLOps best practices**.

---

### 8. 🐳 Production-Ready Deployment
- Fully **Dockerized**
- Deployed on **Hugging Face Spaces**
- Secure configuration using environment variables
- No secrets or customer data stored in the repository

---

## 🏗️ System Architecture

VoiceCare AI is designed using a **modular, scalable, and production-oriented architecture**.

### High-Level Architecture Flow

```text
Customer
   │
   ▼
Voice / Text Input
   │
   ▼
Speech-to-Text (Whisper)
   │
   ▼
Issue Normalization & Classification
   │
   ├──► CRM Lookup (Customer Data)
   │
   ▼
RAG Retrieval (FAISS + Embeddings)
   │
   ▼
LLM Formatter (Gemma)
   │
   ▼
Solution Response
   │
   ├──► Feedback & Rating
   │
   └──► Behavioral Prediction Engine
                │
                ▼
        Next-Issue Prediction
---

## 📋 Prerequisites

- Python 3.10+
- Docker
- FFmpeg (for audio processing)
- Google Sheets API access
- Hugging Face account
