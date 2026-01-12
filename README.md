# 🎙️ VoiceCare AI
**Intelligent AI-Powered Telecom Service Agent**

VoiceCare AI is a production-grade AI system designed to automate and enhance telecom customer support by combining **Voice AI**, **LLMs**, **RAG**, **Machine Learning**, **CRM integration**, and **Real-time Analytics**.

The system helps telecom providers handle customer complaints efficiently, predict recurring issues, and improve customer satisfaction using data-driven intelligence.

---

## 🚀 Key Features
- 🎤 **Voice-to-Text Complaint Handling** (Whisper)
- 🤖 **AI Service Agent with LLMs & RAG**
- 🧠 **Intelligent Issue Classification (Arabic-focused)**
- 🔮 **Behavioral ML Prediction for Next Customer Issue**
- 📞 **CRM Integration (Customer & Subscription Data)**
- 📊 **Live Analytics Dashboard**
- 🔁 **Self-learning & Retraining Pipeline**
- 🐳 **Dockerized & Production-Ready Deployment**
- 🔐 **Security & Privacy by Design**

---

## 🧱 System Architecture
The system follows a modular, scalable architecture:

- **Frontend**: Gradio (Agent UI + Analytics Dashboard)
- **Voice Layer**: Whisper (Speech-to-Text)
- **AI Layer**:
  - LLM (Gemma)
  - RAG with FAISS + Sentence Transformers
- **ML Layer**:
  - Issue Classification
  - Behavioral Prediction Models
- **Data Layer**:
  - Google Sheets (Cases & Feedback)
  - CRM Dataset (Offline)
- **Deployment**:
  - Docker
  - Hugging Face Spaces

> 📌 Architecture diagram and screenshots are available in `/assets`

---

## 🛠️ Tech Stack
- **Python 3.10**
- **Gradio**
- **PyTorch**
- **Transformers**
- **Sentence-Transformers**
- **FAISS**
- **Whisper**
- **Scikit-learn / XGBoost / LightGBM**
- **Plotly**
- **Google Sheets API**
- **Docker**

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
⚙️ Environment Variables

The following environment variables are required (not included in the repository):
HF_TOKEN=your_huggingface_token
GOOGLE_SERVICE_ACCOUNT='{}'
ADMIN_PASSWORD=******
▶️ Run Locally (Docker)
docker build -t voicecare-ai .
docker run -p 7860:7860 voicecare-ai
🌐 Live Demo
🔗 Hugging Face Space:
https://huggingface.co/spaces/youssefmoustafa172/VoiceCare-AI
👥 Team
Eman Taha
Menna Osama
Mariam Maged
Shorok Mohamed
  
🔐 Security & Privacy

No credentials or secrets are stored in the repository

No customer data is publicly shared

Models and embeddings are excluded from GitHub

📜 License

This project is for educational and demonstration purposes.


---

# 📄 `models/README.md`

```md
# 🧠 Models Directory

This directory contains trained Machine Learning models used for:

- Behavioral prediction of next customer issues
- Issue encoding and pattern learning

## 🚫 Not Included in Repository
For security, size, and best-practice reasons, the following files are **not** included in GitHub:
- `.pkl` model files
- Encoders
- Training artifacts

## 📌 Notes
- Models are trained offline
- Stored securely (e.g., Hugging Face, private storage)
- Loaded dynamically at runtime

This ensures:
✔ Better security  
✔ Cleaner repository  
✔ Production-grade MLOps practices
