# 📊 Data Directory – VoiceCare AI

This directory contains all **data resources** used by the VoiceCare AI system, including **knowledge bases**, **CRM datasets**, and **analytics-related inputs**.

All data assets are structured to support **AI reasoning**, **behavioral prediction**, **CRM integration**, and **real-time analytics**, while following privacy-first and production-safe practices.

---

## 📁 Contents Overview

### 1️⃣ Knowledge Base Data
- **telecom_data_egypt_20251225_cleaned.json**
- Contains curated telecom issues and troubleshooting steps.
- Used by the **RAG pipeline (FAISS + Embeddings)**.
- Optimized for **Arabic telecom terminology** and real customer expressions.

**Purpose**
- Semantic retrieval of historical solutions
- Context grounding for LLM responses
- Reducing hallucinations and ensuring technical accuracy

---

### 2️⃣ CRM Dataset
- **customer_subscriptions_2000_final.xlsx**
- Offline CRM snapshot used for:
  - Customer identification
  - Subscription type lookup
  - Bundle price retrieval

**Key Notes**
- Phone numbers are normalized automatically in the pipeline.
- Supports both **registered and unregistered customers**.
- Designed to handle Excel formatting inconsistencies safely.

---

### 3️⃣ Analytics & Case Data (External)
- Live case data is stored in **Google Sheets** (not versioned in this repository).
- Used for:
  - Case tracking
  - Feedback collection
  - KPI computation
  - Dashboard visualization

**Important**
- Google Sheets data is accessed securely via API.
- No live customer data is committed to GitHub.

---

## 🔐 Data Privacy & Security
- No sensitive customer data is stored in the repository.
- All external data access is handled via environment variables.
- CRM and analytics data are treated as **read-only inputs**.
- This directory contains **non-sensitive, sanitized, or sample datasets only**.

---

## ⚙️ Data Usage in the System

| Data Source | Used By |
|------------|--------|
| Knowledge Base JSON | RAG Retrieval + LLM Formatting |
| CRM Excel Dataset | Customer Lookup & Case Enrichment |
| Google Sheets | Case Lifecycle + Analytics Dashboard |

---

## 🧠 Design Principles
- Separation of data from logic
- Production-safe data handling
- Privacy-first architecture
- Scalable to real telecom environments

---

📌 **Note**  
Actual production deployments should replace offline datasets with secure databases or CRM systems while preserving the same data interfaces.
