# 🤖 Models Directory – VoiceCare AI

This directory contains all **machine learning model artifacts** used by the **Behavioral Prediction Engine** in VoiceCare AI.

The models in this folder enable the system to **predict the next likely customer issue** based on historical behavior, issue sequences, and resolution patterns.

All artifacts are generated through a **controlled training and retraining pipeline** and are designed to be **production-safe and reusable**.

---

## 📁 Stored Artifacts

### 1️⃣ Behavioral Prediction Model
- **intelligent_behavioral_model.pkl**
- Trained using an ensemble-based approach:
  - Random Forest
  - XGBoost
  - LightGBM
- Selected automatically based on **macro recall performance**.

**Purpose**
- Predict the most likely next customer issue.
- Improve proactive service and resolution efficiency.

---

### 2️⃣ Issue Encoder
- **issue_encoder.pkl**
- LabelEncoder mapping telecom issue categories to numeric representations.

**Purpose**
- Ensures consistent issue encoding across training and inference.
- Supports safe inverse transformation of predicted outputs.

---

### 3️⃣ Pattern Encoder
- **pattern_encoder.pkl**
- Encodes customer behavior patterns extracted from historical sequences.

**Purpose**
- Captures recurring customer issue transitions.
- Enhances model awareness of personalized behavior.

---

### 4️⃣ Feature Columns Definition
- **feature_columns.pkl**
- Stores the ordered list of features used during model training.

**Purpose**
- Guarantees feature consistency between training and inference.
- Prevents schema drift during retraining.

---

### 5️⃣ Customer Issue Sequences
- **customer_sequences.pkl**
- Precomputed sequences of customer issues over time.

**Purpose**
- Enables pattern-based feature extraction.
- Supports intelligent fallback prediction logic.

---

## 🔁 Model Lifecycle

1. **Initial Training**
   - Models are trained using labeled historical cases.
   - Features include issue frequency, resolution time, temporal gaps, and behavioral patterns.

2. **Inference**
   - Models are loaded at application startup.
   - Used in real time to predict the next issue for a given customer.

3. **Retraining**
   - Triggered manually via the **Admin Panel**.
   - Uses high-quality feedback samples (accepted predictions + high ratings).
   - Automatically replaces existing model artifacts.

---

## 🧠 Design Principles
- Production-ready model packaging
- Clear separation between training and inference
- Safe fallback logic when models are unavailable
- Compatibility with future MLOps pipelines

---

## 🔐 Security & Versioning
- Model files are excluded from GitHub version control.
- No sensitive training data is stored in this directory.
- Artifacts should be stored securely in production environments (e.g., object storage).

---

📌 **Note**  
In production deployments, this directory can be replaced with a secure model registry or artifact store while preserving the same loading interface.

