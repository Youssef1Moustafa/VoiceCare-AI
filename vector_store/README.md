# 🧠 Vector Store – VoiceCare AI

This directory contains the **vector database artifacts** used by the **Retrieval-Augmented Generation (RAG)** pipeline in VoiceCare AI.

The vector store enables **semantic search** over historical telecom issues and troubleshooting solutions, allowing the AI Service Agent to retrieve the most relevant context before generating responses.

---

## 📁 Contents

### 1️⃣ FAISS Index
- **faiss.index**
- A FAISS Inner Product (IP) index built from normalized sentence embeddings.

**Purpose**
- Fast similarity search over telecom-related issues and customer messages.
- Retrieves top-k relevant historical cases for grounding LLM responses.

---

### 2️⃣ Metadata Store
- **meta.json**
- Stores metadata corresponding to each vector entry.

**Typical Metadata Includes**
- Issue description
- Customer message
- Issue category
- Troubleshooting steps

**Purpose**
- Provides structured context alongside retrieved vectors.
- Enables traceability and explainability of retrieved results.

---

## 🔎 How It Is Used in the System

1. Incoming customer issue (text or transcribed voice) is **normalized**.
2. A semantic embedding is generated using:
   - **SentenceTransformer: `intfloat/multilingual-e5-base`**
3. The embedding is compared against the FAISS index.
4. Top matching cases above a similarity threshold are retrieved.
5. Retrieved context is passed to the **LLM Formatter** to produce grounded, step-by-step solutions.

---

## ⚙️ Design Details
- Similarity Metric: **Cosine similarity** (via normalized inner product)
- Threshold-based filtering to avoid low-confidence matches
- Cached index loading to reduce startup latency
- Automatic index rebuild if artifacts are missing or corrupted

---

## 🧠 Design Principles
- Low-latency retrieval
- Language-agnostic semantic matching
- Arabic-friendly embeddings
- Hallucination reduction via grounded context

---

## 🔐 Security & Versioning
- Vector index and metadata files are excluded from GitHub.
- No raw customer data is stored directly in vectors.
- Data used for embeddings is sanitized and non-sensitive.
- In production, vector stores should be hosted in secure storage or managed vector databases.

---

## 🔄 Lifecycle Management

- **Index Creation**:  
  Built automatically from the curated knowledge base.
- **Index Loading**:  
  Loaded at application startup for real-time retrieval.
- **Index Rebuild**:  
  Triggered automatically if files are missing or invalid.

---

📌 **Note**  
In enterprise deployments, this vector store can be replaced with a managed vector database (e.g., Pinecone, Weaviate, Milvus) while preserving the same retrieval interface.
