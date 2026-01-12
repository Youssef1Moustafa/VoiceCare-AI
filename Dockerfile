# ================================
# 🐍 Base Image (Slim & Stable)
# ================================
FROM python:3.10-slim

# ================================
# ⚙️ Environment Variables
# ================================
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV TRANSFORMERS_NO_TIKTOKEN=1

# ================================
# 📦 System Dependencies
# ================================
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ================================
# 📂 Working Directory
# ================================
WORKDIR /app

# ================================
# 📄 Install Python Dependencies
# (Layer cached for faster rebuilds)
# ================================
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ================================
# 📦 Copy Application Code
# ================================
COPY . .

# ================================
# 🌐 Expose Gradio Port
# ================================
EXPOSE 7860

# ================================
# 🚀 Run Application
# ================================
CMD ["python", "app.py"]
