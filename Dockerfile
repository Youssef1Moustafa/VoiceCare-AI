FROM python:3.10-slim-bookworm

# ===============================
# System Dependencies
# ===============================
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libavcodec-dev \
    libavformat-dev \
    libavdevice-dev \
    libavfilter-dev \
    libswscale-dev \
    libswresample-dev \
    pkg-config \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ===============================
# Environment
# ===============================
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

WORKDIR /app

# ===============================
# Python Dependencies (CRITICAL)
# ===============================
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ===============================
# App Files
# ===============================
COPY . .

RUN mkdir -p /app/models /app/data /app/vector_store

EXPOSE 7860

CMD ["python", "app.py"]