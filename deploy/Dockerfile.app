FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.11 python3-pip git curl \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir coqui-tts

COPY vaani ./vaani
COPY config ./config
COPY scripts ./scripts

ENV VAANI_MODELS_DIR=/app/models
ENV VAANI_LLM_BASE_URL=http://brain:8000/v1

EXPOSE 8765
CMD ["uvicorn", "vaani.main:app", "--host", "0.0.0.0", "--port", "8765", "--ws-ping-interval", "20"]