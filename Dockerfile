FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

USER root
WORKDIR /app

RUN apt-get update && apt-get install -y python3 python3-pip git && rm -rf /var/lib/apt/lists/*

RUN pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly

RUN pip install --no-cache-dir fastapi uvicorn PyMuPDF Pillow einops addict easydict img2pdf numpy python-multipart

COPY DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm/ ./DeepSeek-OCR-vllm/
COPY custom_*.py ./DeepSeek-OCR-vllm/
COPY start_server.py .

ENV PYTHONPATH="/app/DeepSeek-OCR-vllm:${PYTHONPATH}"
EXPOSE 8000
ENTRYPOINT ["python3", "/app/start_server.py"]