# Serving image for Hugging Face Spaces (Docker SDK): runs as uid 1000, traffic on port 7860.
FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    MODEL_REPO=hamad470/sms-scam-distilroberta
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user pyproject.toml README.md ./
COPY --chown=user src ./src
RUN pip install --no-cache-dir --user --no-deps .

# Model weights come from the Hub at build time so the container starts warm
COPY --chown=user models ./models
RUN python -c "import os; from huggingface_hub import snapshot_download; \
snapshot_download(os.environ['MODEL_REPO'], local_dir='models/onnx', allow_patterns=['model.onnx', 'tokenizer.json'])"

COPY --chown=user app ./app
ENV MODEL_DIR=/home/user/app/models/onnx

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
