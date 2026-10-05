# Hugging Face Spaces (Docker SDK) runs containers as uid 1000 and routes traffic to port 7860.
FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    NLTK_DATA=/home/user/nltk_data \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt \
    && python -m nltk.downloader -q -d "$NLTK_DATA" stopwords

COPY --chown=user pyproject.toml README.md ./
COPY --chown=user src ./src
RUN pip install --no-cache-dir --user --no-deps .

COPY --chown=user app ./app
COPY --chown=user models ./models
ENV MODEL_DIR=/home/user/app/models

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
