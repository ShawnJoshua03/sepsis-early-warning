FROM python:3.13-slim

# Hugging Face Spaces runs containers as uid 1000 and routes to port 7860.
RUN useradd -m -u 1000 user
USER user
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/src PORT=7860 PATH=/home/user/.local/bin:$PATH
WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
COPY --chown=user src/ src/
COPY --chown=user artifacts/model.joblib artifacts/metrics.json artifacts/

EXPOSE 7860
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT} --app-dir src
