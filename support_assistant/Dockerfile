FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the ChromaDB collection at image build time so the container is
# ready to serve immediately (embeddings/model download need network access
# at build time; no API key or account is required, since sentence-transformers
# and ChromaDB both run locally).
RUN python ingest.py

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
