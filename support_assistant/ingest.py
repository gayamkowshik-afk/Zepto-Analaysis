"""
Run once before starting the FastAPI app:
    python ingest.py
"""

import glob
import os

import chromadb
from sentence_transformers import SentenceTransformer

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(HERE, "docs")
CHROMA_DIR = os.path.join(HERE, "chroma_db")
COLLECTION_NAME = "zepto_policies"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def load_documents() -> list[dict]:
    """One chunk per document -- each doc_NN.txt is short (a single policy
    paragraph), so splitting further would only fragment a coherent policy
    statement without adding retrieval value."""
    chunks = []
    for path in sorted(glob.glob(os.path.join(DOCS_DIR, "doc_*.txt"))):
        doc_id = os.path.splitext(os.path.basename(path))[0]  # e.g. "doc_01"
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        chunks.append({"id": doc_id, "text": text})
    return chunks


def build_collection():
    print(f"Loading embedding model '{EMBED_MODEL_NAME}' (sentence-transformers, local, no API key)...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    chunks = load_documents()
    print(f"Loaded {len(chunks)} document chunks from {DOCS_DIR}")

    ids = [c["id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts).tolist()

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"source": doc_id} for doc_id in ids],
    )
    print(f"Stored {len(ids)} embedded chunks in ChromaDB collection "
          f"'{COLLECTION_NAME}' at {CHROMA_DIR}")


if __name__ == "__main__":
    build_collection()
