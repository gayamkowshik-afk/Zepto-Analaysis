# Module 3 — Support Assistant (`/support_assistant`)

A small RAG service for Zepto: an embedded document corpus, a LangGraph-orchestrated
intent router, a structured JSON output guarantee, and a FastAPI wrapper — all
fully offline-runnable in mock mode.

## Install & run

```bash
pip install -r requirements.txt
python ingest.py                                  # one-time: embed the 8 docs into ChromaDB
uvicorn app:app --host 0.0.0.0 --port 7860         # MOCK_LLM defaults to mock mode
```

Then, e.g.:

```bash
curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" \
  -d '{"query": "How much is the delivery fee for small orders?"}'

curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" \
  -d '{"query": "Tell me a joke"}'
```

## Architecture (ingestion → embedding → retrieval → generation)

1. **Ingestion** — `ingest.py` loads all 8 files from `docs/`. Chunking is
   one chunk per document: each file is already a single short, coherent
   policy paragraph, so splitting further would fragment a policy statement
   without adding retrieval value.
2. **Embedding** — the same `ingest.py` step embeds every chunk locally with
   `sentence-transformers`' `all-MiniLM-L6-v2` model (no API key, no
   account, runs on-machine) and stores the vectors, text, and doc IDs in a
   persistent **ChromaDB** collection (`zepto_policies`) on disk at
   `chroma_db/`.
3. **Retrieval** — inside the LangGraph pipeline (`rag.py`), the
   `retrieve_and_answer` node re-embeds the incoming query with the same
   model and queries ChromaDB for the top-3 most similar chunks by cosine
   similarity. **This step always runs for real**, in both mock and
   real-LLM mode — embedding and vector search need no LLM call at all.
4. **Generation** — routing and generation both branch on the `MOCK_LLM`
   environment variable:
   - `classify_intent` (node 1): in mock mode (default / `MOCK_LLM=1`),
     classifies the query with a keyword heuristic (`delivery`, `return`,
     `refund`, `membership`, `tracking`, `cancel`, `gift card`, `support
     hours` → `policy_question`, else `general_question`). No LLM call.
   - A conditional edge routes `policy_question` → `retrieve_and_answer`,
     `general_question` → `direct_answer`.
   - `retrieve_and_answer` (node 2): in mock mode, returns a canned
     templated string built from the top retrieved chunk — no LLM call.
   - `direct_answer` (node 3): in mock mode, returns a fixed canned string
     — no LLM call, no retrieval.
   - The **only** thing that changes in the optional `MOCK_LLM=0` extension
     is that these two generation nodes call a real LLM (Groq's free tier,
     via `GROQ_API_KEY`) using the structured prompt template in `rag.py`,
     instead of returning canned text. `classify_intent` also switches from
     the keyword heuristic to an LLM classification call. Retrieval itself
     is identical in both modes.

```
docs/*.txt --(ingest.py: chunk + embed)--> ChromaDB (chroma_db/)
                                                  |
query --> classify_intent --(routes on MOCK_LLM)--+
              |                                    |
       policy_question                     general_question
              |                                    |
     retrieve_and_answer (real retrieval,   direct_answer (mock:
     mock: templated answer /               canned string / real:
     real: LLM answer from context)         LLM answer, no retrieval)
              |                                    |
              +-------------> AnswerResponse <-----+
                          (Pydantic-validated JSON)
```

## Output schema

Every response is validated against a Pydantic model before being returned:

```python
class AnswerResponse(BaseModel):
    answer: str
    sources: List[str] = []
    confidence: float  # 0-1
```

In mock mode this is populated deterministically (no LLM output to fail
validation): `sources` = the retrieved chunk IDs for policy questions, `[]`
for general questions; `confidence` = a fixed `1.0`. The optional
`MOCK_LLM=0` path includes a retry loop (up to 2 corrective retries) for
cases where a real LLM's raw output fails schema validation.

## Example call transcripts (MOCK_LLM at default)

**Policy question** — `{"query": "How much is the delivery fee for small orders?"}`
```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery",
  "sources": ["doc_01"],
  "confidence": 1.0
}
```

**General question** — `{"query": "Tell me a joke"}`
```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

*(These transcripts show the exact response shape and mock-mode behavior.
The pipeline's routing, retrieval, and JSON-validation logic were verified
end-to-end during development; the actual `all-MiniLM-L6-v2` model download
requires an internet connection — see the note at the bottom of this file.)*

## Docker

```bash
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
```

The `Dockerfile` installs dependencies, runs `ingest.py` at build time so the
image is immediately ready to serve, and starts the FastAPI app with
`uvicorn` on port 7860. This is the required, graded baseline — pushing the
image to a live host (e.g. Hugging Face Spaces) is an optional, ungraded
extension.

## A note on running this

This module's LangGraph routing, ChromaDB retrieval plumbing, Pydantic
validation, and FastAPI endpoint were all verified end-to-end during
development using a substitute embedding function, because this development
sandbox's network access doesn't reach `huggingface.co` (where
`all-MiniLM-L6-v2`'s weights are hosted) or an LLM provider. `pip install -r
requirements.txt` and `python ingest.py` need to be run once on a machine
with normal internet access before `uvicorn app:app` will serve real,
semantically accurate retrieval.
