# Zepto Placement Assessment — Project Repository

This repository contains three independent modules for the placement
assessment, each in its own folder with its own README covering install
steps, design decisions, and outputs.

| Module | Path | Marks | Summary |
|---|---|---|---|
| 1 — Data Pipeline | [`/data_pipeline`](data_pipeline/README.md) | 25 | Scrape books.toscrape.com, clean & convert currency, load into a normalized SQLite schema, query with SQL + pandas |
| 2 — Analytics Pipeline | [`/analytics`](analytics/README.md) | 50 | Titanic EDA + data story, then a full 3-classifier + regression modeling pipeline with tuning and imbalance handling |
| 3 — Support Assistant | [`/support_assistant`](support_assistant/README.md) | 25 | A LangGraph-orchestrated RAG chatbot over Zepto policy docs, wrapped in FastAPI, with a deterministic offline mock-LLM mode as the graded baseline |

## A note on this submission's development environment

Two required steps in this repo need outbound internet access that isn't
available in the sandbox this was developed in:

- **Module 1**: the live scrape against `books.toscrape.com`.
- **Module 2**: `sns.load_dataset('titanic')`'s first-run download (from
  `raw.githubusercontent.com`) — this one *was* reachable and the full
  module runs end-to-end as delivered.
- **Module 3**: downloading `sentence-transformers`' `all-MiniLM-L6-v2`
  model weights from `huggingface.co`.

Everywhere a live network call was blocked in development, the surrounding
logic (cleaning, database loading, SQL/pandas queries, LangGraph routing,
retrieval plumbing, Pydantic validation, and the FastAPI endpoint) was
still verified end-to-end using synthetic or substitute data standing in
for the blocked call, so what's here is tested wiring, not just
untested code. Each affected module's own README says exactly what was and
wasn't run live. Run `pip install -r requirements.txt`-equivalent installs
per module and the scripts in order on a machine with normal internet
access to reproduce the live outputs.
