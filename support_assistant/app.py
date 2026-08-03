"""
Module 3 -- Support Assistant, FastAPI wrapper (/support_assistant)

Exposes the LangGraph RAG pipeline as a POST /ask endpoint.

Run (after `python ingest.py` has been run at least once):
    uvicorn app:app --host 0.0.0.0 --port 7860
"""

from fastapi import FastAPI
from pydantic import BaseModel

from rag import AnswerResponse, ask

app = FastAPI(title="Zepto Support Assistant")


class AskRequest(BaseModel):
    query: str


@app.post("/ask", response_model=AnswerResponse)
def ask_endpoint(request: AskRequest) -> AnswerResponse:
    return ask(request.query)


@app.get("/")
def health():
    return {"status": "ok", "service": "zepto-support-assistant"}
