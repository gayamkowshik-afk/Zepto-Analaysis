"""
Run (after `python ingest.py` has been run at least once):

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
