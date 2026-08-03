"""
Module 3 -- Support Assistant, core RAG pipeline (/support_assistant)

A LangGraph StateGraph with 3 nodes (classify_intent, retrieve_and_answer,
direct_answer) and a conditional edge that routes between them, wrapping a
ChromaDB + sentence-transformers retrieval step.

LLM calls are gated behind the MOCK_LLM environment variable:
  - MOCK_LLM unset, or "1" (DEFAULT, REQUIRED GRADED BASELINE): fully
    deterministic, rule-based logic. No network call to any LLM provider.
  - MOCK_LLM="0" (OPTIONAL, UNGRADED extension): calls a real LLM (Groq's
    free tier by default) using the structured prompt template below. This
    path is not required and must not affect the required MOCK_LLM=1 output.
"""

import os
from typing import List, Optional, TypedDict

import chromadb
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, ValidationError
from sentence_transformers import SentenceTransformer

HERE = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(HERE, "chroma_db")
COLLECTION_NAME = "zepto_policies"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

POLICY_KEYWORDS = [
    "delivery", "return", "refund", "membership", "tracking", "cancel",
    "gift card", "support hours",
]

# ---------------------------------------------------------------------------
# Structured prompt template (role-context-task-format-length skeleton)
# Used only by the optional MOCK_LLM=0 real-LLM extension.
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """ROLE: You are a helpful customer support assistant for Zepto, a quick-commerce grocery delivery app.

CONTEXT: Use only the following retrieved policy excerpt to answer the customer's question:
---
{context}
---

TASK: Answer the customer's question below using only the information in the CONTEXT above.

NEGATIVE CONSTRAINT: Do not answer using information not present in the provided context. If the context does not contain the answer, say you don't have that information rather than guessing.

FORMAT: Respond with a single short paragraph in plain language, no markdown, no bullet points.

LENGTH: 1-3 sentences.

FEW-SHOT EXAMPLE:
Context: "Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation..."
Question: "How fast is Zepto delivery?"
Answer: "Zepto typically delivers within 10 to 30 minutes of order confirmation, depending on your delivery zone and current order volume."

Customer question: {question}
Answer:"""


# ---------------------------------------------------------------------------
# Pydantic output schema
# ---------------------------------------------------------------------------

class AnswerResponse(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class GraphState(TypedDict):
    query: str
    intent: Optional[str]           # "policy_question" | "general_question"
    retrieved_chunks: Optional[List[dict]]
    response: Optional[dict]        # dict matching AnswerResponse fields


# ---------------------------------------------------------------------------
# Retrieval backend (embedding model + Chroma collection), lazily loaded
# ---------------------------------------------------------------------------

_embedder = None
_collection = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def is_mock_mode() -> bool:
    return os.environ.get("MOCK_LLM", "1") != "0"


def call_real_llm(prompt: str) -> str:
    """Optional, ungraded extension: call Groq's free-tier API.
    Requires GROQ_API_KEY to be set as an environment variable. Never called
    when MOCK_LLM is left at its default (mock mode)."""
    import requests
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "MOCK_LLM=0 requires a GROQ_API_KEY environment variable "
            "(get a free key at console.groq.com)."
        )
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Node 1: classify_intent
# ---------------------------------------------------------------------------

def classify_intent(state: GraphState) -> GraphState:
    query = state["query"]

    if is_mock_mode():
        # keyword heuristic, graded baseline -- no LLM call
        lowered = query.lower()
        intent = "policy_question" if any(kw in lowered for kw in POLICY_KEYWORDS) else "general_question"
    else:
        # optional extension: ask the real LLM to classify
        classify_prompt = (
            "Classify the following customer question as exactly one word, "
            "either 'policy_question' (about Zepto delivery, returns, refunds, "
            "membership, tracking, cancellation, gift cards, or support hours) "
            f"or 'general_question' (anything else).\n\nQuestion: {query}\n\nClassification:"
        )
        raw = call_real_llm(classify_prompt).strip().lower()
        intent = "policy_question" if "policy" in raw else "general_question"

    return {**state, "intent": intent}


# ---------------------------------------------------------------------------
# Node 2: retrieve_and_answer (policy_question branch)
# ---------------------------------------------------------------------------

def retrieve_and_answer(state: GraphState) -> GraphState:
    query = state["query"]

    # Retrieval always runs for real, in both mock and real-LLM modes --
    # embedding + ChromaDB cosine similarity need no API key and no network call.
    embedder = get_embedder()
    collection = get_collection()
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=3)

    chunk_ids = results["ids"][0]
    chunk_texts = results["documents"][0]
    retrieved = [{"id": cid, "text": text} for cid, text in zip(chunk_ids, chunk_texts)]

    top_chunk_snippet = retrieved[0]["text"][:200] if retrieved else ""

    if is_mock_mode():
        answer_text = f"Based on the retrieved context: {top_chunk_snippet}"
        confidence = 1.0
    else:
        prompt = PROMPT_TEMPLATE.format(context=top_chunk_snippet, question=query)
        answer_text = call_real_llm(prompt)
        confidence = 0.9  # heuristic confidence for the real-LLM path

    response = {
        "answer": answer_text,
        "sources": [c["id"] for c in retrieved],
        "confidence": confidence,
    }
    return {**state, "retrieved_chunks": retrieved, "response": response}


# ---------------------------------------------------------------------------
# Node 3: direct_answer (general_question branch)
# ---------------------------------------------------------------------------

def direct_answer(state: GraphState) -> GraphState:
    query = state["query"]

    if is_mock_mode():
        answer_text = "I can only answer questions about Zepto policies right now."
        confidence = 1.0
    else:
        prompt = (
            "You are a helpful assistant for the Zepto app. Answer the "
            f"following question briefly:\n\n{query}"
        )
        answer_text = call_real_llm(prompt)
        confidence = 0.7

    response = {"answer": answer_text, "sources": [], "confidence": confidence}
    return {**state, "response": response}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_by_intent(state: GraphState) -> str:
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {"retrieve_and_answer": "retrieve_and_answer", "direct_answer": "direct_answer"},
    )
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)

    return graph.compile()


def validate_response(raw: dict, retries_left: int = 2) -> AnswerResponse:
    """In mock mode this always validates on the first try, since the schema
    is populated deterministically by our own code. The retry loop exists for
    the optional real-LLM path, where a raw LLM response could fail schema
    validation and gets up to 2 corrective retries before returning a clearly
    marked error response."""
    try:
        return AnswerResponse(**raw)
    except ValidationError as e:
        if retries_left > 0 and not is_mock_mode():
            # Optional extension: ask the LLM to correct its own malformed output
            correction_prompt = (
                f"Your previous output failed schema validation with error: {e}. "
                f"Please respond again, strictly as JSON with fields "
                f"'answer' (string), 'sources' (list of strings), and "
                f"'confidence' (float 0-1)."
            )
            corrected_raw_text = call_real_llm(correction_prompt)
            import json
            try:
                corrected = json.loads(corrected_raw_text)
                return validate_response(corrected, retries_left - 1)
            except Exception:
                return validate_response(raw, retries_left - 1)
        return AnswerResponse(
            answer="Error: response failed schema validation after retries.",
            sources=[], confidence=0.0,
        )


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def ask(query: str) -> AnswerResponse:
    graph = get_graph()
    result = graph.invoke({"query": query, "intent": None, "retrieved_chunks": None, "response": None})
    return validate_response(result["response"])
