from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any
from src.boardgame_agents.rag.rag_oop import RAGService, ChatRequest, ChatResponse

app = FastAPI()
rag_service: RAGService | None = None


@app.on_event("startup")
def on_startup() -> None:
    """
    Build the RAG chain once when the app starts.
    This is where all the heavy initialization happens.
    """
    global rag_service
    rag_service = RAGService()


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest) -> ChatResponse:
    """
    POST /chat
    Body: {"user_id": "...", "message": "..."}
    Returns: {"answer": "..."}
    """
    if rag_service is None:
        # Should never happen if startup runs correctly
        raise HTTPException(
            status_code=500, detail="RAG service not initialized")

    answer = rag_service.chat(req.user_id, req.message)
    return ChatResponse(answer=answer)

# Optionally, a simple health endpoint


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
