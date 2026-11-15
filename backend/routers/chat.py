"""Offline AI Chatbot API route (POST /api/chat)."""
from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from backend.database import get_db
from backend.security import get_session_user

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    question: str
    astronaut_id: str | None = None


@router.post("/chat")
def chat_query(
    payload: ChatRequest,
    request: Request,
    user: dict[str, Any] | None = Depends(get_session_user),
    con: sqlite3.Connection = Depends(get_db),
) -> dict[str, Any]:
    """Wrap OfflineChatbot cleanly with session-enforced RBAC, scope gate, and urgent escalation."""
    chat_service = request.app.state.chat_service
    return chat_service.ask(con, user, payload.question, requested_astronaut_id=payload.astronaut_id)
