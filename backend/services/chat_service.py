"""Offline AI Chatbot service wrapping OfflineChatbot with strict RBAC and urgent escalation."""
from __future__ import annotations

import re
import sqlite3
from typing import Any

from offline_ai_chatbot import OfflineChatbot
from backend.security import resolve_authorized_astronaut_id

_PERSONAL_PRONOUNS = re.compile(r"\b(my|mine|i|me|myself|am i|do i|have i|did i)\b", re.IGNORECASE)
_PERSONAL_TOPICS = re.compile(
    r"\b(heart rate|blood pressure|bp|spo2|oxygen|temperature|glucose|sugar|sleep|mood|immune|bone|muscle|"
    r"cortisol|hormone|alert|warning|danger|status|baseline|to ?do|task|profile|weight|height|allerg|"
    r"chest pain|left arm|passed out|fainted|short of breath|dizzy|lightheaded|headache|pain)\b",
    re.IGNORECASE,
)


class ChatService:
    """Thread-safe wrapper around OfflineChatbot using startup-loaded resources."""

    def __init__(self) -> None:
        self.bot = OfflineChatbot()

    def ask(
        self,
        con: sqlite3.Connection,
        user: dict[str, Any] | None,
        question: str,
        requested_astronaut_id: str | None = None,
    ) -> dict[str, Any]:
        """Answer or refuse a user query cleanly while enforcing server-session RBAC."""
        q = str(question or "").strip()
        if not q:
            return {
                "ok": False,
                "refused": True,
                "login_required": False,
                "urgent_escalation": False,
                "intent": "refusal",
                "answer": "Please enter a question about your health telemetry, alerts, or countermeasures.",
                "sources": "none",
            }

        target_aid: str | None = None
        astronaut_name: str | None = None
        surgeon_contact: str | None = None

        if user is not None:
            if user["role"] == "astronaut":
                target_aid = user["astronaut_id"]
            elif user["role"] == "doctor":
                candidate = str(requested_astronaut_id or "").strip().upper()
                if not candidate:
                    first = con.execute(
                        "SELECT astronaut_id FROM doctor_astronaut_links WHERE doctor_user_id = ? AND status = 'approved' ORDER BY astronaut_id ASC LIMIT 1",
                        (int(user["user_id"]),),
                    ).fetchone()
                    if first is None:
                        return {
                            "ok": True,
                            "refused": True,
                            "login_required": False,
                            "urgent_escalation": False,
                            "intent": "refusal",
                            "answer": "No approved astronaut links exist for your doctor account yet.",
                            "sources": "doctor_astronaut_links",
                        }
                    candidate = first["astronaut_id"]
                target_aid = resolve_authorized_astronaut_id(con, user, candidate)

            prof = con.execute(
                "SELECT name, contact_flight_surgeon FROM astronaut_profile WHERE astronaut_id = ?",
                (target_aid,),
            ).fetchone()
            if prof:
                astronaut_name = prof["name"]
                surgeon_contact = prof["contact_flight_surgeon"]
            raw = self.bot.ask(target_aid, q)
        else:
            # Logged-out users: evaluate scope safely using a non-existent placeholder so no DB telemetry leaks;
            # personal "my ..." questions get a clean login prompt card rather than guessing values.
            norm_q = q.lower()
            is_urgent = self.bot.is_urgent_symptom(q, norm_q)
            if is_urgent:
                return {
                    "ok": True,
                    "refused": False,
                    "login_required": False,
                    "urgent_escalation": True,
                    "intent": "alerts",
                    "confidence": 1.0,
                    "answer": (
                        "URGENT ESCALATION: Contact the flight surgeon immediately (Ground Medical Control). "
                        "Stop current exertion and initiate real-time ground medical telemetry review.\n"
                        "Please log in with your crew account if you also want to view your active onboard telemetry alerts."
                    ),
                    "sources": "urgent_symptom_escalation",
                    "did_you_mean": None,
                    "flight_surgeon_contact": "Flight Surgeon Desk (Ground Control)",
                }

            # Check non-personal / out-of-scope questions first (e.g. "What's the temperature on Venus?").
            probe = self.bot.ask("A001", q)
            if probe.get("refused"):
                raw = probe
            elif _PERSONAL_PRONOUNS.search(q) or _PERSONAL_TOPICS.search(q):
                return {
                    "ok": True,
                    "refused": False,
                    "login_required": True,
                    "urgent_escalation": False,
                    "intent": "login_required",
                    "confidence": 1.0,
                    "answer": "Sign in to your astronaut or approved doctor account so I can answer using your real mission telemetry and pre-flight baseline.",
                    "sources": "authentication",
                    "did_you_mean": None,
                }
            else:
                raw = probe

        return {
            "ok": True,
            "astronaut_id": target_aid,
            "astronaut_name": astronaut_name,
            "flight_surgeon_contact": surgeon_contact,
            "login_required": False,
            **raw,
        }
