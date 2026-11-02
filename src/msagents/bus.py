"""Message bus between agents (direct, by role, or broadcast)."""
from __future__ import annotations

import json

from .db import dumps, now


def send(conn, from_agent: str, to: str, subject: str, body: str,
         kind: str = "info", refs=None, reply_to: int | None = None) -> int:
    cur = conn.execute(
        "INSERT INTO messages(from_agent, to_agent, kind, subject, body, refs, reply_to, created_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (from_agent, to, kind, subject, body, dumps(refs or []), reply_to, now()),
    )
    return cur.lastrowid


def inbox(conn, agent: str, role: str, unread_only: bool = True, limit: int = 20) -> list[dict]:
    rows = conn.execute(
        f"""
        SELECT m.*, r.read_at FROM messages m
        LEFT JOIN message_reads r ON r.message_id = m.id AND r.agent = ?
        WHERE (m.to_agent = ? OR m.to_agent = ? OR m.to_agent = '*') AND m.from_agent != ?
        {"AND r.read_at IS NULL" if unread_only else ""}
        ORDER BY m.id DESC LIMIT ?
        """,
        (agent, agent, f"role:{role}", agent, limit),
    ).fetchall()
    out = []
    for r in rows:
        conn.execute(
            "INSERT OR IGNORE INTO message_reads(message_id, agent, read_at) VALUES (?,?,?)",
            (r["id"], agent, now()),
        )
        out.append({
            "id": r["id"], "from": r["from_agent"], "kind": r["kind"],
            "subject": r["subject"], "body": r["body"],
            "refs": json.loads(r["refs"]), "reply_to": r["reply_to"],
        })
    return list(reversed(out))


def rate(conn, agent: str, message_id: int, rating: float) -> None:
    rating = max(0.0, min(1.0, float(rating)))
    conn.execute(
        "INSERT INTO message_reads(message_id, agent, read_at, rating) VALUES (?,?,?,?) "
        "ON CONFLICT(message_id, agent) DO UPDATE SET rating=excluded.rating",
        (message_id, agent, now(), rating),
    )
