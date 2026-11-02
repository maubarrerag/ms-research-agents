"""Per-agent long-term memory and per-task conversation checkpoints."""
from __future__ import annotations

import json
import re

from .db import dumps, now


# ---- long-term memory ------------------------------------------------------

def remember(conn, agent: str, content: str, kind: str = "note", tags: str = "") -> int:
    cur = conn.execute(
        "INSERT INTO memories(agent, kind, content, tags, created_at) VALUES (?,?,?,?,?)",
        (agent, kind, content, tags, now()),
    )
    return cur.lastrowid


def fts_query(text: str) -> str:
    words = re.findall(r"\w+", text.lower())
    return " OR ".join(f'"{w}"' for w in words[:12]) or '""'


def recall(conn, agent: str, query: str, limit: int = 8) -> list[dict]:
    rows = conn.execute(
        "SELECT m.id, m.kind, m.content, m.tags FROM memories_fts f JOIN memories m ON m.id = f.rowid "
        "WHERE memories_fts MATCH ? AND m.agent = ? ORDER BY rank LIMIT ?",
        (fts_query(query), agent, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def digest(conn, agent: str, limit: int = 12) -> str:
    """Short memory summary put into the system prompt at the start of each task."""
    pinned = conn.execute(
        "SELECT content FROM memories WHERE agent=? AND kind='pinned' ORDER BY id DESC LIMIT 5", (agent,)
    ).fetchall()
    lessons = conn.execute(
        "SELECT content FROM memories WHERE agent=? AND kind='lesson' ORDER BY id DESC LIMIT ?", (agent, limit)
    ).fetchall()
    notes = conn.execute(
        "SELECT content FROM memories WHERE agent=? AND kind='note' ORDER BY id DESC LIMIT 5", (agent,)
    ).fetchall()
    parts = []
    if pinned:
        parts.append("Pinned:\n" + "\n".join(f"- {r['content']}" for r in pinned))
    if lessons:
        parts.append("Lessons learned:\n" + "\n".join(f"- {r['content']}" for r in lessons))
    if notes:
        parts.append("Recent notes:\n" + "\n".join(f"- {r['content']}" for r in notes))
    return "\n\n".join(parts) or "(no memories yet)"


# ---- checkpoints -----------------------------------------------------------

def save_checkpoint(conn, task_id: int, agent: str, system: str, messages: list, step: int) -> None:
    conn.execute(
        "INSERT INTO checkpoints(task_id, agent, system, messages, step, updated_at) VALUES (?,?,?,?,?,?) "
        "ON CONFLICT(task_id) DO UPDATE SET system=excluded.system, messages=excluded.messages, "
        "step=excluded.step, updated_at=excluded.updated_at, agent=excluded.agent",
        (task_id, agent, system, dumps(messages), step, now()),
    )


def load_checkpoint(conn, task_id: int):
    row = conn.execute("SELECT * FROM checkpoints WHERE task_id=?", (task_id,)).fetchone()
    if not row:
        return None
    return {"system": row["system"], "messages": json.loads(row["messages"]), "step": row["step"]}


# ---- idempotent tool results ----------------------------------------------

def cached_tool_result(conn, tool_use_id: str):
    row = conn.execute(
        "SELECT output, is_error FROM tool_calls WHERE tool_use_id=?", (tool_use_id,)
    ).fetchone()
    return (row["output"], bool(row["is_error"])) if row else None


def store_tool_result(conn, tool_use_id, task_id, agent, name, input_, output, is_error) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO tool_calls(tool_use_id, task_id, agent, name, input, output, is_error, created_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (tool_use_id, task_id, agent, name, dumps(input_), output, int(is_error), now()),
    )
