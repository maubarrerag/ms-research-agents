"""Task queue with leases, so interrupted work is resumed rather than lost."""
from __future__ import annotations

import json
import sqlite3

from .db import dumps, now

LEASE_SECONDS = 300
MAX_ATTEMPTS = 5


def create_task(conn, role: str, title: str, details: str = "", refs=None,
                priority: float = 0.0, created_by: str = "system") -> int:
    t = now()
    cur = conn.execute(
        "INSERT INTO tasks(role, title, details, refs, priority, created_by, created_at, updated_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (role, title, details, dumps(refs or []), priority, created_by, t, t),
    )
    return cur.lastrowid


def claim(conn: sqlite3.Connection, agent: str, role: str) -> sqlite3.Row | None:
    """Return the task this agent should work on next.

    Order: (1) a task this agent already owns (it was interrupted, resume it),
    (2) a task of this role whose lease expired, (3) the best pending task.
    """
    t = now()
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            "SELECT * FROM tasks WHERE status='running' AND owner=? ORDER BY id LIMIT 1",
            (agent,),
        ).fetchone()
        if row is None:
            row = conn.execute(
                "SELECT * FROM tasks WHERE role=? AND attempts < ? AND "
                "(status='pending' OR (status='running' AND lease_until < ?)) "
                "ORDER BY priority DESC, id LIMIT 1",
                (role, MAX_ATTEMPTS, t),
            ).fetchone()
        if row is None:
            conn.execute("COMMIT")
            return None
        conn.execute(
            "UPDATE tasks SET status='running', owner=?, lease_until=?, attempts=attempts+1, updated_at=? "
            "WHERE id=?",
            (agent, t + LEASE_SECONDS, t, row["id"]),
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return conn.execute("SELECT * FROM tasks WHERE id=?", (row["id"],)).fetchone()


def heartbeat(conn, task_id: int, agent: str) -> None:
    conn.execute(
        "UPDATE tasks SET lease_until=?, updated_at=? WHERE id=? AND owner=?",
        (now() + LEASE_SECONDS, now(), task_id, agent),
    )


def finish(conn, task_id: int, agent: str, result: str, status: str = "done") -> None:
    conn.execute(
        "UPDATE tasks SET status=?, result=?, lease_until=NULL, updated_at=? WHERE id=? AND owner=?",
        (status, result, now(), task_id, agent),
    )


def release(conn, task_id: int, agent: str) -> None:
    """Graceful stop: keep ownership so the same agent resumes it next start."""
    conn.execute("UPDATE tasks SET lease_until=? WHERE id=? AND owner=?", (now(), task_id, agent))


def requeue_failed(conn) -> int:
    cur = conn.execute(
        "UPDATE tasks SET status='pending', owner=NULL, attempts=0 WHERE status='failed'"
    )
    return cur.rowcount


def refs_of(task) -> list[str]:
    return json.loads(task["refs"] or "[]")
