"""SQLite storage shared by all agents.

One file (data/lab.db) holds the message bus, the shared knowledge base
("blackboard"), the task queue, per-agent memory, checkpoints and the reward
ledger. WAL mode lets several agent processes read and write at once.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

DATA_DIR = Path(os.environ.get("MSAGENTS_DATA", Path.cwd() / "data"))
DB_PATH = DATA_DIR / "lab.db"

SCHEMA = """
-- Task queue. A task is claimed with a lease; if the agent dies the lease
-- expires and the task can be resumed (by the same agent, from its checkpoint).
CREATE TABLE IF NOT EXISTS tasks (
    id            INTEGER PRIMARY KEY,
    role          TEXT NOT NULL,            -- which role should do it
    title         TEXT NOT NULL,
    details       TEXT NOT NULL DEFAULT '',
    refs          TEXT NOT NULL DEFAULT '[]', -- JSON list of artifact ids
    status        TEXT NOT NULL DEFAULT 'pending', -- pending|running|done|failed
    owner         TEXT,                     -- agent name holding the lease
    lease_until   REAL,
    attempts      INTEGER NOT NULL DEFAULT 0,
    priority      REAL NOT NULL DEFAULT 0,
    created_by    TEXT NOT NULL DEFAULT 'system',
    result        TEXT,
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS tasks_by_role ON tasks(role, status, priority);

-- Conversation checkpoint per task: the full message list, saved after
-- every model turn and every tool result.
CREATE TABLE IF NOT EXISTS checkpoints (
    task_id     INTEGER PRIMARY KEY REFERENCES tasks(id),
    agent       TEXT NOT NULL,
    system      TEXT NOT NULL,
    messages    TEXT NOT NULL,     -- JSON
    step        INTEGER NOT NULL,
    updated_at  REAL NOT NULL
);

-- Tool calls already executed, keyed by tool_use_id. If the process crashes
-- after running a tool but before the checkpoint, the resumed run reuses the
-- stored result instead of repeating the side effect.
CREATE TABLE IF NOT EXISTS tool_calls (
    tool_use_id TEXT PRIMARY KEY,
    task_id     INTEGER,
    agent       TEXT NOT NULL,
    name        TEXT NOT NULL,
    input       TEXT NOT NULL,
    output      TEXT NOT NULL,
    is_error    INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL
);

-- Long-term memory: notes and lessons each agent keeps across tasks.
CREATE TABLE IF NOT EXISTS memories (
    id          INTEGER PRIMARY KEY,
    agent       TEXT NOT NULL,
    kind        TEXT NOT NULL DEFAULT 'note',  -- note|lesson|pinned
    content     TEXT NOT NULL,
    tags        TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(content, tags, content='memories', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, tags) VALUES (new.id, new.content, new.tags);
END;

-- Message bus. to_agent can be an agent name, a role ('role:analyst') or '*'.
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY,
    from_agent  TEXT NOT NULL,
    to_agent    TEXT NOT NULL,
    kind        TEXT NOT NULL DEFAULT 'info',  -- info|request|reply
    subject     TEXT NOT NULL,
    body        TEXT NOT NULL,
    refs        TEXT NOT NULL DEFAULT '[]',
    reply_to    INTEGER REFERENCES messages(id),
    created_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS message_reads (
    message_id  INTEGER NOT NULL,
    agent       TEXT NOT NULL,
    read_at     REAL NOT NULL,
    rating      REAL,                -- recipient's helpfulness rating 0..1
    PRIMARY KEY (message_id, agent)
);

-- Shared knowledge base. Every artifact has a stable id like F12, H3, E7, R7, V2.
-- kind: finding|dataset|hypothesis|experiment|result|review
CREATE TABLE IF NOT EXISTS artifacts (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    author      TEXT NOT NULL,
    title       TEXT NOT NULL,
    body        TEXT NOT NULL,       -- JSON payload, schema depends on kind
    status      TEXT NOT NULL DEFAULT 'open',
    locked_hash TEXT,                -- set for pre-registered experiments
    created_at  REAL NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS artifacts_fts USING fts5(id UNINDEXED, title, body, content='artifacts', content_rowid='rowid');
CREATE TRIGGER IF NOT EXISTS artifacts_ai AFTER INSERT ON artifacts BEGIN
    INSERT INTO artifacts_fts(rowid, id, title, body) VALUES (new.rowid, new.id, new.title, new.body);
END;

-- Citation graph: who built on whose work. Used for collaboration credit.
CREATE TABLE IF NOT EXISTS citations (
    from_id     TEXT NOT NULL,
    to_id       TEXT NOT NULL,
    PRIMARY KEY (from_id, to_id)
);

-- Reward ledger. Append-only; scores are sums over it.
CREATE TABLE IF NOT EXISTS rewards (
    id          INTEGER PRIMARY KEY,
    agent       TEXT NOT NULL,
    channel     TEXT NOT NULL,       -- collaboration|testing
    amount      REAL NOT NULL,
    reason      TEXT NOT NULL,
    ref         TEXT,
    dedupe_key  TEXT UNIQUE,         -- stops the same event being paid twice
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS counters (name TEXT PRIMARY KEY, value INTEGER NOT NULL);
"""


def connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection | None = None) -> None:
    conn = conn or connect()
    conn.executescript(SCHEMA)


def now() -> float:
    return time.time()


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def next_id(conn: sqlite3.Connection, prefix: str) -> str:
    """Allocate ids like H1, H2 ... atomically across processes."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute("SELECT value FROM counters WHERE name=?", (prefix,)).fetchone()
        value = (row["value"] if row else 0) + 1
        conn.execute(
            "INSERT INTO counters(name, value) VALUES(?, ?) "
            "ON CONFLICT(name) DO UPDATE SET value=excluded.value",
            (prefix, value),
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return f"{prefix}{value}"


def get_counter(conn, name: str, default: int = 0) -> int:
    row = conn.execute("SELECT value FROM counters WHERE name=?", (name,)).fetchone()
    return row["value"] if row else default


def set_counter(conn, name: str, value: int) -> None:
    conn.execute(
        "INSERT INTO counters(name, value) VALUES(?, ?) ON CONFLICT(name) DO UPDATE SET value=excluded.value",
        (name, value),
    )
