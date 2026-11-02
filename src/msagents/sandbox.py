"""Run analysis code written by agents.

This is process isolation only (separate interpreter, time limit, working
directory inside data/). It is NOT a security boundary. For anything beyond
local experiments, run agents inside the Docker image (see README).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .db import DATA_DIR

TIMEOUT = int(os.environ.get("MSAGENTS_SANDBOX_TIMEOUT", "300"))
MAX_OUTPUT = 8000


def safe_path(rel: str) -> Path:
    p = (DATA_DIR / rel).resolve()
    if not str(p).startswith(str(DATA_DIR.resolve())):
        raise ValueError("path must stay inside the data directory")
    return p


def run_python(code: str | None, script_rel: str | None, save_as: str) -> dict:
    if script_rel:
        script = safe_path(script_rel)
        if not script.exists():
            raise ValueError(f"script not found: {script_rel}")
    else:
        script = safe_path(save_as)
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(code or "")
    env = {"PATH": os.environ.get("PATH", ""), "MPLBACKEND": "Agg", "PYTHONHASHSEED": "0"}
    try:
        proc = subprocess.run(
            [sys.executable, "-I", str(script)], cwd=DATA_DIR, env=env,
            capture_output=True, text=True, timeout=TIMEOUT,
        )
        out, err, code_ = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired:
        out, err, code_ = "", f"timed out after {TIMEOUT}s", -1
    return {
        "script_path": str(script.relative_to(DATA_DIR.resolve())),
        "return_code": code_,
        "stdout": out[-MAX_OUTPUT:],
        "stderr": err[-MAX_OUTPUT // 2:],
    }
