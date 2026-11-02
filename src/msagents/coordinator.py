"""Coordinator: starts one process per agent, feeds the research agenda into
the task queue, and turns collaboration signals into rewards.

It is deliberately plain code, not an LLM: scheduling and scoring must be
predictable and impossible for agents to talk their way around. The LLM
"pi" agent handles the scientific direction.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import signal
import threading
from pathlib import Path

import yaml

from . import rewards, tasks
from .agent import Agent, RoleConfig
from .db import connect, get_counter, init_db, set_counter

log = logging.getLogger(__name__)
DEFAULT_CONFIG = Path("config/agents.yaml")


def load_config(path: Path = DEFAULT_CONFIG, backend: str | None = None, model: str | None = None) -> dict:
    """backend/model (e.g. from the command line) override the file for every role."""
    raw = yaml.safe_load(Path(path).read_text())
    d = raw.get("defaults", {})
    backends = raw.get("backends", {})
    chosen = backend or os.environ.get("MSAGENTS_BACKEND") or d.get("backend", "anthropic")
    model = model or os.environ.get("MSAGENTS_MODEL")

    def role_model(r: dict) -> str:
        if model:
            return model
        if chosen == "anthropic":
            return r.get("model", d.get("model", "claude-opus-5-5"))
        # other backends: per-role "<backend>_model", else the backend's default model
        return r.get(f"{chosen}_model") or backends.get(chosen, {}).get("model") or "mock"

    raw["backend"] = chosen
    raw["role_configs"] = {
        role: RoleConfig(
            role=role,
            prompt=r["prompt"],
            tools=list(dict.fromkeys(raw.get("common_tools", []) + r.get("tools", []))),
            mission=raw["mission"],
            model=role_model(r),
            effort=r.get("effort", d.get("effort", "high")),
            max_steps=r.get("max_steps", d.get("max_steps", 40)),
            max_tokens=r.get("max_tokens", d.get("max_tokens", 16000)),
            backend=chosen,
            backend_opts=dict(backends.get(chosen, {})),
        )
        for role, r in raw["roles"].items()
    }
    raw["agents"] = {f"{role}-{i + 1}": role for role, r in raw["roles"].items() for i in range(r.get("count", 1))}
    return raw


def make_llm(cfg: RoleConfig, mock: bool):
    from .llm import make_backend
    return make_backend("mock" if mock else cfg.backend, cfg.model, cfg.effort, cfg.max_tokens, cfg.backend_opts)


def agent_process(name: str, config_path: str, mock: bool, log_level: str,
                  backend: str | None = None, model: str | None = None) -> None:
    logging.basicConfig(level=log_level, format=f"%(asctime)s {name:<24} %(message)s", datefmt="%H:%M:%S")
    raw = load_config(Path(config_path), backend, model)
    role = raw["agents"][name]
    cfg = raw["role_configs"][role]
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    Agent(name, cfg, make_llm(cfg, mock), list(raw["agents"])).run_forever(stop, poll=1.0 if mock else 5.0)


class Coordinator:
    def __init__(self, raw: dict, topics: int | None = None):
        self.raw = raw
        self.conn = connect()
        self.agenda = raw.get("agenda", [])[: topics or None]
        self.roles = set(raw["roles"])

    def _open(self, role: str) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) c FROM tasks WHERE role=? AND status IN ('pending','running')", (role,)
        ).fetchone()["c"]

    def _count(self, kind: str) -> int:
        return self.conn.execute("SELECT COUNT(*) c FROM artifacts WHERE kind=?", (kind,)).fetchone()["c"]

    def seed(self) -> int:
        created = 0
        # 1. Next agenda topic once scouts are free.
        idx = get_counter(self.conn, "agenda_idx")
        if idx < len(self.agenda) and self._open("literature_scout") == 0:
            topic = self.agenda[idx]
            tasks.create_task(self.conn, "literature_scout", f"Literature review: {topic}",
                              f"Find the strongest evidence on: {topic}")
            tasks.create_task(self.conn, "data_curator", f"Find data: {topic}",
                              f"Find public datasets that could test hypotheses about: {topic}")
            created += 2
            if "systems_biologist" in self.roles:
                tasks.create_task(self.conn, "systems_biologist", f"Mechanism analysis: {topic}",
                                  f"Reason from immunology, neurobiology and physiology about: {topic}. "
                                  "Map the causal chain, find where it predicts something not yet tested, "
                                  "probe it in the virtual mouse/patient, and propose mechanistic hypotheses.")
                created += 1
            set_counter(self.conn, "agenda_idx", idx + 1)
        # 2. Hypothesis round when enough new evidence has accumulated.
        evidence = self._count("finding") + self._count("dataset") + self._count("mechanism")
        mark = get_counter(self.conn, "hyp_watermark")
        if evidence - mark >= 3 and self._open("hypothesis_generator") == 0:
            tasks.create_task(self.conn, "hypothesis_generator", "Generate hypotheses from new evidence",
                              f"{evidence - mark} new findings/datasets since the last round.")
            set_counter(self.conn, "hyp_watermark", evidence)
            created += 1
        # 3. PI review after new reviewed results.
        reviews = self._count("review")
        pmark = get_counter(self.conn, "pi_watermark")
        if reviews > pmark and self._open("pi") == 0:
            tasks.create_task(self.conn, "pi", "Program review",
                              f"{reviews - pmark} new reviewed results. Check scores and set priorities.")
            set_counter(self.conn, "pi_watermark", reviews)
            created += 1
        return created

    def tick(self) -> dict:
        created = self.seed()
        paid = rewards.sweep(self.conn)
        busy = self.conn.execute(
            "SELECT COUNT(*) c FROM tasks WHERE status IN ('pending','running')"
        ).fetchone()["c"]
        return {"created": created, "paid": paid, "busy": busy}


def run_lab(config_path: Path, mock: bool, only: list[str] | None, topics: int | None,
            until_idle: bool, log_level: str = "INFO", tick_seconds: float | None = None,
            backend: str | None = None, model: str | None = None, write_report: bool = True) -> None:
    init_db()
    raw = load_config(config_path, backend, model)
    log.info("backend: %s", "mock" if mock else raw["backend"])
    names = [n for n, r in raw["agents"].items() if not only or r in only or n in only]

    def spawn(n):
        return mp.Process(target=agent_process, args=(n, str(config_path), mock, log_level, backend, model), name=n)

    procs = {n: spawn(n) for n in names}
    for p in procs.values():
        p.start()
    coord = Coordinator(raw, topics)
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    tick_seconds = tick_seconds or (1.0 if mock else 15.0)
    idle_ticks = 0
    try:
        while not stop.is_set():
            st = coord.tick()
            for n, p in list(procs.items()):
                if not p.is_alive() and not stop.is_set():
                    log.warning("agent %s exited (code %s); restarting - it will resume its task", n, p.exitcode)
                    procs[n] = spawn(n)
                    procs[n].start()
            idle_ticks = idle_ticks + 1 if st["busy"] == 0 and st["created"] == 0 else 0
            if until_idle and idle_ticks >= 3:
                log.info("all work finished")
                break
            stop.wait(tick_seconds)
    finally:
        for p in procs.values():
            p.terminate()  # SIGTERM -> agents stop after their current step; leases kept for resume
        for p in procs.values():
            p.join(timeout=30)
        rewards.sweep(coord.conn)
        if write_report:
            from .report import write_report as _write
            path = _write(coord.conn)
            log.info("report written to %s", path)
