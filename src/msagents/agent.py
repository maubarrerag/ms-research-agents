"""The agent loop: claim a task, resume from checkpoint if one exists, call
the model, run tools, checkpoint after every step, repeat.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field

from . import bus, memory, rewards, tasks, tools
from .db import connect

log = logging.getLogger(__name__)


@dataclass
class RoleConfig:
    role: str
    prompt: str
    tools: list[str]
    mission: str
    model: str
    effort: str
    max_steps: int
    max_tokens: int
    backend: str = "anthropic"          # anthropic | claude_code | litellm | mock
    backend_opts: dict = field(default_factory=dict)


@dataclass
class LLMCtx:
    """What a backend may know about the current task (the mock uses it)."""
    role: str
    refs: list[str]
    scratch: dict = field(default_factory=dict)


class Agent:
    def __init__(self, name: str, cfg: RoleConfig, llm, roster: list[str]):
        self.name, self.cfg, self.llm, self.roster = name, cfg, llm, roster
        self.conn = connect()
        self.tool_schemas = tools.schemas(cfg.tools)

    # ---- prompts ----------------------------------------------------------

    def build_system(self) -> str:
        sc = rewards.scorecard(self.conn, self.name)
        return "\n\n".join([
            self.cfg.mission.strip(),
            f"Your name is {self.name}. Your role: {self.cfg.role}.\n{self.cfg.prompt.strip()}",
            "Team members: " + ", ".join(self.roster),
            "Your long-term memory:\n" + memory.digest(self.conn, self.name),
            "Your reward so far: collaboration={collaboration}, testing={testing}.\nRecent:\n".format(**sc)
            + ("\n".join(sc["recent"]) or "(none)"),
            "Working rules: check your inbox first; search the knowledge base before adding to it; "
            "before finishing, save at most one useful lesson with remember(kind='lesson'); "
            "finish by calling complete_task.",
        ])

    def build_task_prompt(self, task) -> str:
        unread = bus.inbox(self.conn, self.name, self.cfg.role, unread_only=True, limit=5)
        inbox_txt = "\n".join(f"- [#{m['id']} {m['kind']} from {m['from']}] {m['subject']}: {m['body'][:300]}"
                              for m in unread) or "(empty)"
        return (f"Task #{task['id']}: {task['title']}\n\n{task['details']}\n\n"
                f"Referenced artifacts: {', '.join(tasks.refs_of(task)) or 'none'}\n\n"
                f"Unread messages:\n{inbox_txt}")

    # ---- main loop --------------------------------------------------------

    def run_forever(self, stop: threading.Event | None = None, poll: float = 5.0, max_tasks: int | None = None):
        stop = stop or threading.Event()
        done = 0
        while not stop.is_set():
            task = tasks.claim(self.conn, self.name, self.cfg.role)
            if task is None:
                stop.wait(poll)
                continue
            try:
                self.run_task(task, stop)
            except Exception:
                log.exception("%s crashed on task #%s (attempt %s)", self.name, task["id"], task["attempts"])
                if task["attempts"] >= tasks.MAX_ATTEMPTS:
                    tasks.finish(self.conn, task["id"], self.name, "gave up after repeated crashes", status="failed")
                else:
                    tasks.release(self.conn, task["id"], self.name)  # checkpoint kept; retried after backoff
                stop.wait(min(60, 2 ** task["attempts"]))
            done += 1
            if max_tasks and done >= max_tasks:
                return

    def run_task(self, task, stop: threading.Event | None = None) -> str:
        tid = task["id"]
        ckpt = memory.load_checkpoint(self.conn, tid)
        if ckpt:
            system, messages, step = ckpt["system"], ckpt["messages"], ckpt["step"]
            log.info("%s resuming task #%s at step %s", self.name, tid, step)
        else:
            system = self.build_system()
            messages = [{"role": "user", "content": self.build_task_prompt(task)}]
            step = 0
            memory.save_checkpoint(self.conn, tid, self.name, system, messages, step)
            log.info("%s started task #%s: %s", self.name, tid, task["title"])

        ctx = tools.Ctx(self.conn, self.name, self.cfg.role, tid)
        llm_ctx = LLMCtx(self.cfg.role, tasks.refs_of(task))

        while step < self.cfg.max_steps:
            if stop is not None and stop.is_set():
                tasks.release(self.conn, tid, self.name)
                return "interrupted"
            last = messages[-1]

            if last["role"] == "assistant":
                uses = [b for b in last["content"] if b.get("type") == "tool_use"]
                if not uses:
                    text = " ".join(b.get("text", "") for b in last["content"] if b.get("type") == "text")
                    return self._finish(tid, text or "finished without summary")
                results = [self._run_tool(ctx, u) for u in uses]
                messages.append({"role": "user", "content": results})
                memory.save_checkpoint(self.conn, tid, self.name, system, messages, step)
                if ctx.completed is not None:
                    return self._finish(tid, ctx.completed)
                continue

            resp = self.llm.complete(system, messages, self.tool_schemas, llm_ctx)
            if resp["stop_reason"] == "refusal":
                tasks.finish(self.conn, tid, self.name, json.dumps(resp.get("stop_details")), status="failed")
                log.warning("%s task #%s declined by model: %s", self.name, tid, resp.get("stop_details"))
                return "refused"
            messages.append({"role": "assistant", "content": resp["content"]})
            step += 1
            memory.save_checkpoint(self.conn, tid, self.name, system, messages, step)
            tasks.heartbeat(self.conn, tid, self.name)

        tasks.finish(self.conn, tid, self.name, f"stopped after {step} steps", status="failed")
        return "max_steps"

    def _run_tool(self, ctx: tools.Ctx, use: dict) -> dict:
        ctx.tool_use_id = use["id"]
        cached = memory.cached_tool_result(self.conn, use["id"])
        if cached:
            out, is_err = cached
            if use["name"] == "complete_task" and not is_err:
                ctx.completed = use["input"].get("summary", "")
        else:
            out, is_err = tools.execute(ctx, use["name"], use.get("input") or {})
            memory.store_tool_result(self.conn, use["id"], ctx.task_id, self.name, use["name"],
                                     use.get("input"), out, is_err)
        log.debug("%s %s -> %s", self.name, use["name"], out[:200])
        return {"type": "tool_result", "tool_use_id": use["id"], "content": out, "is_error": is_err}

    def _finish(self, tid: int, summary: str) -> str:
        tasks.finish(self.conn, tid, self.name, summary)
        log.info("%s finished task #%s: %s", self.name, tid, summary[:120])
        return "done"
