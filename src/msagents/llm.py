"""Model backends. All return {"content": [block dicts], "stop_reason": str}
in Anthropic message format, so the agent loop and checkpoints don't care
which backend produced a turn (you can even switch backend and resume).

  anthropic    Claude API with an API key (best: native tools, caching, fallback)
  claude_code  your Claude Pro/Max login, through the `claude` CLI in headless mode
  litellm      any model LiteLLM supports: Ollama, OpenAI, Gemini, Groq, OpenRouter...
  mock         offline scripted agents (mock.py)
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import re
import shutil
import subprocess
import time
import uuid

log = logging.getLogger(__name__)


def _blocks(text: str, calls: list[tuple[str, str, dict]]) -> dict:
    content = []
    if text:
        content.append({"type": "text", "text": text})
    for cid, name, inp in calls:
        content.append({"type": "tool_use", "id": cid, "name": name, "input": inp})
    return {"content": content, "stop_reason": "tool_use" if calls else "end_turn"}


# ---- 1. Anthropic API ---------------------------------------------------------

class ClaudeLLM:
    def __init__(self, model: str = "claude-opus-5-5", effort: str = "high", max_tokens: int = 16000, **_):
        import anthropic
        self.anthropic = anthropic
        self.client = anthropic.Anthropic()  # ANTHROPIC_API_KEY or `ant auth login` profile
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens

    def complete(self, system: str, messages: list, tools: list, ctx=None) -> dict:
        anthropic = self.anthropic
        for attempt in range(6):
            try:
                resp = self.client.beta.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=system,
                    messages=messages,
                    tools=tools,
                    thinking={"type": "adaptive"},
                    output_config={"effort": self.effort},
                    cache_control={"type": "ephemeral"},  # system + tools + history are reused every turn
                    # If a safety classifier declines (biomedical topics can trip the
                    # `bio` category), retry server-side on Anthropic's recommended model.
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                )
                data = resp.model_dump(mode="json", exclude_none=True)
                return {"content": data["content"], "stop_reason": resp.stop_reason,
                        "stop_details": data.get("stop_details"), "request_id": resp._request_id}
            except anthropic.RateLimitError as e:
                wait = int(e.response.headers.get("retry-after", "30"))
                log.warning("rate limited, waiting %ss", wait)
                time.sleep(wait)
            except anthropic.APIStatusError as e:
                if e.status_code >= 500:
                    time.sleep(2 ** attempt)
                    continue
                raise  # 4xx: a bug in the request, don't retry
            except anthropic.APIConnectionError:
                time.sleep(2 ** attempt)
        raise RuntimeError("Claude API unavailable after retries")


# ---- shared: render a tool conversation as plain text --------------------------

def render_tools(tools: list) -> str:
    lines = []
    for t in tools:
        lines.append(f"- {t['name']}: {t['description']}\n  input schema: "
                     + json.dumps(t["input_schema"].get("properties", {}), separators=(",", ":"))
                     + f"  required: {t['input_schema'].get('required', [])}")
    return "\n".join(lines)


def render_transcript(messages: list) -> str:
    out = []
    for m in messages:
        if isinstance(m["content"], str):
            out.append(f"=== {m['role'].upper()} ===\n{m['content']}")
            continue
        parts = []
        for b in m["content"]:
            t = b.get("type")
            if t == "text":
                parts.append(b["text"])
            elif t == "tool_use":
                parts.append(f"[tool_call id={b['id']}] {b['name']} {json.dumps(b.get('input') or {}, ensure_ascii=False)}")
            elif t == "tool_result":
                c = b.get("content", "")
                c = c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
                tag = "tool_error" if b.get("is_error") else "tool_result"
                parts.append(f"[{tag} for {b['tool_use_id']}]\n{c}")
        label = "YOU (assistant)" if m["role"] == "assistant" else ("TOOL RESULTS" if any(
            b.get("type") == "tool_result" for b in m["content"]) else "USER")
        out.append(f"=== {label} ===\n" + "\n".join(parts))
    return "\n\n".join(out)


# ---- 2. Claude Pro/Max login via the Claude Code CLI ---------------------------

PROTOCOL = """\
# How you act in this session
You are running inside an automated research harness, not a chat. The tools
listed below are NOT callable by you directly, and must never be invoked as
tool calls. Your only action is to give your structured output (the
StructuredOutput tool / JSON reply), once per turn, with:
  "text":       short reasoning or a note for the record (may be empty)
  "tool_calls": the tools you want the harness to run now, in order, each
                {"name": <tool name>, "input": {...arguments...}}
The harness runs them and shows you the results in the next turn as
[tool_result for <id>] blocks. Keep calling tools until the work is done,
then call complete_task. An empty tool_calls list ends your turn.
Only use action names listed below, with arguments that match their schemas.

# Actions you can request in "tool_calls" (the harness runs them, not you)
{tools}
"""


class ClaudeCodeLLM:
    """Uses your Claude subscription (Pro/Max) by running `claude -p` once per
    model turn. Tools are described in the prompt and returned as structured
    JSON, then executed by our own agent loop (so checkpoints, resume and the
    tool log work exactly as with the API backend).

    Notes:
    * Pro has usage limits shared with your own Claude use; a full lab run
      will hit them. `max_parallel` caps simultaneous calls across agents.
    * ANTHROPIC_API_KEY is removed from the child's environment so the CLI
      uses your logged-in subscription (run `claude` once and /login first).
    * Anthropic's terms allow using your own subscription through Claude Code
      for yourself; don't offer it to other people as a service.
    """

    def __init__(self, model: str = "sonnet", effort: str = "high", max_tokens: int = 16000,
                 max_parallel: int = 2, timeout: int = 900, cli: str = "claude", **_):
        self.cli = shutil.which(cli) or cli
        self.model, self.effort, self.timeout = model, effort, timeout
        self.max_parallel = max(1, int(max_parallel))
        from .db import DATA_DIR
        self.lock_dir = DATA_DIR / "locks"
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        self.cwd = DATA_DIR / "claude_code_cwd"   # empty dir: no project CLAUDE.md gets loaded
        self.cwd.mkdir(parents=True, exist_ok=True)

    @contextlib.contextmanager
    def _slot(self):
        """Cross-process semaphore with file locks: at most max_parallel CLI calls at once."""
        while True:
            for i in range(self.max_parallel):
                f = open(self.lock_dir / f"claude_code_{i}.lock", "w")
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    f.close()
                    continue
                try:
                    yield
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
                    f.close()
                return
            time.sleep(1.0)

    def _schema(self, tools: list) -> str:
        return json.dumps({
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "tool_calls": {"type": "array", "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string", "enum": [t["name"] for t in tools]},
                                   "input": {"type": "object"}},
                    "required": ["name", "input"]}},
            },
            "required": ["text", "tool_calls"],
        })

    def complete(self, system: str, messages: list, tools: list, ctx=None) -> dict:
        sys_prompt = system + ("\n\n" + PROTOCOL.replace("{tools}", render_tools(tools)) if tools else "")
        prompt = render_transcript(messages) + "\n\n=== YOUR NEXT REPLY (JSON) ==="
        if len(sys_prompt) > 100_000:            # keep argv small; move it into stdin
            prompt, sys_prompt = sys_prompt + "\n\n" + prompt, "Follow the instructions in the input."
        cmd = [self.cli, "-p", "--output-format", "json", "--tools", "", "--strict-mcp-config",
               "--no-session-persistence", "--model", self.model, "--system-prompt", sys_prompt]
        if tools:
            cmd += ["--json-schema", self._schema(tools)]
        if self.effort:
            cmd += ["--effort", self.effort]
        env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
        for attempt in range(8):
            with self._slot():
                try:
                    proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                                          timeout=self.timeout, env=env, cwd=self.cwd)
                except subprocess.TimeoutExpired:
                    log.warning("claude CLI timed out (attempt %s)", attempt + 1)
                    continue
            try:
                out = json.loads(proc.stdout)
            except json.JSONDecodeError:
                out = {"is_error": True, "result": (proc.stdout + proc.stderr)[-2000:]}
            if not out.get("is_error"):
                return self._parse(out)
            msg = str(out.get("result") or out.get("error") or proc.stderr)[:500]
            wait = self._limit_wait(msg)
            if wait is None and attempt >= 2:
                raise RuntimeError(f"claude CLI failed: {msg}")
            wait = wait if wait is not None else 10 * (attempt + 1)
            log.warning("claude CLI error (%s); waiting %ss", msg[:160], wait)
            time.sleep(wait)
        raise RuntimeError("claude CLI unavailable after retries")

    @staticmethod
    def _limit_wait(msg: str) -> int | None:
        """Seconds to wait if this is a subscription usage limit, else None."""
        if not re.search(r"usage limit|rate limit|limit reached|overloaded|429|529", msg, re.I):
            return None
        m = re.search(r"\|(\d{10})", msg)        # "...limit reached|<reset epoch>"
        if m:
            return max(30, min(int(m.group(1)) - int(time.time()) + 30, 6 * 3600))
        return 300

    @staticmethod
    def _parse(out: dict) -> dict:
        data = out.get("structured_output")
        if data is None and "tool_calls" not in (out.get("result") or ""):
            data = {"text": out.get("result") or "", "tool_calls": []}   # plain-text turn (no tools)
        if data is None:
            raw = out.get("result") or ""
            m = re.search(r"\{.*\}", raw, re.S)
            try:
                data = json.loads(m.group(0)) if m else {"text": raw, "tool_calls": []}
            except json.JSONDecodeError:
                data = {"text": raw, "tool_calls": []}
        calls = [(f"toolu_cc_{uuid.uuid4().hex[:20]}", c["name"], c.get("input") or {})
                 for c in data.get("tool_calls") or [] if c.get("name")]
        return _blocks((data.get("text") or "").strip(), calls)


# ---- 3. LiteLLM: local or third-party models -----------------------------------

def to_openai_messages(system: str, messages: list) -> list[dict]:
    out = [{"role": "system", "content": system}]
    for m in messages:
        if isinstance(m["content"], str):
            out.append({"role": m["role"], "content": m["content"]})
            continue
        if m["role"] == "assistant":
            text = "\n".join(b["text"] for b in m["content"] if b.get("type") == "text")
            calls = [{"id": b["id"], "type": "function",
                      "function": {"name": b["name"], "arguments": json.dumps(b.get("input") or {})}}
                     for b in m["content"] if b.get("type") == "tool_use"]
            msg = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
            continue
        texts = []
        for b in m["content"]:
            if b.get("type") == "tool_result":
                c = b.get("content", "")
                out.append({"role": "tool", "tool_call_id": b["tool_use_id"],
                            "content": c if isinstance(c, str) else json.dumps(c)})
            elif b.get("type") == "text":
                texts.append(b["text"])
        if texts:
            out.append({"role": "user", "content": "\n".join(texts)})
    return out


def to_openai_tools(tools: list) -> list[dict]:
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                              "parameters": t["input_schema"]}} for t in tools]


class LiteLLM:
    """Any model LiteLLM can reach. The model must support tool/function calling.

    Examples for `model`:
      ollama_chat/qwen3:14b          local, free (run `ollama serve`)
      openai/gpt-5-mini              needs OPENAI_API_KEY
      gemini/gemini-2.5-flash        needs GEMINI_API_KEY
      openrouter/<provider>/<model>  needs OPENROUTER_API_KEY
    """

    def __init__(self, model: str, effort: str = "high", max_tokens: int = 8000,
                 api_base: str | None = None, temperature: float | None = None, **extra):
        import litellm
        litellm.drop_params = True     # ignore params a provider doesn't support
        self.litellm = litellm
        self.model, self.max_tokens, self.api_base, self.temperature = model, max_tokens, api_base, temperature
        self.extra = {k: v for k, v in extra.items() if k not in ("max_parallel", "timeout", "cli")}

    def complete(self, system: str, messages: list, tools: list, ctx=None) -> dict:
        kw = dict(model=self.model, messages=to_openai_messages(system, messages),
                  max_tokens=self.max_tokens, num_retries=4, **self.extra)
        if tools:
            kw["tools"] = to_openai_tools(tools)
        if self.api_base:
            kw["api_base"] = self.api_base
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        resp = self.litellm.completion(**kw)
        msg = resp.choices[0].message
        calls = []
        for tc in getattr(msg, "tool_calls", None) or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_unparsed_arguments": tc.function.arguments}
            calls.append((tc.id or f"toolu_ll_{uuid.uuid4().hex[:20]}", tc.function.name, args))
        return _blocks((msg.content or "").strip(), calls)


def make_backend(backend: str, model: str, effort: str, max_tokens: int, opts: dict | None = None):
    opts = dict(opts or {})
    opts.pop("model", None)
    if backend == "mock":
        from .mock import MockLLM
        return MockLLM()
    if backend == "anthropic":
        return ClaudeLLM(model, effort, max_tokens)
    if backend == "claude_code":
        return ClaudeCodeLLM(model, effort, max_tokens, **opts)
    if backend == "litellm":
        return LiteLLM(model, effort, max_tokens, **opts)
    raise ValueError(f"unknown backend {backend!r} (anthropic | claude_code | litellm | mock)")

