"""Tools the agents can call. Each tool = JSON schema + Python function.

Which role gets which tools is set in config/agents.yaml.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import httpx

from . import blackboard, bus, memory, rewards, sandbox, simulator, tasks
from .db import DATA_DIR
from .sources import europepmc

ALLOWED_DATA_HOSTS = {
    "ftp.ncbi.nlm.nih.gov", "www.ncbi.nlm.nih.gov", "eutils.ncbi.nlm.nih.gov",
    "www.ebi.ac.uk", "ftp.ebi.ac.uk", "zenodo.org", "figshare.com", "ndownloader.figshare.com",
    "raw.githubusercontent.com", "imsgc.net", "gwas.mrcieu.ac.uk",
}
MAX_DOWNLOAD = 200 * 1024 * 1024


@dataclass
class Ctx:
    conn: object
    agent: str
    role: str
    task_id: int
    tool_use_id: str = ""
    completed: str | None = None   # set by complete_task


@dataclass
class Tool:
    name: str
    description: str
    schema: dict
    fn: Callable


REGISTRY: dict[str, Tool] = {}


def tool(name: str, description: str, properties: dict, required: list[str]):
    def deco(fn):
        REGISTRY[name] = Tool(name, description, {
            "type": "object", "properties": properties, "required": required,
            "additionalProperties": False,
        }, fn)
        return fn
    return deco


S = {"type": "string"}
N = {"type": "number"}
LIST_S = {"type": "array", "items": {"type": "string"}}


# ---- communication ---------------------------------------------------------

@tool("read_inbox", "Read unread messages sent to you, your role, or everyone.", {}, [])
def read_inbox(ctx: Ctx):
    return bus.inbox(ctx.conn, ctx.agent, ctx.role)


@tool("send_message",
      "Send a message to another agent (by name), to a role ('role:analyst'), or to everyone ('*'). "
      "Use kind='request' when you need something back. Be concise and reference artifact ids.",
      {"to": S, "subject": S, "body": S, "kind": {"type": "string", "enum": ["info", "request", "reply"]},
       "refs": LIST_S, "reply_to": {"type": "integer"}}, ["to", "subject", "body"])
def send_message(ctx: Ctx, to, subject, body, kind="info", refs=None, reply_to=None):
    mid = bus.send(ctx.conn, ctx.agent, to, subject, body, kind, refs, reply_to)
    return {"message_id": mid}


@tool("rate_message",
      "Rate how useful a message you received was (0 = noise, 1 = changed what you did). "
      "Honest ratings decide the sender's collaboration reward.",
      {"message_id": {"type": "integer"}, "rating": N}, ["message_id", "rating"])
def rate_message(ctx: Ctx, message_id, rating):
    bus.rate(ctx.conn, ctx.agent, message_id, rating)
    return "ok"


@tool("create_task", "Ask another role to do a piece of work. It goes into the shared task queue.",
      {"role": S, "title": S, "details": S, "refs": LIST_S, "priority": N}, ["role", "title", "details"])
def create_task(ctx: Ctx, role, title, details, refs=None, priority=0.0):
    return {"task_id": tasks.create_task(ctx.conn, role, title, details, refs, priority, ctx.agent)}


@tool("complete_task", "Mark your current task finished. Call this once, at the end, with a summary.",
      {"summary": S}, ["summary"])
def complete_task(ctx: Ctx, summary):
    ctx.completed = summary
    return "task marked complete"


# ---- memory ----------------------------------------------------------------

@tool("remember", "Save something to your long-term memory. kind='lesson' for things you learned "
      "that should change how you work; 'pinned' for standing facts.",
      {"content": S, "kind": {"type": "string", "enum": ["note", "lesson", "pinned"]}, "tags": S}, ["content"])
def remember(ctx: Ctx, content, kind="note", tags=""):
    return {"memory_id": memory.remember(ctx.conn, ctx.agent, content, kind, tags)}


@tool("recall", "Search your long-term memory.", {"query": S}, ["query"])
def recall(ctx: Ctx, query):
    return memory.recall(ctx.conn, ctx.agent, query)


# ---- shared knowledge ------------------------------------------------------

@tool("search_knowledge", "Full-text search over the shared knowledge base (findings, datasets, "
      "hypotheses, experiments, results, reviews).",
      {"query": S, "kind": {"type": "string", "enum": list(blackboard.PREFIX)}, "limit": {"type": "integer"}}, [])
def search_knowledge(ctx: Ctx, query="", kind=None, limit=10):
    return blackboard.search(ctx.conn, query, kind, limit)


@tool("get_artifact", "Read one artifact in full by id (e.g. F3, H1, E2, R2).", {"id": S}, ["id"])
def get_artifact(ctx: Ctx, id):
    a = blackboard.get(ctx.conn, id)
    if not a:
        raise ValueError(f"{id} not found")
    a.pop("locked_hash", None)
    return a


@tool("view_scores", "See the reward leaderboard for all agents.", {}, [])
def view_scores(ctx: Ctx):
    return rewards.leaderboard(ctx.conn)


# ---- literature ------------------------------------------------------------

@tool("search_literature", "Search PubMed/Europe PMC. Returns titles, years, citation counts and abstracts.",
      {"query": S, "limit": {"type": "integer"}}, ["query"])
def search_literature(ctx: Ctx, query, limit=8):
    return europepmc.search(query, limit)


@tool("add_finding",
      "Record one evidence-backed claim from the literature. evidence_level: "
      "meta-analysis|rct|cohort|case-control|mendelian-randomization|animal|in-vitro|review|other.",
      {"claim": S, "source": {"type": "string", "description": "PMID or DOI"}, "evidence_level": S,
       "details": S, "cites": LIST_S}, ["claim", "source", "evidence_level"])
def add_finding(ctx: Ctx, claim, source, evidence_level, details="", cites=()):
    return {"id": blackboard.add_finding(ctx.conn, ctx.agent, claim, source, evidence_level, details, cites)}


# ---- data ------------------------------------------------------------------

@tool("download_dataset", "Download a public dataset file into data/datasets/. Only well-known "
      "public research hosts are allowed (NCBI, EBI, Zenodo, Figshare, GitHub raw, IMSGC, IEU OpenGWAS).",
      {"url": S, "filename": S}, ["url", "filename"])
def download_dataset(ctx: Ctx, url, filename):
    host = urlparse(url).hostname or ""
    if host not in ALLOWED_DATA_HOSTS:
        raise ValueError(f"host {host} not in allowlist {sorted(ALLOWED_DATA_HOSTS)}")
    dest = sandbox.safe_path(f"datasets/{Path(filename).name}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes():
                size += len(chunk)
                if size > MAX_DOWNLOAD:
                    raise ValueError("file larger than 200 MB limit")
                f.write(chunk)
    return {"local_path": str(dest.relative_to(DATA_DIR.resolve())), "bytes": size}


@tool("inspect_file", "Show the first lines of a file inside the data directory, or list a folder.",
      {"path": S, "lines": {"type": "integer"}}, ["path"])
def inspect_file(ctx: Ctx, path, lines=20):
    p = sandbox.safe_path(path)
    if p.is_dir():
        return sorted(str(x.relative_to(DATA_DIR.resolve())) for x in p.iterdir())[:200]
    with open(p, "r", errors="replace") as f:
        return "".join(f.readline() for _ in range(min(lines, 200)))


@tool("add_dataset", "Register a dataset that is ready for analysis so others can use it.",
      {"name": S, "url": S, "local_path": S, "description": S, "variables": S, "cites": LIST_S},
      ["name", "local_path", "description"])
def add_dataset(ctx: Ctx, name, local_path, description, url="", variables="", cites=()):
    return {"id": blackboard.add_dataset(ctx.conn, ctx.agent, name, url, local_path, description, variables, cites)}


@tool("run_analysis",
      "Run a Python script (cwd = data/). Give `code` to run new code, or `script_path` to re-run an "
      "existing script exactly (used for replication). numpy/scipy/pandas/statsmodels are available if "
      "installed. Print the numbers you need; output is truncated to 8 KB.",
      {"code": S, "script_path": S}, [])
def run_analysis(ctx: Ctx, code=None, script_path=None):
    if not code and not script_path:
        raise ValueError("give code or script_path")
    save_as = f"workspace/{ctx.agent}/task{ctx.task_id}/{ctx.tool_use_id[-12:] or 'run'}.py"
    return sandbox.run_python(code, script_path, save_as)


# ---- mechanistic reasoning and the virtual mouse / patient -----------------

@tool("describe_model",
      "Describe the in-silico MS models (virtual EAE mouse, cuprizone mouse, virtual patient): their "
      "state variables, every parameter and what biology it stands for, available drugs/conditions and "
      "their mechanism, and the simulation spec format. Read this before designing a simulation.", {}, [])
def describe_model(ctx: Ctx):
    return simulator.describe()


SIM_SPEC = {"type": "object", "description":
            "Simulation spec: {model, n_per_arm, days?, seed, arms:[{name, interventions:[{name, dose, "
            "start_day, stop_day} | {name:'custom', effects:{param: multiplier}, start_day}], conditions:[...], "
            "param_overrides:{param: multiplier}}], cohort_conditions:[...]}. Arm 0 is the control. "
            "See describe_model."}


@tool("run_simulation",
      "Run an experiment on virtual animals or patients (mechanistic ODE model with between-subject "
      "variability). Give experiment_id to run a pre-registered in-silico experiment exactly as locked "
      "(optionally with a different seed to check robustness), or spec for an exploratory run. "
      "Exploratory runs are for building intuition and are labelled as such; they never count as tests. "
      "Returns per-arm means/SDs, each arm vs control (difference, Cohen's d, Welch p) and short curves.",
      {"experiment_id": S, "spec": SIM_SPEC, "seed": {"type": "integer"}}, [])
def run_simulation(ctx: Ctx, experiment_id=None, spec=None, seed=None):
    if experiment_id:
        exp = blackboard.get(ctx.conn, experiment_id)
        if not exp or exp["kind"] != "experiment" or not exp["body"].get("simulation"):
            raise ValueError(f"{experiment_id} is not a pre-registered in-silico experiment")
        spec, purpose = dict(exp["body"]["simulation"]), "preregistered"
    elif spec:
        purpose = "exploratory"
    else:
        raise ValueError("give experiment_id or spec")
    if seed is not None:
        spec = {**spec, "seed": int(seed)}
    res = simulator.run(spec)
    rel = f"simulations/{ctx.agent}/task{ctx.task_id}/{purpose}-{res['spec_hash']}-seed{res['spec']['seed']}.json"
    path = sandbox.safe_path(rel)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**res, "purpose": purpose, "experiment_id": experiment_id}, indent=1))
    step = max(1, len(next(iter(res["curves"].values()))["day"]) // 12)
    short = {arm: {k: v[::step] for k, v in c.items()} for arm, c in res["curves"].items()}
    return {"purpose": purpose, "experiment_id": experiment_id, "saved_to": rel, "model": res["model"],
            "spec_hash": res["spec_hash"], "seed": res["spec"]["seed"], "summary": res["summary"],
            "comparisons": res["comparisons"], "curves": short, "note": res["note"]}


@tool("add_mechanism",
      "Record a causal mechanism map reasoned from how the body and the disease work (immunology, "
      "neurobiology, physiology, pharmacology). causal_chain is a list of links "
      "{cause, effect, relation: increases|decreases|required_for|blocks, basis: "
      "established|literature|inferred|speculative, source (PMID/DOI/F-id, required for 'literature'), note}. "
      "predictions: what the chain implies that could be checked. gaps: weakest or untested links.",
      {"title": S, "summary": S,
       "causal_chain": {"type": "array", "items": {"type": "object"}},
       "predictions": LIST_S, "gaps": S, "cites": LIST_S},
      ["title", "summary", "causal_chain", "predictions"])
def add_mechanism(ctx: Ctx, **kw):
    return {"id": blackboard.add_mechanism(ctx.conn, ctx.agent, **kw)}


# ---- scientific pipeline ---------------------------------------------------

@tool("propose_hypothesis",
      "Propose a testable hypothesis about MS cause or treatment. Must cite findings, datasets or mechanism "
      "maps (C*). basis: literature (from papers), mechanistic (reasoned from physiology/immunology, may be "
      "new), data, or mixed. prior_probability = your honest chance it is true (0.01-0.99). If it can be "
      "tested in the virtual mouse/patient, say how in testable_prediction.",
      {"statement": S, "mechanism": S, "rationale": S, "prior_probability": N,
       "testable_prediction": S, "cites": LIST_S,
       "basis": {"type": "string", "enum": list(blackboard.BASIS)}},
      ["statement", "mechanism", "rationale", "prior_probability", "testable_prediction", "cites"])
def propose_hypothesis(ctx: Ctx, **kw):
    return {"id": blackboard.propose_hypothesis(ctx.conn, ctx.agent, **kw)}


@tool("preregister_experiment",
      "Lock a test plan BEFORE any analysis. After this it cannot be edited. predicted_probability = "
      "your honest chance the success criteria will be met. You are scored on calibration. For an "
      "in-silico test, put the full simulation spec in `simulation` (see describe_model) and give success "
      "criteria in terms of its metrics, e.g. 'arm 1 vs control: peak_score diff <= -0.5 and p_welch < 0.01'.",
      {"hypothesis_id": S, "method": S, "dataset": {"type": "string", "description": "dataset id or path"},
       "prediction": S, "success_criteria": {"type": "string", "description": "exact statistic and threshold"},
       "predicted_probability": N, "analysis_plan": S, "cites": LIST_S, "simulation": SIM_SPEC},
      ["hypothesis_id", "method", "dataset", "prediction", "success_criteria", "predicted_probability", "analysis_plan"])
def preregister_experiment(ctx: Ctx, **kw):
    return {"id": blackboard.preregister_experiment(ctx.conn, ctx.agent, **kw)}


@tool("record_result",
      "Report the outcome of a pre-registered experiment. Disclose every deviation from the plan.",
      {"experiment_id": S, "outcome": {"type": "string", "enum": ["supported", "refuted", "inconclusive"]},
       "statistics": S, "summary": S, "code_path": S, "deviations": LIST_S},
      ["experiment_id", "outcome", "statistics", "summary"])
def record_result(ctx: Ctx, **kw):
    return {"id": blackboard.record_result(ctx.conn, ctx.agent, **kw)}


@tool("review",
      "Independently review a result: replicated (you re-ran it and got the same outcome), valid "
      "(sound but not re-run), flawed (deviates from plan, bug, bias), inconclusive.",
      {"target_id": S, "verdict": {"type": "string", "enum": ["replicated", "valid", "flawed", "inconclusive"]},
       "comments": S, "replicated_outcome": S}, ["target_id", "verdict", "comments"])
def review(ctx: Ctx, **kw):
    return {"id": blackboard.review(ctx.conn, ctx.agent, **kw)}


# ---- dispatch --------------------------------------------------------------

def schemas(names: list[str]) -> list[dict]:
    return [{"name": n, "description": REGISTRY[n].description, "input_schema": REGISTRY[n].schema}
            for n in names]


def execute(ctx: Ctx, name: str, args: dict) -> tuple[str, bool]:
    t = REGISTRY.get(name)
    if not t:
        return f"unknown tool {name}", True
    try:
        out = t.fn(ctx, **(args or {}))
        return (out if isinstance(out, str) else json.dumps(out, ensure_ascii=False, default=str)), False
    except Exception as e:  # tool errors go back to the model, not up the stack
        return f"Error: {type(e).__name__}: {e}", True
