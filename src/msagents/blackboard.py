"""Shared knowledge base. Agents post typed artifacts that cite each other.

Posting some artifact kinds starts the next pipeline step automatically:
hypothesis -> design task, experiment -> analysis task, result -> review task,
review -> scoring.

Mechanism maps (C*) are causal chains reasoned from physiology and
immunology, the "how the body works" side of the evidence; hypotheses can
build on them as well as on literature findings (F*) and datasets (D*).
"""
from __future__ import annotations

import hashlib
import json

from . import tasks
from .db import dumps, next_id, now
from .memory import fts_query

PREFIX = {"finding": "F", "dataset": "D", "mechanism": "C", "hypothesis": "H", "experiment": "E",
          "result": "R", "review": "V"}
BASIS = ("literature", "mechanistic", "data", "mixed")
LINK_BASIS = ("established", "literature", "inferred", "speculative")


class BlackboardError(ValueError):
    pass


def get(conn, artifact_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM artifacts WHERE id=?", (artifact_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["body"] = json.loads(d["body"])
    d["cites"] = [r["to_id"] for r in conn.execute("SELECT to_id FROM citations WHERE from_id=?", (artifact_id,))]
    return d


def search(conn, query: str = "", kind: str | None = None, limit: int = 10) -> list[dict]:
    if query:
        sql = ("SELECT a.id, a.kind, a.author, a.title, a.status FROM artifacts_fts f "
               "JOIN artifacts a ON a.rowid=f.rowid WHERE artifacts_fts MATCH ?")
        args: list = [fts_query(query)]
        if kind:
            sql += " AND a.kind=?"
            args.append(kind)
        sql += " ORDER BY rank LIMIT ?"
    else:
        sql = "SELECT id, kind, author, title, status FROM artifacts"
        args = []
        if kind:
            sql += " WHERE kind=?"
            args.append(kind)
        sql += " ORDER BY created_at DESC LIMIT ?"
    args.append(limit)
    return [dict(r) for r in conn.execute(sql, args)]


def post(conn, kind: str, author: str, title: str, body: dict, cites=(), status="open",
         locked_hash: str | None = None) -> str:
    missing = [c for c in cites if not conn.execute("SELECT 1 FROM artifacts WHERE id=?", (c,)).fetchone()]
    if missing:
        raise BlackboardError(f"cited artifacts do not exist: {missing}")
    aid = next_id(conn, PREFIX[kind])
    conn.execute(
        "INSERT INTO artifacts(id, kind, author, title, body, status, locked_hash, created_at) VALUES (?,?,?,?,?,?,?,?)",
        (aid, kind, author, title, dumps(body), status, locked_hash, now()),
    )
    for c in set(cites):
        conn.execute("INSERT OR IGNORE INTO citations(from_id, to_id) VALUES (?,?)", (aid, c))
    return aid


def set_status(conn, artifact_id: str, status: str) -> None:
    conn.execute("UPDATE artifacts SET status=? WHERE id=?", (status, artifact_id))


def _require(conn, artifact_id: str, kind: str) -> dict:
    a = get(conn, artifact_id)
    if not a or a["kind"] != kind:
        raise BlackboardError(f"{artifact_id} is not an existing {kind}")
    return a


def _prob(x, name) -> float:
    try:
        p = float(x)
    except (TypeError, ValueError):
        raise BlackboardError(f"{name} must be a number between 0 and 1")
    if not 0.01 <= p <= 0.99:
        raise BlackboardError(f"{name} must be between 0.01 and 0.99")
    return p


# ---- typed operations used by the tools -----------------------------------

def add_finding(conn, author, claim, source, evidence_level, details="", cites=()):
    return post(conn, "finding", author, claim,
                {"claim": claim, "source": source, "evidence_level": evidence_level, "details": details}, cites)


def add_dataset(conn, author, name, url, local_path, description, variables, cites=()):
    return post(conn, "dataset", author, name,
                {"url": url, "local_path": local_path, "description": description, "variables": variables}, cites)


def add_mechanism(conn, author, title, summary, causal_chain, predictions, gaps="", cites=()):
    """A causal map: links like 'EBNA1 IgG -[increases]-> GlialCAM binding', each with
    how well it's known. Speculative links are allowed but must be labelled."""
    if not causal_chain:
        raise BlackboardError("causal_chain needs at least one link")
    links = []
    for i, l in enumerate(causal_chain):
        if not isinstance(l, dict) or not l.get("cause") or not l.get("effect"):
            raise BlackboardError(f"link {i}: needs cause and effect")
        basis = l.get("basis", "speculative")
        if basis not in LINK_BASIS:
            raise BlackboardError(f"link {i}: basis must be one of {LINK_BASIS}")
        if basis == "literature" and not l.get("source"):
            raise BlackboardError(f"link {i}: basis 'literature' needs a source (PMID/DOI or F-id)")
        links.append({"cause": l["cause"], "effect": l["effect"], "relation": l.get("relation", "increases"),
                      "basis": basis, "source": l.get("source", ""), "note": l.get("note", "")})
    return post(conn, "mechanism", author, title, {
        "summary": summary, "causal_chain": links, "predictions": list(predictions or []), "gaps": gaps,
    }, cites)


def propose_hypothesis(conn, author, statement, mechanism, rationale, prior_probability, testable_prediction,
                       cites, basis="literature"):
    if not cites:
        raise BlackboardError("a hypothesis must cite at least one finding, dataset or mechanism map")
    if basis not in BASIS:
        raise BlackboardError(f"basis must be one of {BASIS}")
    p = _prob(prior_probability, "prior_probability")
    hid = post(conn, "hypothesis", author, statement, {
        "statement": statement, "mechanism": mechanism, "rationale": rationale, "basis": basis,
        "prior_probability": p, "testable_prediction": testable_prediction,
    }, cites)
    tasks.create_task(conn, "experiment_designer", f"Design a test for {hid}",
                      f"Hypothesis {hid}: {statement}\nPrediction: {testable_prediction}",
                      refs=[hid], created_by=author, priority=1.0)
    return hid


def preregister_experiment(conn, author, hypothesis_id, method, dataset, prediction,
                           success_criteria, predicted_probability, analysis_plan, cites=(), simulation=None):
    _require(conn, hypothesis_id, "hypothesis")
    p = _prob(predicted_probability, "predicted_probability")
    body = {
        "hypothesis_id": hypothesis_id, "method": method, "dataset": dataset,
        "prediction": prediction, "success_criteria": success_criteria,
        "predicted_probability": p, "analysis_plan": analysis_plan,
    }
    if simulation:
        from . import simulator
        try:
            body["simulation"] = simulator._check_spec(simulation)
        except simulator.SimError as e:
            raise BlackboardError(f"invalid simulation spec: {e}")
        body["dataset"] = f"in_silico:{body['simulation']['model']}"
    # Locking the plan before data is analysed is what makes the test honest.
    locked = hashlib.sha256(dumps(body).encode()).hexdigest()
    eid = post(conn, "experiment", author, f"Test of {hypothesis_id}: {prediction[:80]}",
               body, [hypothesis_id, *cites], status="registered", locked_hash=locked)
    set_status(conn, hypothesis_id, "under_test")
    how = (" It is an in-silico experiment: run it with run_simulation(experiment_id='%s')." % eid
           if simulation else "")
    tasks.create_task(conn, "analyst", f"Run pre-registered experiment {eid}",
                      f"Follow the locked analysis plan of {eid} exactly. Report any deviation.{how}",
                      refs=[eid, hypothesis_id], created_by=author, priority=2.0)
    return eid


def record_result(conn, author, experiment_id, outcome, statistics, summary, code_path="", deviations=()):
    exp = _require(conn, experiment_id, "experiment")
    if hashlib.sha256(dumps(exp["body"]).encode()).hexdigest() != exp["locked_hash"]:
        raise BlackboardError("experiment plan was modified after registration")
    if outcome not in ("supported", "refuted", "inconclusive"):
        raise BlackboardError("outcome must be supported|refuted|inconclusive")
    rid = post(conn, "result", author, f"{experiment_id}: {outcome}", {
        "experiment_id": experiment_id, "outcome": outcome, "statistics": statistics,
        "summary": summary, "code_path": code_path, "deviations": list(deviations),
    }, [experiment_id], status="awaiting_review")
    set_status(conn, experiment_id, "analysed")
    tasks.create_task(conn, "critic", f"Review and replicate result {rid}",
                      f"Check {rid} against the pre-registered plan in {experiment_id}. Re-run the code if possible.",
                      refs=[rid, experiment_id], created_by=author, priority=3.0)
    return rid


def review(conn, author, target_id, verdict, comments, replicated_outcome=None):
    target = get(conn, target_id)
    if not target:
        raise BlackboardError(f"{target_id} does not exist")
    if target["author"] == author:
        raise BlackboardError("agents cannot review their own work")
    if verdict not in ("replicated", "valid", "flawed", "inconclusive"):
        raise BlackboardError("verdict must be replicated|valid|flawed|inconclusive")
    vid = post(conn, "review", author, f"Review of {target_id}: {verdict}", {
        "target_id": target_id, "verdict": verdict, "comments": comments,
        "replicated_outcome": replicated_outcome,
    }, [target_id])
    if target["kind"] == "result":
        set_status(conn, target_id, f"reviewed_{verdict}")
        from . import rewards  # avoid import cycle
        rewards.score_result(conn, target_id, vid)
    return vid
