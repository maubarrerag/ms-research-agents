"""Reward ledger: collaboration score + hypothesis-testing score.

Design goals (see ARCHITECTURE.md, "Rewards"):
* Reward calibrated, pre-registered tests, not "positive" results. An agent
  that predicts 0.9 and is refuted loses points; a clean refutation of an
  over-hyped idea is worth something. This removes the incentive to p-hack.
* Reward collaboration only when another agent shows it was useful: citing
  your artifact, rating your message, or getting a reply to their request.
  Self-citation and self-rating earn nothing.
"""
from __future__ import annotations

import math
import sqlite3

from . import blackboard
from .db import now

OUTCOME_VALUE = {"supported": 1.0, "refuted": 0.0}


def award(conn, agent: str, channel: str, amount: float, reason: str, ref: str | None, dedupe_key: str) -> bool:
    try:
        conn.execute(
            "INSERT INTO rewards(agent, channel, amount, reason, ref, dedupe_key, created_at) VALUES (?,?,?,?,?,?,?)",
            (agent, channel, round(amount, 4), reason, ref, dedupe_key, now()),
        )
        return True
    except sqlite3.IntegrityError:
        return False  # already paid


def calibration(p: float, outcome: float) -> float:
    """1 at perfect confidence-and-right, 0 for a coin flip (p=0.5), -3 at worst."""
    return 1.0 - 4.0 * (p - outcome) ** 2


def score_result(conn, result_id: str, review_id: str) -> None:
    result = blackboard.get(conn, result_id)
    rev = blackboard.get(conn, review_id)
    exp = blackboard.get(conn, result["body"]["experiment_id"])
    hyp = blackboard.get(conn, exp["body"]["hypothesis_id"])
    verdict = rev["body"]["verdict"]
    outcome = result["body"]["outcome"]
    key = f"score:{result_id}"

    # Analyst: rigor of execution, judged by an independent critic.
    rigor = {"replicated": 1.0, "valid": 0.5, "inconclusive": 0.0, "flawed": -1.0}[verdict]
    rigor -= 0.1 * len(result["body"].get("deviations") or [])
    award(conn, result["author"], "testing", rigor, f"execution rigor ({verdict})", result_id, f"{key}:analyst")

    # Critic: flat reward for doing the review, independent of which verdict
    # was chosen, so there's no reason to be harsh or lenient.
    award(conn, rev["author"], "collaboration", 0.3, "completed independent review", review_id, f"{key}:critic")

    trusted = verdict in ("replicated", "valid") and outcome in OUTCOME_VALUE
    if outcome == "inconclusive" and verdict != "flawed":
        award(conn, exp["author"], "testing", -0.25, "design was underpowered / inconclusive", exp["id"], f"{key}:designer")
        return
    if not trusted:
        return

    o = OUTCOME_VALUE[outcome]
    p_design = exp["body"]["predicted_probability"]
    award(conn, exp["author"], "testing", calibration(p_design, o),
          f"calibration of predicted p={p_design} vs outcome={outcome}", exp["id"], f"{key}:designer")

    # Hypothesis author: a proper scoring rule on their own prior (honesty pays),
    # plus a discovery bonus. The bonus uses the designer's independent forecast,
    # not the author's prior, so understating confidence can't inflate it.
    prior = hyp["body"]["prior_probability"]
    hyp_reward = 0.5 * calibration(prior, o)
    if o == 1.0:
        hyp_reward += min(2.0, -math.log2(p_design)) * (1.0 if verdict == "replicated" else 0.5)
        reason = f"hypothesis supported (prior {prior}, forecast {p_design})"
    else:
        hyp_reward += 0.2
        reason = f"hypothesis cleanly refuted (prior {prior})"
    award(conn, hyp["author"], "testing", hyp_reward, reason, hyp["id"], f"{key}:hypothesis")
    blackboard.set_status(conn, hyp["id"], outcome if verdict == "replicated" else f"{outcome}_unreplicated")

    # Upstream credit: authors of the findings/datasets the hypothesis built on.
    for cited in hyp["cites"]:
        src = blackboard.get(conn, cited)
        if src and src["author"] != hyp["author"]:
            award(conn, src["author"], "collaboration", 0.2,
                  f"your {src['kind']} {cited} fed a tested hypothesis", hyp["id"], f"{key}:upstream:{cited}")


def sweep(conn) -> int:
    """Turn collaboration signals into rewards. Safe to run any number of times."""
    paid = 0
    # Citations by other agents.
    for r in conn.execute(
        "SELECT c.from_id, c.to_id, a.author AS citer, b.author AS cited_author FROM citations c "
        "JOIN artifacts a ON a.id=c.from_id JOIN artifacts b ON b.id=c.to_id WHERE a.author != b.author"
    ).fetchall():
        paid += award(conn, r["cited_author"], "collaboration", 0.3,
                      f"{r['citer']} built on {r['to_id']}", r["to_id"], f"cite:{r['from_id']}:{r['to_id']}")
    # Helpfulness ratings from recipients. Rating below 0.3 costs points.
    for r in conn.execute(
        "SELECT m.id, m.from_agent, mr.agent AS rater, mr.rating FROM message_reads mr "
        "JOIN messages m ON m.id=mr.message_id WHERE mr.rating IS NOT NULL AND mr.agent != m.from_agent"
    ).fetchall():
        paid += award(conn, r["from_agent"], "collaboration", r["rating"] - 0.3,
                      f"message rated {r['rating']:.2f} by {r['rater']}", f"M{r['id']}", f"rate:{r['id']}:{r['rater']}")
    # Answered requests.
    for r in conn.execute(
        "SELECT reply.id, reply.from_agent FROM messages reply JOIN messages req ON req.id=reply.reply_to "
        "WHERE req.kind='request' AND reply.from_agent != req.from_agent"
    ).fetchall():
        paid += award(conn, r["from_agent"], "collaboration", 0.3, "answered a request",
                      f"M{r['id']}", f"reply:{r['id']}")
    return paid


def scorecard(conn, agent: str) -> dict:
    rows = conn.execute("SELECT channel, SUM(amount) s FROM rewards WHERE agent=? GROUP BY channel", (agent,)).fetchall()
    sc = {"collaboration": 0.0, "testing": 0.0}
    sc.update({r["channel"]: round(r["s"], 2) for r in rows})
    recent = conn.execute(
        "SELECT amount, reason FROM rewards WHERE agent=? ORDER BY id DESC LIMIT 5", (agent,)
    ).fetchall()
    sc["recent"] = [f"{r['amount']:+.2f} {r['reason']}" for r in recent]
    return sc


def leaderboard(conn) -> list[dict]:
    rows = conn.execute(
        "SELECT agent, "
        "SUM(CASE WHEN channel='collaboration' THEN amount ELSE 0 END) collab, "
        "SUM(CASE WHEN channel='testing' THEN amount ELSE 0 END) testing, "
        "SUM(amount) total FROM rewards GROUP BY agent ORDER BY total DESC"
    ).fetchall()
    return [dict(r) for r in rows]
