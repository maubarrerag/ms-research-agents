import importlib
import threading

import pytest


@pytest.fixture()
def lab(tmp_path, monkeypatch):
    monkeypatch.setenv("MSAGENTS_DATA", str(tmp_path))
    from msagents import db
    importlib.reload(db)
    for mod in ("tasks", "bus", "memory", "blackboard", "rewards", "sandbox", "tools", "agent"):
        importlib.reload(importlib.import_module(f"msagents.{mod}"))
    db.init_db()
    return db.connect()


def _cfg(role, tools_):
    from msagents.agent import RoleConfig
    return RoleConfig(role, "test", tools_, "mission", "mock", "low", 10, 1000)


class CrashAfterToolLLM:
    """First turn calls a side-effecting tool; the process 'dies' on the next model call."""
    def __init__(self):
        self.calls = 0

    def complete(self, system, messages, tools, ctx):
        self.calls += 1
        if self.calls == 1:
            return {"stop_reason": "tool_use", "content": [{"type": "tool_use", "id": "toolu_1",
                    "name": "send_message", "input": {"to": "*", "subject": "hi", "body": "once"}}]}
        if self.calls == 2:
            raise KeyboardInterrupt("simulated crash")
        return {"stop_reason": "tool_use", "content": [{"type": "tool_use", "id": "toolu_2",
                "name": "complete_task", "input": {"summary": "finished after resume"}}]}


def test_resume_after_crash_does_not_repeat_side_effects(lab):
    from msagents import tasks
    from msagents.agent import Agent
    tid = tasks.create_task(lab, "scout", "t")
    llm = CrashAfterToolLLM()
    agent = Agent("scout-1", _cfg("scout", ["send_message", "complete_task"]), llm, ["scout-1"])
    with pytest.raises(KeyboardInterrupt):
        agent.run_task(tasks.claim(lab, "scout-1", "scout"))

    # New process, same agent name: it gets its own task back and resumes.
    agent2 = Agent("scout-1", _cfg("scout", ["send_message", "complete_task"]), llm, ["scout-1"])
    task = tasks.claim(lab, "scout-1", "scout")
    assert task["id"] == tid
    assert agent2.run_task(task) == "done"
    assert lab.execute("SELECT COUNT(*) c FROM messages").fetchone()["c"] == 1
    assert lab.execute("SELECT status FROM tasks WHERE id=?", (tid,)).fetchone()["status"] == "done"


def test_graceful_stop_keeps_task_for_same_agent(lab):
    from msagents import tasks
    from msagents.agent import Agent
    tasks.create_task(lab, "scout", "t")
    stop = threading.Event()
    stop.set()
    agent = Agent("scout-1", _cfg("scout", ["complete_task"]), CrashAfterToolLLM(), ["scout-1"])
    assert agent.run_task(tasks.claim(lab, "scout-1", "scout"), stop) == "interrupted"
    assert tasks.claim(lab, "scout-1", "scout") is not None


def _pipeline(lab, p_design=0.8, prior=0.3, outcome="supported", verdict="replicated"):
    from msagents import blackboard as bb
    f = bb.add_finding(lab, "scout-1", "claim", "PMID:1", "cohort")
    h = bb.propose_hypothesis(lab, "hyp-1", "s", "m", "r", prior, "pred", [f])
    e = bb.preregister_experiment(lab, "design-1", h, "t-test", "d.csv", "pred", "p<0.05", p_design, "plan")
    r = bb.record_result(lab, "analyst-1", e, outcome, "t=3", "ok")
    bb.review(lab, "critic-1", r, verdict, "re-ran")
    return {row["agent"]: row["total"] for row in __import__("msagents.rewards").rewards.leaderboard(lab)}


def test_calibration_rewards_and_penalises_overconfidence(lab):
    scores = _pipeline(lab, p_design=0.95, outcome="refuted")
    assert scores["design-1"] < -2.5          # confident and wrong
    assert scores["analyst-1"] == pytest.approx(1.0)
    assert scores["scout-1"] > 0              # upstream credit


def test_surprising_discovery_pays_more(lab):
    expected = _pipeline(lab, prior=0.5, p_design=0.9, outcome="supported")["hyp-1"]
    lab.execute("DELETE FROM rewards")
    surprising = _pipeline(lab, prior=0.5, p_design=0.2, outcome="supported")["hyp-1"]
    assert surprising > expected + 1


def test_understating_prior_does_not_pay(lab):
    honest = _pipeline(lab, prior=0.8, outcome="supported")["hyp-1"]
    lab.execute("DELETE FROM rewards")
    sandbagged = _pipeline(lab, prior=0.1, outcome="supported")["hyp-1"]
    assert honest > sandbagged


def test_preregistered_plan_cannot_be_edited(lab):
    from msagents import blackboard as bb
    f = bb.add_finding(lab, "a", "c", "s", "cohort")
    h = bb.propose_hypothesis(lab, "b", "s", "m", "r", 0.5, "p", [f])
    e = bb.preregister_experiment(lab, "c", h, "m", "d", "p", "crit", 0.5, "plan")
    lab.execute("UPDATE artifacts SET body=json_set(body,'$.success_criteria','anything') WHERE id=?", (e,))
    with pytest.raises(bb.BlackboardError, match="modified"):
        bb.record_result(lab, "d", e, "supported", "", "")


def test_no_self_review_and_no_self_credit(lab):
    from msagents import blackboard as bb, bus, rewards
    f = bb.add_finding(lab, "a", "c", "s", "cohort")
    with pytest.raises(bb.BlackboardError):
        bb.review(lab, "a", f, "valid", "mine")
    mid = bus.send(lab, "a", "a", "s", "b")
    bus.rate(lab, "a", mid, 1.0)
    rewards.sweep(lab)
    assert rewards.leaderboard(lab) == []
