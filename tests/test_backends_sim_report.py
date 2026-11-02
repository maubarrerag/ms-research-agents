import importlib
import json
import os
import stat
import sys
import types

import pytest


@pytest.fixture()
def lab(tmp_path, monkeypatch):
    monkeypatch.setenv("MSAGENTS_DATA", str(tmp_path))
    from msagents import db
    importlib.reload(db)
    for mod in ("tasks", "bus", "memory", "blackboard", "rewards", "sandbox", "simulator", "tools", "agent",
                "report", "llm"):
        importlib.reload(importlib.import_module(f"msagents.{mod}"))
    db.init_db()
    return db.connect()


SPEC = {"model": "eae_mouse", "n_per_arm": 12, "seed": 4, "arms": [
    {"name": "vehicle"}, {"name": "nat", "interventions": [{"name": "natalizumab", "start_day": 0}]}]}


# ---- simulator ---------------------------------------------------------------

def test_simulation_is_deterministic_per_seed():
    from msagents import simulator as S
    a, b = S.run(SPEC), S.run(SPEC)
    assert a["summary"] == b["summary"]
    c = S.run({**SPEC, "seed": 5})
    assert c["summary"] != a["summary"]


def test_simulation_rejects_bad_specs():
    from msagents import simulator as S
    with pytest.raises(S.SimError):
        S.run({"model": "zebrafish", "arms": [{}]})
    with pytest.raises(S.SimError, match="unknown parameter"):
        S.run({"model": "eae_mouse", "arms": [{"param_overrides": {"nonsense": 2}}]})
    with pytest.raises(S.SimError, match="unknown modifier"):
        S.run({"model": "eae_mouse", "arms": [{"interventions": [{"name": "aspirin"}]}]})


def test_custom_mechanism_changes_outcome():
    from msagents import simulator as S
    res = S.run({"model": "eae_mouse", "n_per_arm": 12, "seed": 1, "arms": [
        {"name": "vehicle"},
        {"name": "block CNS T reactivation", "interventions": [{"name": "custom", "effects": {"r_loc": 0.2}}]}]})
    assert res["comparisons"][0]["peak_score"]["diff"] < 0


def test_face_validity_benchmarks_pass():
    from msagents import simulator as S
    failed = [r["check"] for r in S.face_validity(n=12) if not r["pass"]]
    assert not failed


# ---- knowledge base: mechanisms and in-silico tests ---------------------------

def test_mechanism_links_need_sources_when_claimed_from_literature(lab):
    from msagents import blackboard as bb
    with pytest.raises(bb.BlackboardError, match="needs a source"):
        bb.add_mechanism(lab, "sb-1", "t", "s", [{"cause": "a", "effect": "b", "basis": "literature"}], [])
    cid = bb.add_mechanism(lab, "sb-1", "t", "s", [{"cause": "a", "effect": "b", "basis": "speculative"}], ["p"])
    assert cid == "C1"


def test_in_silico_test_end_to_end_and_half_reward(lab):
    from msagents import blackboard as bb, rewards, tools
    cid = bb.add_mechanism(lab, "sb-1", "entry", "s", [{"cause": "a4", "effect": "entry", "basis": "established"}], [])
    h = bb.propose_hypothesis(lab, "sb-1", "s", "m", "r", 0.5, "pred", [cid], basis="mechanistic")
    e = bb.preregister_experiment(lab, "design-1", h, "sim", "x", "p", "peak diff < 0", 0.9, "plan", simulation=SPEC)
    assert bb.get(lab, e)["body"]["dataset"] == "in_silico:eae_mouse"
    ctx = tools.Ctx(lab, "analyst-1", "analyst", 1, "toolu_abc123456789")
    out, err = tools.execute(ctx, "run_simulation", {"experiment_id": e})
    assert not err, out
    out = json.loads(out)
    assert out["purpose"] == "preregistered" and out["comparisons"][0]["peak_score"]["diff"] < 0
    r = bb.record_result(lab, "analyst-1", e, "supported", "d<0", "ok", code_path=out["saved_to"])
    bb.review(lab, "critic-1", r, "replicated", "same")
    scores = {x["agent"]: x["testing"] for x in rewards.leaderboard(lab)}
    assert scores["design-1"] == pytest.approx(0.5 * rewards.calibration(0.9, 1.0))
    assert bb.get(lab, h)["status"] == "supported_in_silico"


def test_exploratory_simulation_needs_no_experiment(lab):
    from msagents import tools
    out, err = tools.execute(tools.Ctx(lab, "sb-1", "systems_biologist", 1, "toolu_x"), "run_simulation", {"spec": SPEC})
    assert not err and json.loads(out)["purpose"] == "exploratory"


# ---- report ------------------------------------------------------------------

def test_report_lists_hypotheses_tests_and_mechanisms(lab):
    from msagents import blackboard as bb, report
    f = bb.add_finding(lab, "scout-1", "EBV raises MS risk", "PMID:35025605", "cohort")
    cid = bb.add_mechanism(lab, "sb-1", "Mimicry map", "s", [{"cause": "EBNA1", "effect": "GlialCAM IgG",
                                                              "basis": "literature", "source": "PMID:35073561"}], ["x"])
    h = bb.propose_hypothesis(lab, "hyp-1", "EBV drives MS", "mimicry", "r", 0.6, "pred", [f, cid], basis="mixed")
    path = report.write_report(lab)
    md = open(path).read()
    assert "EBV drives MS" in md and h in md and "Mimicry map" in md
    assert "https://pubmed.ncbi.nlm.nih.gov/35025605/" in md and "```mermaid" in md
    assert os.path.exists(os.path.join(os.environ["MSAGENTS_DATA"], "report.md"))


# ---- backends -----------------------------------------------------------------

TOOLS = [{"name": "add", "description": "add numbers",
          "input_schema": {"type": "object", "properties": {"a": {"type": "number"}}, "required": ["a"]}}]
CONVO = [
    {"role": "user", "content": "task"},
    {"role": "assistant", "content": [{"type": "thinking", "thinking": "x"}, {"type": "text", "text": "ok"},
                                      {"type": "tool_use", "id": "t1", "name": "add", "input": {"a": 1}}]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "2", "is_error": False}]},
]


def test_litellm_message_conversion_and_response(lab, monkeypatch):
    from msagents import llm
    msgs = llm.to_openai_messages("sys", CONVO)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool"]
    assert msgs[2]["tool_calls"][0]["function"] == {"name": "add", "arguments": '{"a": 1}'}
    assert msgs[3] == {"role": "tool", "tool_call_id": "t1", "content": "2"}

    seen = {}

    def completion(**kw):
        seen.update(kw)
        call = types.SimpleNamespace(id="call_9", function=types.SimpleNamespace(name="add", arguments='{"a": 5}'))
        msg = types.SimpleNamespace(content="thinking aloud", tool_calls=[call])
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=completion, drop_params=False))
    out = llm.LiteLLM("ollama_chat/qwen3:14b").complete("sys", CONVO, TOOLS)
    assert out["stop_reason"] == "tool_use"
    assert out["content"][1] == {"type": "tool_use", "id": "call_9", "name": "add", "input": {"a": 5}}
    assert seen["tools"][0]["function"]["name"] == "add"


def test_claude_code_backend_with_fake_cli(lab, tmp_path, monkeypatch):
    """A fake `claude` executable stands in for the real CLI."""
    from msagents import llm
    fake = tmp_path / "claude"
    fake.write_text("#!/bin/sh\ncat > \"$(dirname \"$0\")/stdin.txt\"\n"
                    "env > \"$(dirname \"$0\")/env.txt\"\n"
                    "echo '{\"type\":\"result\",\"is_error\":false,\"result\":\"\","
                    "\"structured_output\":{\"text\":\"adding\",\"tool_calls\":[{\"name\":\"add\",\"input\":{\"a\":3}}]}}'\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    out = llm.ClaudeCodeLLM(cli=str(fake)).complete("you are an agent", CONVO, TOOLS)
    assert out["stop_reason"] == "tool_use"
    assert out["content"][0] == {"type": "text", "text": "adding"}
    assert out["content"][1]["name"] == "add" and out["content"][1]["input"] == {"a": 3}
    sent = (tmp_path / "stdin.txt").read_text()
    assert "[tool_call id=t1] add" in sent and "[tool_result for t1]" in sent
    assert "sk-should-not-leak" not in (tmp_path / "env.txt").read_text()


def test_claude_code_usage_limit_wait():
    from msagents.llm import ClaudeCodeLLM
    assert ClaudeCodeLLM._limit_wait("Claude AI usage limit reached|9999999999") == 6 * 3600
    assert ClaudeCodeLLM._limit_wait("some other error") is None


def test_config_selects_backend(tmp_path):
    from msagents.coordinator import load_config
    raw = load_config("config/agents.yaml", backend="claude_code")
    cfg = raw["role_configs"]["critic"]
    assert cfg.backend == "claude_code" and cfg.model == "sonnet"
    raw = load_config("config/agents.yaml", backend="litellm", model="openai/gpt-5-mini")
    assert raw["role_configs"]["pi"].model == "openai/gpt-5-mini"
