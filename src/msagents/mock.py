"""Offline scripted model, so the whole pipeline can be run and tested
without an API key or network. Each role follows a fixed script.

The data it creates is SYNTHETIC and labelled as such. It demonstrates the
plumbing (messages, memory, pre-registration, replication, rewards), not science.
"""
from __future__ import annotations

import json
import uuid

SYNTH_CODE = r'''
# SYNTHETIC demo data - not real patients.
import csv, random
random.seed(42)
rows = []
for i in range(400):
    case = i < 200
    vitd = random.gauss(52 if case else 60, 15)       # nmol/L
    ebna = random.gauss(1.6 if case else 1.0, 0.5)     # anti-EBNA1 IgG index
    rows.append({"id": i, "ms_case": int(case), "vitamin_d": round(vitd, 1), "anti_ebna1": round(ebna, 2)})
import os; os.makedirs("datasets", exist_ok=True)
with open("datasets/synthetic_ms_cohort.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print("wrote", len(rows), "rows")
'''

TEST_CODE = r'''
import csv, statistics, math
rows = list(csv.DictReader(open("datasets/synthetic_ms_cohort.csv")))
case = [float(r["anti_ebna1"]) for r in rows if r["ms_case"] == "1"]
ctrl = [float(r["anti_ebna1"]) for r in rows if r["ms_case"] == "0"]
m1, m0 = statistics.mean(case), statistics.mean(ctrl)
s1, s0 = statistics.stdev(case), statistics.stdev(ctrl)
se = math.sqrt(s1**2/len(case) + s0**2/len(ctrl))
t = (m1 - m0) / se
d = (m1 - m0) / math.sqrt((s1**2 + s0**2) / 2)
p = math.erfc(abs(t) / math.sqrt(2))   # normal approx, two-sided
print(f"mean_case={m1:.3f} mean_ctrl={m0:.3f} t={t:.2f} cohen_d={d:.2f} p={p:.2e}")
print("SUPPORTED" if (d >= 0.5 and p < 0.001) else "REFUTED")
'''


def _tu(tool_name, **inp):
    return {"type": "tool_use", "id": f"toolu_mock_{uuid.uuid4().hex[:16]}", "name": tool_name, "input": inp}


def _text(t):
    return {"type": "text", "text": t}


def _last_results(messages) -> list[str]:
    last = messages[-1]
    if last["role"] != "user" or isinstance(last["content"], str):
        return []
    return [b.get("content", "") for b in last["content"] if b.get("type") == "tool_result"]


class MockLLM:
    def complete(self, system, messages, tools, ctx) -> dict:
        turn = sum(1 for m in messages if m["role"] == "assistant")
        refs = ctx.refs
        prev = _last_results(messages)
        role = ctx.role
        script = getattr(self, f"_{role}", self._default)
        blocks = script(turn, refs, prev, ctx)
        return {"content": blocks, "stop_reason": "tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn"}

    def _done(self, msg):
        return [_text(msg), _tu("complete_task", summary=msg)]

    def _default(self, turn, refs, prev, ctx):
        return [_text("Nothing scripted for this role.")] if turn else self._done("no-op")

    def _pi(self, turn, refs, prev, ctx):
        if turn == 0:
            return [_tu("view_scores"), _tu("search_knowledge", kind="hypothesis")]
        return [_tu("send_message", to="*", subject="Program update",
                    body="Keep citing each other's artifacts; prioritise replication."),
                *self._done("Reviewed leaderboard and broadcast priorities.")]

    def _literature_scout(self, turn, refs, prev, ctx):
        if turn == 0:
            return [
                _tu("add_finding", claim="EBV infection is associated with a ~32-fold higher MS risk in a military cohort",
                    source="PMID:35025605", evidence_level="cohort",
                    details="Bjornevik et al., Science 2022; 10M US military personnel, serial sera."),
                _tu("add_finding", claim="Antibodies to EBNA1 cross-react with GlialCAM in a subset of MS patients",
                    source="PMID:35073561", evidence_level="in-vitro",
                    details="Lanz et al., Nature 2022; molecular mimicry mechanism."),
            ]
        if turn == 1:
            return [_tu("remember", kind="lesson", content="EBV evidence is strongest from prospective serology cohorts.")]
        return self._done("Added 2 findings on EBV.")

    def _data_curator(self, turn, refs, prev, ctx):
        if turn == 0:
            return [_tu("run_analysis", code=SYNTH_CODE)]
        if turn == 1:
            return [_tu("add_dataset", name="SYNTHETIC MS cohort (demo)", local_path="datasets/synthetic_ms_cohort.csv",
                        description="Synthetic 200 cases / 200 controls for pipeline testing only.",
                        variables="ms_case, vitamin_d, anti_ebna1")]
        return self._done("Registered synthetic demo dataset.")

    def _hypothesis_generator(self, turn, refs, prev, ctx):
        if turn == 0:
            return [_tu("search_knowledge", kind="finding"), _tu("search_knowledge", kind="dataset")]
        if turn == 1:
            ids = [a["id"] for p in prev for a in json.loads(p)]
            return [_tu("propose_hypothesis",
                        statement="MS cases have higher anti-EBNA1 IgG than controls",
                        mechanism="EBNA1 antibodies cross-react with CNS GlialCAM (molecular mimicry)",
                        rationale="Prospective EBV cohort + mimicry mechanism", prior_probability=0.7,
                        testable_prediction="anti-EBNA1 index higher in cases, Cohen's d >= 0.5",
                        cites=ids)]
        return self._done("Proposed one hypothesis.")

    def _experiment_designer(self, turn, refs, prev, ctx):
        if turn == 0:
            return [_tu("preregister_experiment", hypothesis_id=refs[0], method="Welch t-test",
                        dataset="datasets/synthetic_ms_cohort.csv",
                        prediction="anti_ebna1 higher in ms_case=1",
                        success_criteria="Cohen's d >= 0.5 and two-sided p < 0.001",
                        predicted_probability=0.8,
                        analysis_plan="Compare anti_ebna1 between groups; report d and p; no covariates.")]
        return self._done("Pre-registered experiment.")

    def _analyst(self, turn, refs, prev, ctx):
        if turn == 0:
            return [_tu("get_artifact", id=refs[0]), _tu("run_analysis", code=TEST_CODE)]
        if turn == 1:
            out = json.loads(prev[1])
            outcome = "supported" if "SUPPORTED" in out["stdout"] else "refuted"
            return [_tu("record_result", experiment_id=refs[0], outcome=outcome,
                        statistics=out["stdout"].splitlines()[0], summary="Ran the locked plan exactly.",
                        code_path=out["script_path"], deviations=[])]
        if turn == 2:
            rid = json.loads(prev[0])["id"]
            return [_tu("send_message", to="role:critic", kind="request", subject=f"Please replicate {rid}",
                        body=f"Script path is in {rid}. It runs in <1s.", refs=[rid])]
        return self._done("Experiment run and result recorded.")

    def _critic(self, turn, refs, prev, ctx):
        if turn == 0:
            return [_tu("read_inbox"), _tu("get_artifact", id=refs[0])]
        if turn == 1:
            ctx.scratch["inbox"] = json.loads(prev[0])
            result = json.loads(prev[1])
            ctx.scratch["orig"] = result["body"]["outcome"]
            return [_tu("run_analysis", script_path=result["body"]["code_path"])]
        if turn == 2:
            out = json.loads(prev[0])
            got = "supported" if "SUPPORTED" in out["stdout"] else "refuted"
            orig = ctx.scratch.get("orig", got)
            verdict = "replicated" if got == orig else "flawed"
            blocks = [_tu("review", target_id=refs[0], verdict=verdict, replicated_outcome=got,
                          comments=f"Re-ran script: {out['stdout'].splitlines()[0]}. Data is SYNTHETIC.")]
            for m in ctx.scratch.get("inbox", []):
                blocks.append(_tu("rate_message", message_id=m["id"], rating=0.9))
                if m["kind"] == "request":
                    blocks.append(_tu("send_message", to=m["from"], kind="reply", reply_to=m["id"],
                                      subject="Replicated", body=f"Verdict on {refs[0]}: {verdict}"))
            return blocks
        return self._done("Review submitted.")
