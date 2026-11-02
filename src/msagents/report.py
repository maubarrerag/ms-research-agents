"""Human-readable report of what the team did and found.

Written automatically when `msagents run` ends, or on demand with
`msagents report`. Output: data/reports/report-<time>.md and data/report.md
(always the latest). Open it in any Markdown viewer (VS Code: Ctrl+Shift+V);
the mechanism diagrams use Mermaid, which GitHub and VS Code render.

Everything here is assembled from the knowledge base by plain code, so it
can't invent anything. `--narrative` adds a plain-language summary written
by the model on top, clearly marked as such.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from collections import Counter, defaultdict

from . import blackboard, rewards
from .db import DATA_DIR

VERDICT_WORDS = {
    "replicated": "independently re-run and confirmed",
    "valid": "checked and judged sound (not re-run)",
    "flawed": "judged flawed by the reviewer - do not rely on it",
    "inconclusive": "reviewer could not decide",
}
OUTCOME_WORDS = {"supported": "SUPPORTED", "refuted": "REFUTED", "inconclusive": "INCONCLUSIVE"}
BASIS_WORDS = {"literature": "from published studies", "mechanistic": "reasoned from biology / physiology",
               "data": "from data analysis", "mixed": "literature + mechanistic reasoning"}


def _all(conn, kind):
    rows = conn.execute("SELECT id FROM artifacts WHERE kind=? ORDER BY created_at", (kind,)).fetchall()
    return [blackboard.get(conn, r["id"]) for r in rows]


def _link(source: str) -> str:
    s = (source or "").strip()
    m = re.match(r"(?i)pmid:?\s*(\d+)", s)
    if m:
        return f"[PMID {m.group(1)}](https://pubmed.ncbi.nlm.nih.gov/{m.group(1)}/)"
    m = re.match(r"(?i)(?:doi:?\s*)?(10\.\S+)", s)
    if m:
        return f"[doi:{m.group(1)}](https://doi.org/{m.group(1)})"
    return s or "no source"


def _status(st: str) -> str:
    parts = st.split("_")
    base = parts[0]
    notes = []
    if "unreplicated" in parts:
        notes.append("not replicated")
    if "silico" in parts:
        notes.append("in silico")
    if base == "under":
        return "under test"
    return base + (f" ({', '.join(notes)})" if notes else "")


def _esc(t) -> str:
    return str(t or "").replace("|", "/").replace("\n", " ").strip()


def _children(conn, parent_id, kind):
    rows = conn.execute(
        "SELECT a.id FROM citations c JOIN artifacts a ON a.id=c.from_id WHERE c.to_id=? AND a.kind=? "
        "ORDER BY a.created_at", (parent_id, kind)).fetchall()
    return [blackboard.get(conn, r["id"]) for r in rows]


def _sim_table(result: dict) -> list[str]:
    """Arm-by-arm numbers for an in-silico result, read from the saved simulation file."""
    path = result["body"].get("code_path") or ""
    if not path.endswith(".json"):
        return []
    try:
        sim = json.loads((DATA_DIR / path).read_text())
    except (OSError, json.JSONDecodeError):
        return []
    metrics = [k for k in sim["summary"][0] if isinstance(sim["summary"][0][k], dict)]
    out = ["", f"Virtual experiment (`{sim['model']}`, {sim['summary'][0]['n']} virtual subjects per arm, "
           f"seed {sim['spec']['seed']}):", "",
           "| arm | " + " | ".join(metrics) + " |", "|---" * (len(metrics) + 1) + "|"]
    for arm in sim["summary"]:
        out.append(f"| {_esc(arm['arm'])} | " + " | ".join(
            "-" if arm[m]["mean"] is None else f"{arm[m]['mean']:.3g}" for m in metrics) + " |")
    return out


def _mermaid(mech: dict) -> list[str]:
    nodes, lines = {}, ["```mermaid", "flowchart LR"]

    def nid(label):
        if label not in nodes:
            nodes[label] = f"n{len(nodes)}"
            lines.append(f'    {nodes[label]}["{label.replace(chr(34), chr(39))[:60]}"]')
        return nodes[label]

    style = {"established": "-->", "literature": "-->", "inferred": "-.->", "speculative": "-.->"}
    for link in mech["body"]["causal_chain"]:
        a, b = nid(link["cause"]), nid(link["effect"])
        lines.append(f'    {a} {style.get(link["basis"], "-->")}|{link["relation"]} / {link["basis"]}| {b}')
    lines.append("```")
    return lines


def build(conn, narrative: str | None = None) -> str:
    rewards.sweep(conn)
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    findings, datasets, mechs = _all(conn, "finding"), _all(conn, "dataset"), _all(conn, "mechanism")
    hyps = _all(conn, "hypothesis")
    task_counts = Counter(r["status"] for r in conn.execute("SELECT status FROM tasks"))

    L = [f"# MS research team report", "", f"_Generated {now}. Research-support output for scientists to "
         "check - not medical advice._", ""]

    # --- at a glance
    tested = []
    for h in hyps:
        for e in _children(conn, h["id"], "experiment"):
            for r in _children(conn, e["id"], "result"):
                revs = _children(conn, r["id"], "review")
                tested.append((h, e, r, revs[-1] if revs else None))
    trusted = [t for t in tested if t[3] and t[3]["body"]["verdict"] in ("replicated", "valid")]
    L += ["## At a glance", "",
          f"- **{len(hyps)} hypotheses** proposed "
          f"({sum(1 for h in hyps if h['body'].get('basis') in ('mechanistic', 'mixed'))} from mechanistic reasoning)",
          f"- **{len(tested)} tests** run, **{len(trusted)}** passed independent review "
          f"({sum(1 for t in trusted if t[2]['body']['outcome'] == 'supported')} supported, "
          f"{sum(1 for t in trusted if t[2]['body']['outcome'] == 'refuted')} refuted)",
          f"- {sum(1 for t in tested if t[1]['body'].get('simulation'))} of the tests were in silico "
          "(virtual mouse/patient)",
          f"- Evidence base: {len(findings)} literature findings, {len(datasets)} datasets, "
          f"{len(mechs)} mechanism maps",
          f"- Tasks: {task_counts.get('done', 0)} done, {task_counts.get('running', 0)} running, "
          f"{task_counts.get('pending', 0)} waiting, {task_counts.get('failed', 0)} failed", ""]

    if narrative:
        L += ["## Summary in plain language", "", "_Written by the model from the sections below; check the "
              "details before relying on it._", "", narrative.strip(), ""]

    # --- headline table
    if hyps:
        L += ["## Hypotheses", "", "| id | hypothesis | where it came from | prior | status |", "|---|---|---|---|---|"]
        for h in hyps:
            L.append(f"| {h['id']} | {_esc(h['body']['statement'])} | "
                     f"{BASIS_WORDS.get(h['body'].get('basis', 'literature'), '')} | "
                     f"{h['body']['prior_probability']:.0%} | {_status(h['status'])} |")
        L.append("")

    # --- each hypothesis in detail
    for h in hyps:
        b = h["body"]
        L += [f"### {h['id']}: {b['statement']}", "",
              f"**Proposed by** {h['author']}, {BASIS_WORDS.get(b.get('basis', 'literature'), '')}. "
              f"Honest prior: {b['prior_probability']:.0%}.", "",
              f"**Proposed mechanism.** {b['mechanism']}", "",
              f"**Why.** {b['rationale']}", "",
              f"**What it predicts.** {b['testable_prediction']}", ""]
        if h["cites"]:
            L.append("**Built on:**")
            for cid in h["cites"]:
                c = blackboard.get(conn, cid)
                if not c:
                    continue
                if c["kind"] == "finding":
                    L.append(f"- {cid} ({c['body']['evidence_level']}): {c['body']['claim']} - {_link(c['body']['source'])}")
                elif c["kind"] == "mechanism":
                    L.append(f"- {cid} (mechanism map): {c['title']}")
                else:
                    L.append(f"- {cid} ({c['kind']}): {c['title']}")
            L.append("")
        exps = _children(conn, h["id"], "experiment")
        if not exps:
            L += ["_Not tested yet._", ""]
        for e in exps:
            eb = e["body"]
            kind = "In-silico test" if eb.get("simulation") else "Test"
            L += [f"**{kind} {e['id']}** (designed by {e['author']}, locked before analysis): {eb['method']} on "
                  f"`{eb['dataset']}`.", "",
                  f"- Success if: {eb['success_criteria']}",
                  f"- Designer's forecast that it would succeed: {eb['predicted_probability']:.0%}"]
            results = _children(conn, e["id"], "result")
            if not results:
                L += ["- _Not run yet._", ""]
            for r in results:
                rb = r["body"]
                L += [f"- **Result {r['id']}: {OUTCOME_WORDS.get(rb['outcome'], rb['outcome'])}** - {rb['summary']}",
                      f"- Statistics: {rb['statistics']}"]
                if rb.get("deviations"):
                    L.append(f"- Deviations from plan: {'; '.join(rb['deviations'])}")
                for v in _children(conn, r["id"], "review"):
                    vb = v["body"]
                    L.append(f"- **Review {v['id']}** by {v['author']}: {VERDICT_WORDS.get(vb['verdict'], vb['verdict'])}. "
                             f"{vb['comments']}")
                L += _sim_table(r)
                if eb.get("simulation"):
                    L += ["", "> In-silico results show what the mechanism implies inside a hand-tuned model. "
                          "They are a reason to run (or skip) a real experiment, not evidence about patients."]
            L.append("")

    # --- mechanism maps
    if mechs:
        L += ["## Mechanism maps (reasoned from biology)", "",
              "Solid arrows: established or published links. Dotted arrows: inferred or speculative.", ""]
        for m in mechs:
            mb = m["body"]
            L += [f"### {m['id']}: {m['title']}", "", mb["summary"], ""] + _mermaid(m) + [""]
            L.append("| cause | relation | effect | how well known | source |")
            L.append("|---|---|---|---|---|")
            for link in mb["causal_chain"]:
                L.append(f"| {_esc(link['cause'])} | {link['relation']} | {_esc(link['effect'])} | {link['basis']} | "
                         f"{_link(link['source']) if link['source'] else ''} |")
            if mb.get("predictions"):
                L += ["", "**Predictions:**"] + [f"- {p}" for p in mb["predictions"]]
            if mb.get("gaps"):
                L += ["", f"**Weakest links / gaps:** {mb['gaps']}"]
            L.append("")

    # --- evidence
    if findings:
        L += ["## Literature findings", ""]
        by_level = defaultdict(list)
        for f in findings:
            by_level[f["body"]["evidence_level"]].append(f)
        order = ["meta-analysis", "mendelian-randomization", "rct", "cohort", "case-control", "animal", "in-vitro",
                 "review", "other"]
        for lvl in sorted(by_level, key=lambda x: order.index(x) if x in order else 99):
            L.append(f"**{lvl}**")
            for f in by_level[lvl]:
                L.append(f"- {f['id']}: {f['body']['claim']} - {_link(f['body']['source'])}")
            L.append("")
    if datasets:
        L += ["## Datasets", ""]
        for d in datasets:
            L.append(f"- {d['id']}: **{d['title']}** - {d['body']['description']} (`{d['body']['local_path']}`)")
        L.append("")

    # --- problems
    failed = conn.execute("SELECT id, role, title, result FROM tasks WHERE status='failed' ORDER BY id").fetchall()
    open_h = [h for h in hyps if h["status"] in ("open",)]
    if failed or open_h:
        L += ["## Open items", ""]
        for h in open_h:
            L.append(f"- {h['id']} has no test yet.")
        for t in failed:
            L.append(f"- Task #{t['id']} ({t['role']}) failed: {t['title']} - {_esc(t['result'])[:200]}")
        L.append("")

    # --- team
    board = rewards.leaderboard(conn)
    if board:
        L += ["## Team scores", "", "Testing = calibrated forecasts and rigor; collaboration = work others used.", "",
              "| agent | collaboration | testing | total |", "|---|---|---|---|"]
        for r in board:
            L.append(f"| {r['agent']} | {r['collab']:.2f} | {r['testing']:.2f} | {r['total']:.2f} |")
        L.append("")

    L += ["---", "_How to read this: a hypothesis is only as good as its test. Trust results marked "
          "'independently re-run and confirmed' most, and treat in-silico results as leads, not findings._"]
    return "\n".join(L) + "\n"


def narrative_summary(conn, llm, report_md: str) -> str:
    system = ("You write plain-language summaries of research reports for a smart reader who is not a "
              "scientist. Be accurate and cautious. Never add facts that are not in the report.")
    prompt = ("Summarise this report in at most 250 words: what the team looked at, what held up under "
              "independent review, what was refuted, which ideas came from biology reasoning, and what "
              "should be done next. Say clearly that in-silico results are model predictions, not evidence. "
              "Plain text, no headings.\n\n" + report_md[:60000])
    resp = llm.complete(system, [{"role": "user", "content": prompt}], [], None)
    return " ".join(b.get("text", "") for b in resp["content"] if b.get("type") == "text").strip()


def write_report(conn, llm=None) -> str:
    md = build(conn)
    if llm is not None:
        md = build(conn, narrative_summary(conn, llm, md))
    out_dir = DATA_DIR / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"report-{stamp}.md"
    path.write_text(md)
    (DATA_DIR / "report.md").write_text(md)
    return str(path)
