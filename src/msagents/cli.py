"""Command line: msagents <command>."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from . import blackboard, bus, memory, rewards, tasks
from .coordinator import DEFAULT_CONFIG, agent_process, load_config, make_llm, run_lab
from .db import DB_PATH, connect, init_db


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="msagents")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="create the database")
    r = sub.add_parser("run", help="start the coordinator and all agents")
    r.add_argument("--mock", action="store_true", help="offline scripted agents, no API calls")
    r.add_argument("--only", nargs="*", help="only start these roles or agent names")
    r.add_argument("--topics", type=int, help="use only the first N agenda topics")
    r.add_argument("--until-idle", action="store_true", help="exit when no work is left")
    r.add_argument("--no-report", action="store_true", help="don't write data/report.md at the end")
    a = sub.add_parser("agent", help="run a single agent in the foreground (for debugging)")
    a.add_argument("name")
    a.add_argument("--mock", action="store_true")
    for p_ in (r, a):
        p_.add_argument("--backend", choices=["anthropic", "claude_code", "litellm"],
                        help="model backend for all agents (default: config file)")
        p_.add_argument("--model", help="model name for that backend, e.g. sonnet or ollama_chat/qwen3:14b")
    rp = sub.add_parser("report", help="write a human-readable report to data/report.md")
    rp.add_argument("--narrative", action="store_true", help="add a plain-language summary written by the model")
    rp.add_argument("--backend", choices=["anthropic", "claude_code", "litellm"])
    rp.add_argument("--model")
    sm = sub.add_parser("simulate", help="run the virtual mouse/patient model")
    sm.add_argument("spec", nargs="?", help="JSON spec file (see `msagents simulate --describe`)")
    sm.add_argument("--describe", action="store_true", help="print models, parameters and drugs")
    sm.add_argument("--check", action="store_true", help="run the face-validity benchmarks")
    sub.add_parser("status", help="tasks and artifacts overview")
    sub.add_parser("scores", help="reward leaderboard")
    s = sub.add_parser("show", help="print one artifact")
    s.add_argument("id")
    m = sub.add_parser("memory", help="print an agent's memory")
    m.add_argument("agent")
    i = sub.add_parser("messages", help="recent messages on the bus")
    i.add_argument("--limit", type=int, default=20)
    t = sub.add_parser("add-task", help="put a task in the queue by hand")
    t.add_argument("role")
    t.add_argument("title")
    t.add_argument("--details", default="")
    t.add_argument("--refs", nargs="*", default=[])
    sub.add_parser("retry-failed", help="put failed tasks back in the queue")

    args = ap.parse_args(argv)
    level = "DEBUG" if args.verbose else "INFO"
    logging.basicConfig(level=level, format="%(asctime)s %(name)-22s %(message)s", datefmt="%H:%M:%S")

    if args.cmd == "init":
        init_db()
        print(f"database ready at {DB_PATH}")
        return
    if args.cmd == "run":
        run_lab(args.config, args.mock, args.only, args.topics, args.until_idle, level,
                backend=args.backend, model=args.model, write_report=not args.no_report)
        return
    if args.cmd == "agent":
        init_db()
        agent_process(args.name, str(args.config), args.mock, level, args.backend, args.model)
        return
    if args.cmd == "simulate":
        from . import simulator
        if args.describe:
            print(json.dumps(simulator.describe(), indent=2))
        elif args.check:
            for row in simulator.face_validity():
                print(f"{'PASS' if row['pass'] else 'FAIL'}  {row['check']}  "
                      f"({row['metric']}: {row['control']} -> {row['treated']}, d={row['cohen_d']})")
        elif args.spec:
            res = simulator.run(json.loads(Path(args.spec).read_text()))
            res.pop("curves")
            print(json.dumps(res, indent=2))
        else:
            ap.error("give a spec file, --describe or --check")
        return

    init_db()
    conn = connect()
    if args.cmd == "status":
        print("TASKS")
        for row in conn.execute("SELECT id, role, status, owner, attempts, title FROM tasks ORDER BY id"):
            print(f"  #{row['id']:<4} {row['status']:<8} {row['role']:<21} {row['owner'] or '':<24} {row['title'][:60]}")
        print("\nARTIFACTS")
        for row in conn.execute("SELECT id, kind, author, status, title FROM artifacts ORDER BY created_at"):
            print(f"  {row['id']:<5} {row['kind']:<11} {row['status']:<22} {row['author']:<24} {row['title'][:60]}")
    elif args.cmd == "scores":
        rewards.sweep(conn)
        print(f"{'agent':<26}{'collab':>9}{'testing':>9}{'total':>9}")
        for r in rewards.leaderboard(conn):
            print(f"{r['agent']:<26}{r['collab']:>9.2f}{r['testing']:>9.2f}{r['total']:>9.2f}")
    elif args.cmd == "show":
        print(json.dumps(blackboard.get(conn, args.id), indent=2, ensure_ascii=False))
    elif args.cmd == "memory":
        print(memory.digest(conn, args.agent, limit=50))
    elif args.cmd == "messages":
        for row in conn.execute("SELECT * FROM messages ORDER BY id DESC LIMIT ?", (args.limit,)):
            print(f"#{row['id']} {row['from_agent']} -> {row['to_agent']} [{row['kind']}] {row['subject']}\n    {row['body'][:200]}")
    elif args.cmd == "add-task":
        print("task", tasks.create_task(conn, args.role, args.title, args.details, args.refs, created_by="human"))
    elif args.cmd == "report":
        from .report import write_report
        llm = None
        if args.narrative:
            raw = load_config(args.config, args.backend, args.model)
            llm = make_llm(raw["role_configs"]["pi"], mock=False)
        path = write_report(conn, llm)
        print(f"report written to {path}\n(latest copy: {DB_PATH.parent / 'report.md'})")
    elif args.cmd == "retry-failed":
        print(tasks.requeue_failed(conn), "tasks requeued")


if __name__ == "__main__":
    main()
