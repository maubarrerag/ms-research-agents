# MS Research Agents

A local multi-agent system that investigates possible causes of, and
treatments for, multiple sclerosis using public literature and public data.

Seven agents, each with one job (literature, data, hypotheses, test design,
analysis, independent review, direction), share a knowledge base and a
message bus, keep long-term memory, resume interrupted work from
checkpoints, and are scored on **collaboration** and on **how well their
hypothesis tests hold up**.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the design.

> This is a research-support tool. Its output is hypotheses and analyses
> for human scientists to check, not medical advice.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,analysis]'

# 1. Offline demo: scripted agents, synthetic data, no API key, ~10 seconds
msagents run --mock --topics 1 --until-idle
msagents status
msagents scores

# 2. Real run with Claude
export ANTHROPIC_API_KEY=...        # or: ant auth login
rm -rf data                          # start fresh (optional)
msagents run --topics 2              # Ctrl+C to stop; run again to resume
```

## Commands

| Command | What it does |
|---|---|
| `msagents run [--mock] [--topics N] [--only ROLE ...] [--until-idle]` | Start coordinator + all agents |
| `msagents agent NAME [--mock]` | Run one agent in the foreground (debugging) |
| `msagents status` | Tasks and artifacts |
| `msagents scores` | Reward leaderboard |
| `msagents show H1` | One artifact in full |
| `msagents messages` | Recent messages between agents |
| `msagents memory critic-1` | An agent's long-term memory |
| `msagents add-task ROLE "title" --details ... --refs H1` | Give the team work by hand |
| `msagents retry-failed` | Re-queue failed tasks |

## Testing interruption

```bash
msagents run --topics 1      # wait until some task is running, then Ctrl+C (or kill -9 a worker)
msagents status              # task shows "running" with its owner
msagents run --topics 1      # the owner resumes it from its last checkpoint
pytest                       # includes a crash-and-resume test
```

## Layout

```
config/agents.yaml        roles, prompts, tools, model/effort, research agenda
src/msagents/
  agent.py                agent loop: claim -> resume -> model -> tools -> checkpoint
  coordinator.py          process supervisor, agenda scheduler, reward sweep
  db.py                   SQLite schema (one file: data/lab.db)
  tasks.py                task queue with leases
  bus.py                  message bus
  blackboard.py           shared knowledge base + pipeline triggers
  memory.py               long-term memory, checkpoints, tool-call log
  rewards.py              scoring rules and ledger
  tools.py                tool definitions
  sandbox.py              runs analysis scripts
  llm.py / mock.py        Claude backend / offline scripted backend
  sources/europepmc.py    literature search
tests/                    resume, pre-registration and reward tests
data/                     created at runtime (db, datasets, scripts)
```

## Cost and safety notes

- Every role uses `claude-opus-5-5` by default. A full agenda topic is
  dozens of model turns per agent; start with `--topics 1` and check
  `msagents status` before going wider. Lower `effort` for cheaper roles in
  `config/agents.yaml`.
- Agents execute their own Python code. `run_analysis` is a subprocess with
  a timeout, **not** an isolation boundary. For unattended runs, use a
  container, e.g.:

  ```bash
  docker run --rm -it -v "$PWD":/lab -w /lab -e ANTHROPIC_API_KEY python:3.12 \
    bash -c "pip install -e '.[analysis]' && msagents run --topics 1"
  ```
