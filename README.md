# MS Research Agents

A local multi-agent system that investigates possible causes of, and
treatments for, multiple sclerosis using public literature and public data.

Eight agents, each with one job (literature, data, mechanistic biology,
hypotheses, test design, analysis, independent review, direction), share a
knowledge base and a message bus, keep long-term memory, resume interrupted
work from checkpoints, and are scored on **collaboration** and on **how well
their hypothesis tests hold up**.

Ideas come from two places: published studies, and a `systems_biologist`
that reasons from how the immune system, the blood-brain barrier, myelin and
neurons work. Hypotheses can be tested on public data or on a **virtual
mouse / virtual patient** (a mechanistic simulation, see below). When a run
ends the team writes a readable report to `data/report.md`.

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

# 2. Real run (pick a backend, see next section)
msagents run --backend claude_code --topics 1     # your Claude Pro/Max login
msagents run --topics 2                           # Claude API key (default)
msagents run --backend litellm --model ollama_chat/qwen3:14b --topics 1

# 3. Read what happened
msagents report              # data/report.md (also written automatically when a run ends)
msagents report --narrative  # + plain-language summary written by the model
```

## Choosing a model backend

| Backend | Needs | Good for | Caveats |
|---|---|---|---|
| `anthropic` (default) | `ANTHROPIC_API_KEY` | best quality, prompt caching, long runs | pay per token |
| `claude_code` | Claude Pro/Max login (`claude` CLI installed, `/login` done) | no API key, no extra bill | Pro usage limits run out fast with 8 agents; agents wait for the limit to reset. Slower (one CLI call per turn). Personal use only. |
| `litellm` | `pip install -e '.[litellm]'` + a provider key, or local Ollama | free local models, other providers | model must support tool calling; small local models make weaker scientists |

Set it per run (`--backend`, `--model`), with `MSAGENTS_BACKEND` /
`MSAGENTS_MODEL`, or in `config/agents.yaml` (`defaults.backend` and the
`backends:` section). With `claude_code`, at most `max_parallel` (default 2)
model calls run at once, and the CLI never sees `ANTHROPIC_API_KEY`, so it
always uses your subscription. Tip for Pro: start small, e.g.
`--only systems_biologist experiment_designer analyst critic` with a task
added by hand.

## The virtual mouse / virtual patient

`src/msagents/simulator.py` is a mechanistic model (13 coupled equations)
of the processes thought to drive MS: T/B cell priming in lymph nodes,
EBV molecular mimicry, regulatory T cells, entry through the blood-brain
barrier, microglial activation, demyelination, oligodendrocyte
remyelination and irreversible axon loss. Each virtual subject gets its own
parameters, so a run is a small randomised experiment with statistics.

| Model | Mimics | Read-outs |
|---|---|---|
| `eae_mouse` | MOG35-55 EAE in C57BL/6 mice | clinical score 0-5, onset, myelin, axons |
| `cuprizone_mouse` | toxic demyelination + recovery | remyelination |
| `virtual_patient` | relapsing MS over 5 years, EBV, infections | relapse rate, lesions, EDSS-like disability, atrophy |

Agents can give 13 drugs (natalizumab, anti-CD20, BTK inhibitors,
clemastine, metformin, aHSCT, EBV-directed T cells...), 8 risk factors
(HLA-DRB1*15:01, low vitamin D, smoking, EBV status, ageing...), and
**custom interventions** that change any parameter, which is how a new
mechanism gets tested. Pre-registered in-silico experiments are locked like
real ones; the critic re-runs them with a second seed.

```bash
msagents simulate --check      # 10 known drug/biology effects the model must reproduce
msagents simulate --describe   # every parameter, drug and condition
msagents simulate spec.json    # run your own experiment
```

**Limits.** Parameters are hand-tuned to reproduce the *direction* of
well-known results, not fitted to patient data; effect sizes are larger than
in real trials. An in-silico result says "this mechanism, as encoded,
implies X". It's a cheap way to sharpen or discard an idea before real data
or a real experiment, never evidence about patients. In-silico tests earn
half the testing reward for this reason.

## Commands

| Command | What it does |
|---|---|
| `msagents run [--mock] [--backend B] [--model M] [--topics N] [--only ROLE ...] [--until-idle]` | Start coordinator + all agents |
| `msagents report [--narrative]` | Write the human-readable report |
| `msagents simulate [--check \| --describe \| spec.json]` | Use the virtual mouse/patient directly |
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
  simulator.py            virtual EAE mouse, cuprizone mouse, virtual patient
  report.py               human-readable report (data/report.md)
  coordinator.py          process supervisor, agenda scheduler, reward sweep
  db.py                   SQLite schema (one file: data/lab.db)
  tasks.py                task queue with leases
  bus.py                  message bus
  blackboard.py           shared knowledge base + pipeline triggers
  memory.py               long-term memory, checkpoints, tool-call log
  rewards.py              scoring rules and ledger
  tools.py                tool definitions
  sandbox.py              runs analysis scripts
  llm.py / mock.py        backends: Claude API, Claude Pro via CLI, LiteLLM / offline mock
  sources/europepmc.py    literature search
tests/                    resume, pre-registration and reward tests
data/                     created at runtime (db, datasets, scripts)
```

## Cost and safety notes

- With the `anthropic` backend every role uses `claude-opus-5-5` by default. A full agenda topic is
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
