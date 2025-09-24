# Never-Again Plan (for me + you, every project)

## 1) Collaboration Contract (non-negotiables)

* **Single source of truth for env/ports:** one `.env` next to the compose file. I will *never* override with shell vars.
* **No refactors or dep changes** unless you explicitly say “change deps” or “refactor.” My default is **tests + glue only**.
* **Full-file drops only when asked.** Otherwise: diffs are advisory, not applied.
* **Deterministic-first:** one tiny smoke must pass before touching Docker/healthchecks/networks.
* **Service minimalism:** bring up *one* service until green; others are parked.
* **Provider reality:** smokes use a **real model** or we add a 3-line `"dummy"→DEFAULT_MODEL` shim first.
* **Healthchecks are /ready, not real calls.** Real provider pings live in ndsmokes, not readiness.
* **zsh-safe commands only:** no comments in copy/paste lines, no parentheses in headings, JSON correctly quoted.

> If I need to breach any rule, I’ll say: “request to violate rule X because Y” and wait for your OK.

## 2) Session Kickoff Checklist (every time, 90 seconds)

I will confirm, in one block, before doing anything:

* **Goal** (1 sentence), **Scope** (files I can touch), **Out of scope**, **Success signal** (one command that returns ✅).
* **Env**: exact compose path, `.env` path, ports, provider base/model.
* **Proof plan**: the single smoke I’ll make pass (file path + exact pytest incantation).
* **Abort criteria**: if smoke isn’t green in N minutes/steps, stop and snapshot STATUS.

## 3) Change Request format (you hand me; I obey)

```
Title: [verb] [file paths] – [goal]
Files to edit: [exact list]
Must NOT change: [deps | compose | healthchecks | provider]
Acceptance: [exact CLI/curl/pytest line expected to pass]
```

If anything is ambiguous, I restate it; otherwise I act—no “optimizing.”

## 4) Test Strategy (guardrails)

* **Deterministic smoke (required gate):**

  * Asserts **shape only** (keys + ok\:true), no network; skips cleanly if port closed.
  * Example file: `tests/smoke/test_agent_api_shape.py`
* **Live ndsmokes (opt-in):**

  * Env-gated, assert shape only, timeouts conservative, skip when env missing.
* **Ready route:**

  * `/ready` returns `{ok:true}` without touching models; healthchecks hit this only.

## 5) Docker & Networking (landmine policy)

* **Ports:** pinned in `.env` only. If compose resolves a different port, I stop and fix the `.env`, not the shell.
* **Networking:** if two containers talk, use a named network and service names (no host hairpins).
* **Uvicorn start:** `python -m uvicorn package.module:app` (no `cd`, no relative imports).
* **Healthchecks:** never call real providers; if none exists, I add `/ready` rather than bending a smoke.

## 6) Dependencies (end the whack-a-mole)

* One **`requirements.runtime.txt`** derived from your `pyproject.toml`. Installed first.
* Your package installed with `pip install -e . --no-deps`.
* I will not trim this list unless you explicitly say to.

## 7) Failure Playbook (to prevent spiral)

* On first failure I do exactly three things, in order:

  1. **Show “what compose resolved”**: `docker compose -f <file> config --services` and the `agent` block.
  2. **Show ports and last 120 lines of logs** for the one service in scope.
  3. **Run the single smoke** and paste pass/fail.
* If still stuck, I snapshot **STATUS\_NOW\.md** (state, commands, logs), stop, and ask for direction—no “try another thing.”

---

# Concrete artifacts you can adopt (templates)

## A. `/ready` endpoint (FastAPI)

```python
# app/ready.py
from fastapi import APIRouter
router = APIRouter()
@router.get("/ready")
async def ready(): return {"ok": True}
```

```python
# app/main.py (or agent_proxy.py)
from app.ready import router as ready_router
app.include_router(ready_router)
```

## B. Deterministic smoke (shape-only)

```python
# tests/smoke/test_agent_api_shape.py
import os, socket, httpx, pytest
def can(h,p,t=0.5):
    try: 
        with socket.create_connection((h,p),timeout=t): return True
    except OSError: return False
@pytest.mark.timeout(15)
def test_agent_api_shape():
    h=os.getenv("MINI_AGENT_API_HOST","127.0.0.1")
    p=int(os.getenv("MINI_AGENT_API_PORT","8788"))
    if not can(h,p): pytest.skip(f"Agent not reachable on {h}:{p}")
    r=httpx.get(f"http://{h}:{p}/ready",timeout=6.0); r.raise_for_status()
    r=httpx.post(f"http://{h}:{p}/agent/run",json={"messages":[{"role":"user","content":"hi"}],
        "model":"ollama/qwen3:8b","tool_backend":"local","use_tools":False,
        "api_base":os.getenv("OLLAMA_URL","http://ollama:11434"),
        "base_url":os.getenv("OLLAMA_URL","http://ollama:11434")},timeout=12.0)
    d=r.json()
    for k in("ok","final_answer","messages","metrics"): assert k in d
    assert d["ok"] is True
```

## C. Healthcheck that won’t flap

```yaml
# compose service
healthcheck:
  test: ["CMD","sh","-lc","curl -sf http://127.0.0.1:${API_CONTAINER_PORT:-8788}/ready >/dev/null"]
  interval: 5s
  timeout: 3s
  retries: 10
  start_period: 5s
```

## D. One-button verify (zsh-safe)

```zsh
# scripts/verify_min_agent.zsh
#!/usr/bin/env zsh
set -euo pipefail; setopt NO_NOMATCH PIPE_FAIL
H="${API_HOST:-127.0.0.1}"; P="${API_PORT:-8788}"
curl -sSf "http://$H:$P/ready" >/dev/null
REQ='{"messages":[{"role":"user","content":"hi"}],"model":"ollama/qwen3:8b",
"api_base":"'"${OLLAMA_URL:-http://ollama:11434}"'","base_url":"'"${OLLAMA_URL:-http://ollama:11434}"'",
"tool_backend":"local","use_tools":false}'
curl -sS -H 'content-type: application/json' --data "$REQ" "http://$H:$P/agent/run" | grep -q '"ok":true' || { echo "agent fail"; exit 1; }
echo "OK"
```

## E. STATUS snapshot (stops amnesia)

```markdown
# STATUS_NOW.md
- Endpoint: http://127.0.0.1:8788/agent/run
- Ready: http://127.0.0.1:8788/ready
- Provider: http://ollama:11434 (model ollama/qwen3:8b)
- Verify: scripts/verify_min_agent.zsh
- Smoke: tests/smoke/test_agent_api_shape.py::test_agent_api_shape
```

---

# How I’ll behave going forward (my own guardrails)

* Start every session with the **Kickoff Checklist** block. If anything is missing, I state assumptions *and stop* until we align.
* Touch **only** the files you list. If I need another file, I’ll ask; otherwise I won’t.
* Prefer **short, zsh-safe commands**; no inline comments in copy/paste blocks.
* When Docker appears in scope, I will **add or use `/ready`** first and wire healthchecks to it—never to real providers.
* If the first smoke isn’t green in a short window, I snapshot `STATUS_NOW.md` and pause.

If you want, I’ll package these into a tiny PR (new `/ready`, the single smoke, `verify_min_agent.zsh`, `STATUS_NOW.md`) so you have the guardrails in-repo. Otherwise, this is the agreement I’ll follow to keep us out of AI-slop territory.
