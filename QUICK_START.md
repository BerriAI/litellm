# Quick Start (Fork)

This fork adds an opt‑in Mini‑Agent, an env‑gated `codex-agent` provider, a pragmatic HTTP tools adapter, and deterministic/live E2E smokes. Defaults remain upstream‑compatible. This page gives copy‑paste, working examples for the paved‑road features.

## 1) Prereqs
- Python 3.10+
- Optional: Docker + Docker Compose
- Optional: model keys (e.g., `OPENAI_API_KEY`, `GEMINI_API_KEY`)

## 2) Install
- Editable install for local iteration:
  - `pip install -e .`

## 3) Run the Mini‑Agent (Docker first; local also supported)
- Docker/compose (recommended):
  - Create network (once): `docker network create llmnet || true`
  - Start: `make docker-up`
  - Logs: `make docker-logs`
  - Health: `curl -sf http://127.0.0.1:8788/ready`
- Local uvicorn (fastest for dev):
  - `uvicorn litellm.experimental_mcp_client.mini_agent.agent_proxy:app --host 127.0.0.1 --port 8788`
  - Health: `curl -sf http://127.0.0.1:8788/ready`

Tips for Ollama in Docker:
- Bring up Ollama on the same network: `make docker-ollama-up`
- Resolve base URL on host: `scripts/resolve_ollama_base.sh` → e.g., `http://127.0.0.1:11434`

## 4) Mini‑Agent in code (local tools)
```python
from litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent import AgentConfig, LocalMCPInvoker, run_mcp_mini_agent

messages = [{"role": "user", "content": "echo hi and finish"}]
cfg = AgentConfig(model="openai/gpt-4o-mini", max_iterations=3)
res = run_mcp_mini_agent(messages, mcp=LocalMCPInvoker(shell_allow_prefixes=("echo",)), cfg=cfg)
print(res.stopped_reason, res.final_answer)
```

## 5) Mini‑Agent over HTTP tools (MCP‑style)
- Start your tools host exposing `/tools` and `/invoke` (bearer‑protected is fine).
- Call via the HTTP tools adapter:

```python
from litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent import AgentConfig, run_mcp_mini_agent
from litellm.experimental_mcp_client.mini_agent.http_tools_invoker import HttpToolsInvoker

mcp = HttpToolsInvoker("http://127.0.0.1:8788", headers={"Authorization": "Bearer <token>"})
messages = [{"role": "user", "content": "call echo('hi') and finish"}]
cfg = AgentConfig(model="openai/gpt-4o-mini", max_iterations=4)
res = run_mcp_mini_agent(messages, mcp=mcp, cfg=cfg)
print(res.stopped_reason, res.final_answer)
```

- Local stub server (no external deps):
  - Start: `uvicorn examples.http_tools_stub:app --host 127.0.0.1 --port 8791`
  - Then call with the HTTP tools adapter using base `http://127.0.0.1:8791`.

## 6) Agent Proxy (HTTP API)
- Run the endpoint:

```bash
uvicorn litellm.experimental_mcp_client.mini_agent.agent_proxy:app --port 8788
```

// Minimal call from Python or: `python examples/agent_proxy_client.py`

```python
import httpx
payload = {
  "messages": [{"role": "user", "content": "hi"}],
  "model": "openai/gpt-4o-mini",
  "tool_backend": "http",
  "tool_http_base_url": "http://127.0.0.1:8788",
  "tool_http_headers": {"Authorization": "Bearer <token>"}
}
resp = httpx.post("http://127.0.0.1:8788/agent/run", json=payload, timeout=30.0)
resp.raise_for_status()
print(resp.json())
```

## 7) Codex‑Agent (env‑gated provider)
- Enable provider and point to an OpenAI‑compatible API (e.g., the agent shim above):
  - `export LITELLM_ENABLE_CODEX_AGENT=1`
  - `export CODEX_AGENT_API_BASE=http://127.0.0.1:8788`
- Router example:
  - `python examples/codex_agent_router.py`
```python
from litellm import Router
r = Router(model_list=[{"model_name":"codex-agent-1","litellm_params":{
  "model":"codex-agent/mini",
  "api_base": "http://127.0.0.1:8788",
  "api_key": ""}}
])
out = r.completion(model="codex-agent-1", messages=[{"role":"user","content":"Say hello and finish."}])
print(getattr(out.choices[0].message, "content", ""))
```

## 8) Exec‑RPC quick check (Docker or local)
- Start (compose path above) or run locally:
  - `uvicorn litellm.experimental_mcp_client.mini_agent.exec_rpc_server:app --host 127.0.0.1 --port 8790`
- Probe and run Python:
```bash
curl -sf http://127.0.0.1:8790/health
curl -sf -X POST http://127.0.0.1:8790/exec -H 'content-type: application/json' -d '{"language":"python","code":"print(123)"}'
```

## 9) Streaming (OpenAI‑compatible)
- Requires `OPENAI_API_BASE` and `OPENAI_API_KEY` (or compatible shim):
```python
import os, asyncio, litellm
async def go():
  chunks = []
  async for ch in await litellm.acompletion(
    model=os.getenv("OPENAI_COMPAT_MODEL","gpt-3.5-turbo"),
    messages=[{"role":"user","content":"Stream the word HELLO."}],
    api_base=os.getenv("OPENAI_API_BASE"),
    api_key=os.getenv("OPENAI_API_KEY"),
    stream=True,
  ):
    try:
      delta = ((ch or {}).get("choices") or [{}])[0].get("delta", {})
      txt = delta.get("content") or ""
      if txt: chunks.append(txt)
    except Exception: pass
  print("got:", "".join(chunks))
asyncio.run(go())
```

## 10) Deterministic + E2E tests
- Deterministic (no network):
  - `PYTHONPATH=$(pwd) pytest -q tests/local_testing`
- E2E (skip‑friendly):
  - Docker mini‑agent: `make docker-up`
  - Docker ndsmokes (loopback by default): `make ndsmoke-docker`
  - Optional Ollama: `make docker-ollama-up` and set `LITELLM_DEFAULT_CODE_MODEL=ollama/<model>`

## 11) Router streaming seam (opt‑in; default=legacy)
- Only for canaries until parity proven:
  - `export LITELLM_ROUTER_CORE=extracted`
- Parity harness:
  - `OPENAI_API_KEY=… LITELLM_DEFAULT_MODEL=openai/gpt-4o-mini PARITY_ROUNDS=5 PARITY_THRESH_PCT=3.0 PARITY_OUT=/tmp/parity.jsonl python local/scripts/router_core_parity.py`
  - `python local/scripts/parity_summarize.py --in /tmp/parity.jsonl --thresh 3.0`
  - Acceptance: same_text True and worst ttft/total ≤ 3% for 1 week.

## 12) Extras (optional helpers)
- Images (local/remote → data URL):

```python
from litellm.extras.images import compress_image
url = compress_image("/path/to/image.png", max_kb=64, cache_dir=".cache")
print(url[:64], "...")
```
- Cache (Redis one‑liner for Router):

```python
from litellm.extras.cache import configure_cache_redis
from litellm.router import Router

r = Router(model_list=[{"model_name":"m","litellm_params":{"model":"openai/gpt-4o-mini","api_key":"sk-..."}}])
configure_cache_redis(r, host="127.0.0.1", port=6379, ttl_seconds=300)
```
- Response utils (extract text from Router response):

```python
from litellm.extras.response_utils import extract_content
from litellm.router import Router

r = Router(model_list=[{"model_name":"m","litellm_params":{"model":"openai/gpt-4o-mini","api_key":"sk-..."}}])
resp = await r.acompletion(model="m", messages=[{"role":"user","content":"hi"}])
print(extract_content(resp))
```
- Batch helper (concurrent acompletions):

```python
import asyncio
from litellm.extras.batch import acompletion_as_completed
from litellm.router import Router

r = Router(model_list=[{"model_name":"m","litellm_params":{"model":"openai/gpt-4o-mini","api_key":"sk-..."}}])
reqs = [
  {"model":"m","messages":[{"role":"user","content":f"hi {i}"}]}
  for i in range(5)
]
async def go():
  async for idx, resp in acompletion_as_completed(r, reqs, concurrency=2):
    print(idx, getattr(getattr(resp.choices[0],"message",{}),"content",None))
asyncio.run(go())
```

## 15) Examples (runnable)
- `python examples/codex_agent_router.py`
- `python examples/router_parallel_multimodal.py`
- `python examples/mini_agent_inprocess.py`
- `python examples/agent_proxy_client.py`
- `uvicorn examples.http_tools_stub:app --host 127.0.0.1 --port 8791`

## 16) MVP Readiness (one-command summary)
- Local deterministic + low E2E (shim) + optional Docker readiness:
  - `python scripts/mvp_check.py`
- Docker‑first quick check:
  - `make docker-up && make ndsmoke-docker`
  - Optional heavier live run (requires model deps): `make ndsmoke-docker-live`

## 13) Troubleshooting
- Mini‑Agent returns empty content with model "dummy": set `MINI_AGENT_ALLOW_DUMMY=1` to force the stub, or provide a real model/key.
- Async tests skipped: ensure you run inside your Poetry env so `pytest-asyncio` is loaded (e.g., `poetry run pytest …`).
- HTTP tools medium E2E skips: start a tools host and set `MINI_AGENT_TOOL_HTTP_HEADERS` and `NDSMOKE_E2E_AGENT_MODEL`.

## 14) More reading
- `STATE_OF_PROJECT.md` — status, capabilities, risks.
- `docs/my-website/docs/experimental/mini-agent.md` — deeper agent guide.
- `docs/my-website/docs/providers/codex_agent.md` — codex‑agent provider usage.
