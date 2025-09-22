# Quick Start (Fork)

This fork adds an opt‑in, experimental mini‑agent and a guarded Router streaming seam. Defaults are unchanged. Use this page to get productive fast.

- Prereqs
  - Python 3.10+
  - Keys for your target model (e.g., OPENAI_API_KEY or GEMINI_API_KEY)

- Install (editable)
  - pip install -e .

- Mini‑Agent (in‑code, guardrailed)
  - In code:
    - from litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent import AgentConfig, LocalMCPInvoker, run_mcp_mini_agent
    - messages = [{"role": "user", "content": "echo hi and finish"}]
    - cfg = AgentConfig(model="openai/gpt-4o-mini", max_iterations=4)
    - result = run_mcp_mini_agent(messages, mcp=LocalMCPInvoker(shell_allow_prefixes=("echo",)), cfg=cfg)
    - print(result.stopped_reason, result.final_answer)

- Optional HTTP Tools Gateway
  - Start a tools endpoint you control (see docs/experimental/mini-agent.md).
  - Pass headers via agent_proxy: tool_http_headers={"Authorization":"Bearer <token>"}.

- Deterministic Smokes
  - PYTHONPATH=$(pwd) pytest -q tests/smoke -k 'mini_agent or http_tools_invoker or router_streaming_'

- Router Streaming Seam (opt‑in; default=legacy)
  - Do not change defaults in production until canary parity is proven.
  - Enable extracted only in a canary service: export LITELLM_ROUTER_CORE=extracted.

- Canary Parity (scripted)
  - OPENAI_API_KEY=… LITELLM_DEFAULT_MODEL=openai/gpt-4o-mini     PARITY_ROUNDS=5 PARITY_THRESH_PCT=3.0     PARITY_OUT=/var/log/litellm_parity.jsonl     uv run local/scripts/router_core_parity.py
  - Summarize over time: uv run local/scripts/parity_summarize.py --in /var/log/litellm_parity.jsonl --thresh 3.0
  - Acceptance: same_text True and worst ttft/total ≤ 3% for 1 week.

- Where to read more
  - STATE_OF_PROJECT.md (status, guardrails, plan)
  - docs/my-website/docs/experimental/mini-agent.md (usage + troubleshooting)
  - local/docs/02_operational/CANARY_PARITY_PLAN.md (operations)


- Mini‑Agent over HTTP Tools (in‑code)
  - Start your tools gateway (see docs) at http://127.0.0.1:8789
  - Then call the mini‑agent using the HTTP tools adapter:

```python
from litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent import AgentConfig, run_mcp_mini_agent
from litellm.experimental_mcp_client.mini_agent.http_tools_invoker import HttpToolsInvoker

mcp = HttpToolsInvoker("http://127.0.0.1:8789", headers={"Authorization": "Bearer <token>"})
messages = [{"role": "user", "content": "call echo('hi') and finish"}]
cfg = AgentConfig(model="openai/gpt-4o-mini", max_iterations=4)
res = run_mcp_mini_agent(messages, mcp=mcp, cfg=cfg)
print(res.stopped_reason, res.final_answer)
```

- Agent Proxy (HTTP entrypoint)
  - Run the endpoint:

```bash
uvicorn litellm.experimental_mcp_client.mini_agent.agent_proxy:app --port 8788
```

  - Call it from Python:

```python
import httpx
payload = {
  "messages": [{"role": "user", "content": "hi"}],
  "model": "openai/gpt-4o-mini",
  "tool_backend": "http",
  "tool_http_base_url": "http://127.0.0.1:8789",
  "tool_http_headers": {"Authorization": "Bearer <token>"}
}
resp = httpx.post("http://127.0.0.1:8788/agent/run", json=payload, timeout=30.0)
resp.raise_for_status()
print(resp.json())
```

- Extras (optional helpers)
  - Images (local/remote to data URL):

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
