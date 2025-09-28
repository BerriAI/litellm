---
sidebar_position: 1
---

# Mini-Agent (Experimental)

A tiny in-code helper with strict guardrails. Defaults remain unchanged; the extracted streaming seam is opt-in via env.



A tiny in-code helper to run iterative tool flows with strict guardrails.

- Router-first, no global changes. Off-by-default fast paths.
- Local tools (`exec_python`, `exec_shell`) with timeouts + allowlist.
- Optional HTTP tool gateway (`/tools`, `/invoke`) with headers support.

## In-Code

```py
from litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent import AgentConfig, LocalMCPInvoker, run_mcp_mini_agent

messages = [{"role": "user", "content": "echo hi and finish"}]
cfg = AgentConfig(model="openai/gpt-4o-mini", max_iterations=4)
result = run_mcp_mini_agent(messages, mcp=LocalMCPInvoker(shell_allow_prefixes=("echo",)), cfg=cfg)
print(result.stopped_reason, result.final_answer)
```

## Endpoint (Optional)

Start:

```bash
uvicorn litellm.experimental_mcp_client.mini_agent.agent_proxy:app --port 8788
```

Invoke:

```bash
curl -s localhost:8788/agent/run -H 'content-type: application/json' -d '{
  "messages":[{"role":"user","content":"hi"}],
  "model":"openai/gpt-4o-mini",
  "tool_backend":"http",
  "tool_http_base_url":"http://127.0.0.1:8788",
  "tool_http_headers":{"Authorization":"Bearer <token>"}
}'
```

## Flags

- Router seam: `LITELLM_ROUTER_CORE=extracted` to use the extracted streaming iterator. Default: `legacy`.
- The research tools module is optional and `httpx`-gated.



## Troubleshooting
- Optional deps: install `httpx` for HTTP tools and research; install `Pillow` for image helpers used by local docs smokes.
- Extracted streaming seam is opt-in: set `LITELLM_ROUTER_CORE=extracted`. Default is `legacy`.
- If a test complains about missing event loop, ensure your test runner creates an event loop (see tests/smoke/conftest.py for a minimal fixture).

> Note: smokes are marked with `@pytest.mark.smoke` and registered in `pytest.ini`.
