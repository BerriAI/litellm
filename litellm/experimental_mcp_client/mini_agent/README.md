Mini-Agent (Experimental)

Summary
- Tiny in-code loop: iterate → tool → observe → repair.
- Guardrails: max iterations/time, tool caps, allowlisted shell, timeouts.
- Tools: Local (exec_python, exec_shell), optional HTTP tools via headers.
- Router-first: uses litellm Router; no default changes; off-by-default fast paths.

Quick Start

Python (in-code)

from litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent import AgentConfig, LocalMCPInvoker, run_mcp_mini_agent

messages = [{"role": "user", "content": "echo hi and finish"}]
cfg = AgentConfig(model="openai/gpt-4o-mini", max_iterations=4)
result = run_mcp_mini_agent(messages, mcp=LocalMCPInvoker(shell_allow_prefixes=("echo",)), cfg=cfg)
print(result.stopped_reason, result.final_answer)

FastAPI endpoint (optional)

uvicorn litellm.experimental_mcp_client.mini_agent.agent_proxy:app --port 8788

POST /agent/run
{
  "messages": [{"role":"user","content":"hi"}],
  "model":"openai/gpt-4o-mini",
  "tool_backend":"http",
  "tool_http_base_url":"http://127.0.0.1:8788",
  "tool_http_headers":{"Authorization":"Bearer <token>"}
}

Notes
- research_tools.py is optional (httpx-gated) and off by default.
- Observation messages are appended as assistant content and kept last.
- Extracted router seam is opt-in: set LITELLM_ROUTER_CORE=extracted. Default is legacy.
