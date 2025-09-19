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
