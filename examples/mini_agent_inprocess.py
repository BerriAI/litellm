"""
Example: Mini-Agent in-process loop with local tools.

Run:
  python examples/mini_agent_inprocess.py
"""
from __future__ import annotations

from litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent import (
    AgentConfig,
    LocalMCPInvoker,
    run_mcp_mini_agent,
)


def main() -> None:
    msgs = [{"role": "user", "content": "echo hi and finish"}]
    cfg = AgentConfig(model="openai/gpt-4o-mini", max_iterations=3)
    res = run_mcp_mini_agent(msgs, mcp=LocalMCPInvoker(shell_allow_prefixes=("echo",)), cfg=cfg)
    print(res.stopped_reason, (res.final_answer or "").strip())


if __name__ == "__main__":
    main()

