"""
Example: Call the Mini-Agent HTTP API (/agent/run).

Prereqs:
  uvicorn litellm.experimental_mcp_client.mini_agent.agent_proxy:app --host 127.0.0.1 --port 8788

Run:
  python examples/agent_proxy_client.py
"""
from __future__ import annotations

import httpx


def main() -> None:
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "model": "openai/gpt-4o-mini",
        "tool_backend": "local",
        "use_tools": False,
    }
    r = httpx.post("http://127.0.0.1:8788/agent/run", json=payload, timeout=30.0)
    r.raise_for_status()
    print(r.json())


if __name__ == "__main__":
    main()

