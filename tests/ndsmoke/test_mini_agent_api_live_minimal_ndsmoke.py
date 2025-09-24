"""
Purpose
- Exercise Agent API in a "live-ish" setting with the same shape guarantees.

Scope
- DOES: env-driven base, reachability, minimal payload
- DOES NOT: validate tool calls, external providers, or content

Run
- MINI_AGENT_API_HOST=127.0.0.1 MINI_AGENT_API_PORT=8788 \
  pytest -q tests/ndsmoke/test_mini_agent_api_live_minimal_ndsmoke.py::test_agent_api_live_minimal_optional
"""
import os, pytest
from tests.ndsmoke._util import can_connect
import httpx

@pytest.mark.timeout(20)
def test_agent_api_live_minimal_optional():
    host = os.getenv("MINI_AGENT_API_HOST", "127.0.0.1")
    port = int(os.getenv("MINI_AGENT_API_PORT", "8788"))
    base = f"http://{host}:{port}"
    if not can_connect(host, port):
        pytest.skip(f"Agent API not reachable on {host}:{port}")
    r = httpx.post(
        base + "/agent/run",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "model": "dummy",
            "tool_backend": "local",
            "use_tools": False,
        },
        timeout=15.0,
    )
    r.raise_for_status()
    d = r.json()
    # same invariants as the core smoke
    for k in ("ok", "final_answer", "messages", "metrics"):
        assert k in d