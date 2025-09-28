"""
Purpose
- Lock the public contract for /agent/run (paved road, hermetic).

Scope
- DOES: assert presence/shape of {ok, final_answer, stopped_reason, messages, metrics}.
- DOES NOT: call external models, tools, Docker, or gateways.

Run
- `pytest tests/smoke -k test_agent_proxy_response_shapes_minimal -q`
"""
import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed for this smoke")
from fastapi.testclient import TestClient  # type: ignore


def test_agent_proxy_response_shapes_minimal(monkeypatch):
    # Lazy import after import guards to avoid optional deps issues
    from litellm.experimental_mcp_client.mini_agent.agent_proxy import app

    client = TestClient(app)
    req = {
        "messages": [{"role": "user", "content": "say hi"}],
        "model": "dummy",
        "tool_backend": "echo",
    }
    r = client.post("/agent/run", json=req)
    assert r.status_code == 200
    data = r.json()
    # Shape assertions
    assert isinstance(data, dict)
    for k in ("ok", "final_answer", "stopped_reason", "messages", "metrics"):
        assert k in data
    assert isinstance(data["ok"], bool)
    assert isinstance(data["messages"], list)
    assert isinstance(data["metrics"], dict)
