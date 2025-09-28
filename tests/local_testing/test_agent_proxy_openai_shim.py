"""
Purpose
- Define acceptance (future) for an OpenAI-compatible /v1/chat/completions shim on the agent API.

Scope
- DOES: mark desired surface and skip (acts as living spec).
- DOES NOT: exercise live providers or tools.

Run
- Always collected; skipped by design until shim is implemented.
"""
import pytest


from fastapi.testclient import TestClient

def test_agent_proxy_openai_shim(monkeypatch):
    """Verify the OpenAI-compatible /v1/chat/completions shim on the agent API."""
    from litellm.experimental_mcp_client.mini_agent.agent_proxy import app
    client = TestClient(app)
    req = {"model": "dummy", "messages": [{"role": "user", "content": "hi"}]}
    r = client.post("/v1/chat/completions", json=req)
    assert r.status_code == 200
    data = r.json()
    assert data.get("object") == "chat.completion"
    assert isinstance((data.get("choices") or []), list)
