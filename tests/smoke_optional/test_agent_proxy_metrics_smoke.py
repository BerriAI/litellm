import pytest


fastapi = pytest.importorskip("fastapi", reason="fastapi not installed for this smoke")


@pytest.mark.smoke
def test_agent_proxy_returns_metrics(monkeypatch):
    from fastapi.testclient import TestClient
    from litellm.experimental_mcp_client.mini_agent.agent_proxy import app
    import litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent as agent

    class StubResult:
        def __init__(self):
            self.final_answer = "ok"
            self.stopped_reason = "success"
            self.messages = [{"role":"assistant","content":"ok"}]
            self.iterations = [agent.IterationRecord(tool_invocations=[]) for _ in range(2)]

    async def fake_run(messages, mcp, cfg):  # type: ignore[override]
        return StubResult()

    monkeypatch.setattr(agent, "arun_mcp_mini_agent", fake_run, raising=True)

    client = TestClient(app)
    payload = {
        "messages": [{"role":"user","content":"hi"}],
        "model": "dummy",
        "tool_backend": "local",
        "use_tools": False,
        "max_iterations": 2,
        "enable_repair": False,
    }
    r = client.post("/agent/run", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data.get("ok") is True
    metrics = data.get("metrics") or {}
    assert metrics.get("iterations") == 2
    assert isinstance(metrics.get("ttotal_ms"), (int, float))
