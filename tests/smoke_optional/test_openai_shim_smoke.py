import pytest


fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")


@pytest.mark.smoke
def test_openai_shim_basic(monkeypatch):
    from fastapi.testclient import TestClient
    from litellm.experimental_mcp_client.mini_agent.agent_proxy import app
    import litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent as agent

    class StubResult:
        def __init__(self):
            self.final_answer = "shim ok"
            self.stopped_reason = "success"
            self.messages = [{"role":"assistant","content":"shim ok"}]
            self.iterations = [agent.IterationRecord(tool_invocations=[])]

    async def fake_run(messages, mcp, cfg):  # type: ignore[override]
        return StubResult()

    monkeypatch.setattr(agent, "arun_mcp_mini_agent", fake_run, raising=True)
    client = TestClient(app)
    body = {"model":"dummy","messages":[{"role":"user","content":"hi"}]}
    r = client.post("/v1/chat/completions", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data.get("choices")[0]["message"]["content"] == "shim ok"
