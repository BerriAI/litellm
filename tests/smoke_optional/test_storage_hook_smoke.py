import os
import json
import tempfile
import pytest


fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")


@pytest.mark.smoke
def test_storage_hook_jsonl(monkeypatch):
    from fastapi.testclient import TestClient
    from litellm.experimental_mcp_client.mini_agent.agent_proxy import app
    import litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent as agent

    class StubResult:
        def __init__(self):
            self.final_answer = "ok"
            self.stopped_reason = "success"
            self.messages = [{"role":"assistant","content":"ok"},{"role":"tool","content":"stdout tail\n"}]
            self.iterations = [agent.IterationRecord(tool_invocations=[]) for _ in range(1)]

    async def fake_run(messages, mcp, cfg):  # type: ignore[override]
        return StubResult()

    monkeypatch.setattr(agent, "arun_mcp_mini_agent", fake_run, raising=True)

    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "runs.jsonl")
        os.environ["MINI_AGENT_STORE_TRACES"] = "1"
        os.environ["MINI_AGENT_STORE_PATH"] = out
        client = TestClient(app)
        r = client.post("/agent/run", json={"messages":[{"role":"user","content":"hi"}],"model":"dummy"})
        assert r.status_code == 200
        # verify a jsonl line was written
        with open(out, "r", encoding="utf-8") as f:
            line = f.readline()
            rec = json.loads(line)
            assert rec.get("metrics", {}).get("iterations") == 1
            assert "final_answer_preview" in rec
        os.environ.pop("MINI_AGENT_STORE_TRACES", None)
        os.environ.pop("MINI_AGENT_STORE_PATH", None)

