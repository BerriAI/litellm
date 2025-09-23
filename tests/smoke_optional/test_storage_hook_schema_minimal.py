import os
import json
import tempfile
import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")

@pytest.mark.smoke
def test_storage_hook_schema_minimal(monkeypatch):
    """
    Golden minimal trace schema for dashboard/replay.
    """
    from fastapi.testclient import TestClient
    from litellm.experimental_mcp_client.mini_agent.agent_proxy import app
    import litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent as agent

    class StubResult:
        def __init__(self):
            self.final_answer = "ok"
            self.stopped_reason = "success"
            self.messages = [
                {"role": "assistant", "content": "ok"},
                {"role": "tool", "content": "stdout tail\n"},
            ]
            self.iterations = [agent.IterationRecord(tool_invocations=[])]

    async def fake_run(messages, mcp, cfg):  # type: ignore[override]
        return StubResult()

    monkeypatch.setattr(agent, "arun_mcp_mini_agent", fake_run, raising=True)

    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "runs.jsonl")
        os.environ["MINI_AGENT_STORE_TRACES"] = "1"
        os.environ["MINI_AGENT_STORE_PATH"] = out
        from fastapi.testclient import TestClient
        client = TestClient(app)
        r = client.post("/agent/run", json={"messages":[{"role":"user","content":"hi"}], "model":"dummy"})
        assert r.status_code == 200
        line = open(out, "r", encoding="utf-8").readline()
        rec = json.loads(line)
        for k in ("ok", "metrics", "final_answer_preview", "iterations", "messages"):
            assert k in rec
        assert isinstance(rec["metrics"].get("iterations"), int)
        assert isinstance(rec.get("final_answer_preview", ""), str)
        os.environ.pop("MINI_AGENT_STORE_TRACES", None)
        os.environ.pop("MINI_AGENT_STORE_PATH", None)
