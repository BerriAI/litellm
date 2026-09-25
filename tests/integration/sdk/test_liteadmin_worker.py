import json
import sys
from pathlib import Path
from typing import Final

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient

from litellm.proxy.liteadmin.endpoints import StartRequest, _run


@pytest.fixture
def worker_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runtime: Final = tmp_path / "liteadmin"
    runtime.mkdir()
    (runtime / "__init__.py").touch()
    (runtime / "worker.py").write_text(
        "import json, os, sys\n"
        "request = json.loads(sys.stdin.readline())\n"
        "print(json.dumps({'type':'message', 'text':request['chat']['messages'][0]['content']}), flush=True)\n"
        "print(json.dumps({'type':'message', 'text':os.getenv('UNRELATED_SECRET', 'absent')}), flush=True)\n"
        "decision = json.loads(sys.stdin.readline())\n"
        "print(json.dumps({'type':'message', 'text':str(decision['approved'])}), flush=True)\n"
        "print(json.dumps({'type':'done'}), flush=True)\n"
    )
    monkeypatch.setenv("LITEADMIN_RUNTIME", str(runtime))
    monkeypatch.setenv("LITEADMIN_PYTHON", sys.executable)
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-reach-the-worker")
    monkeypatch.delenv("LITELLM_UI_API_DOC_BASE_URL", raising=False)
    monkeypatch.delenv("PROXY_BASE_URL", raising=False)
    app: Final = FastAPI()

    @app.websocket("/chat")
    async def chat(websocket: WebSocket) -> None:
        await websocket.accept()
        start: Final = StartRequest.model_validate_json(await websocket.receive_text())
        await _run(websocket, start, {"Authorization": "Bearer test"})
        await websocket.close()

    with TestClient(app, backend_options={"use_uvloop": True}) as client:
        yield client


@pytest.mark.parametrize("approved", (True, False))
def test_uvloop_worker_receives_chat_and_decisions_without_inheriting_secrets(
    worker_client: TestClient, approved: bool
) -> None:
    with worker_client.websocket_connect("/chat") as websocket:
        websocket.send_json(
            {
                "access_token": "test",
                "chat": {
                    "model": "test",
                    "messages": [{"role": "user", "content": "Check the team budget"}],
                    "inference_base_url": "http://testserver",
                },
            }
        )
        assert websocket.receive_json() == {"type": "message", "text": "Check the team budget"}
        assert websocket.receive_json() == {"type": "message", "text": "absent"}
        websocket.send_text(json.dumps({"id": "action", "approved": approved}))
        assert websocket.receive_json() == {"type": "message", "text": str(approved)}
        assert websocket.receive_json() == {"type": "done"}
