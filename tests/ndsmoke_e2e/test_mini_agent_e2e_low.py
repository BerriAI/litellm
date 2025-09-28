import os, socket
import httpx
import pytest


def _can(host: str, port: int, t: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=t):
            return True
    except OSError:
        return False


@pytest.mark.ndsmoke
@pytest.mark.e2e
def test_mini_agent_finalize_via_api_low():
    host = os.getenv("MINI_AGENT_API_HOST", "127.0.0.1")
    port = int(os.getenv("MINI_AGENT_API_PORT", "8788"))
    if not _can(host, port):
        pytest.skip(f"mini-agent API not reachable on {host}:{port}")

    url = f"http://{host}:{port}/agent/run"
    req = {
        "messages": [{"role": "user", "content": "Say hello and finish."}],
        "model": "dummy",
        "tool_backend": "local",
        "use_tools": False,
    }
    r = httpx.post(url, json=req, timeout=30.0)
    r.raise_for_status()
    data = r.json()
    assert data.get("ok") is True
    fa = (data.get("final_answer") or "").strip()
    # Accept either direct final_answer or assistant content present
    if fa:
        return
    for m in data.get("messages", []):
        if m.get("role") == "assistant" and (m.get("content") or "").strip():
            return
    pytest.fail("expected non-empty final answer or assistant content")

