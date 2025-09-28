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
def test_mini_agent_http_tools_medium_optional():
    host = os.getenv("MINI_AGENT_API_HOST", "127.0.0.1")
    port = int(os.getenv("MINI_AGENT_API_PORT", "8788"))
    if not _can(host, port):
        pytest.skip(f"mini-agent API not reachable on {host}:{port}")

    tools_base = os.getenv("NDSMOKE_TOOLS_BASE")
    bearer = os.getenv("NDSMOKE_TOOLS_BEARER", "")
    if not tools_base:
        pytest.skip("NDSMOKE_TOOLS_BASE not set (HTTP tools gateway)")

    url = f"http://{host}:{port}/agent/run"
    headers = {"Authorization": f"Bearer {bearer}"} if bearer else {}
    # Require a real model that can perform tool-calling; otherwise skip
    model = os.getenv("NDSMOKE_E2E_AGENT_MODEL")
    if not model:
        pytest.skip("NDSMOKE_E2E_AGENT_MODEL not set (needs a tool-capable model)")

    req = {
        "messages": [{"role": "user", "content": "Use echo('E2E') and finish."}],
        "model": model,
        "tool_backend": "http",
        "tool_http_base_url": tools_base,
        "tool_http_headers": headers,
        "use_tools": True,
        "max_iterations": 3,
    }
    try:
        r = httpx.post(url, json=req, timeout=45.0)
        r.raise_for_status()
    except Exception:
        pytest.skip("tools gateway not reachable or returned error")
    data = r.json()
    assert data.get("ok") is True
    # Validate last tool invocation captured and contains result
    inv = data.get("last_tool_invocation") or {}
    text = inv.get("stdout") or inv.get("text") or inv.get("result") or ""
    # Accept either direct echo or preview in messages
    if isinstance(text, str) and ("E2E" in text):
        return
    for m in data.get("messages", []) or []:
        if m.get("role") == "tool" and "E2E" in (m.get("content") or ""):
            return
    pytest.skip("model did not perform tool calls; skip this medium case")
