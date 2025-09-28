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
def test_mini_agent_escalation_high_optional(monkeypatch):
    host = os.getenv("MINI_AGENT_API_HOST", "127.0.0.1")
    port = int(os.getenv("MINI_AGENT_API_PORT", "8788"))
    if not _can(host, port):
        pytest.skip(f"mini-agent API not reachable on {host}:{port}")

    # Enable short-circuit for chutes escalation in ndsmoke
    monkeypatch.setenv("NDSMOKE_SHORTCIRCUIT_CHUTES", "1")

    url = f"http://{host}:{port}/agent/run"
    req = {
        "messages": [{"role": "user", "content": "Budget tight; escalate on last step."}],
        "model": "dummy",
        "tool_backend": "local",
        "use_tools": False,
        "max_iterations": 2,
        "escalate_on_budget_exceeded": True,
        "escalate_model": "chutes/deepseek-ai/DeepSeek-R1",
        "max_total_seconds": 0.01,
    }
    r = httpx.post(url, json=req, timeout=30.0)
    r.raise_for_status()
    data = r.json()
    assert data.get("ok") is True
    m = data.get("metrics", {})
    assert m.get("escalated") is True
    assert str(m.get("used_model", "")).startswith("chutes/")

