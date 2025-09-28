"""
Test for last_tool_invocation presence in /agent/run response (optional, gated).

Prereq: a tool-capable model that decides to call a tool at least once.
We provide a tools stub in compose (tools-stub) and set tool_backend=http so the agent
consults the stub for /tools and /invoke. The model must decide to call 'echo'.
"""
import os, socket, pytest, httpx

TIMEOUT = int(os.getenv("NDSMOKE_TIMEOUT", "240"))


def _can(host: str, port: int, t: float = 0.3) -> bool:
    try:
        with socket.create_connection((host, port), timeout=t):
            return True
    except OSError:
        return False


@pytest.mark.ndsmoke
def test_agent_last_tool_invocation_optional():
    if os.getenv("DOCKER_MINI_AGENT", "0") != "1":
        pytest.skip("DOCKER_MINI_AGENT not set; skipping")
    if os.getenv("E2E_DOCKER_MODEL_READY", "0") != "1":
        pytest.skip("Tool-capable model not guaranteed; set E2E_DOCKER_MODEL_READY=1 to enable")

    host = os.getenv("MINI_AGENT_API_HOST", "127.0.0.1")
    port = int(os.getenv("MINI_AGENT_API_PORT", "8788"))
    if not _can(host, port):
        pytest.skip(f"mini-agent API not reachable on {host}:{port}")

    # Assume tools-stub is running in docker compose network under hostname tools-stub:8791
    tools_base = os.getenv("TOOLS_STUB_BASE", "http://tools-stub:8791")

    messages = [
        {"role": "system", "content": "You can call tools from a tool registry."},
        {"role": "user", "content": "Use the echo tool to return the word OK."},
    ]
    payload = {
        "messages": messages,
        "model": os.getenv("LITELLM_DEFAULT_CODE_MODEL", "codex-agent/mini"),
        "tool_backend": "http",
        "tool_http_base_url": tools_base,
        "use_tools": True,
        "max_iterations": 2,
    }

    url = f"http://{host}:{port}/agent/run"
    r = httpx.post(url, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    assert data.get("ok") is True
    assert "last_tool_invocation" in data or any(m.get("role") == "tool" for m in data.get("messages", []) or [])

