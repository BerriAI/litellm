"""
Bounded parallel tool execution (optional, gated).

Note: This requires a tool-capable model that emits multiple tool calls in a single turn,
which is provider-specific. We guard behind E2E_DOCKER_MODEL_READY and skip otherwise.
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
def test_mini_agent_parallel_tools_order_optional():
    if os.getenv("DOCKER_MINI_AGENT", "0") != "1":
        pytest.skip("DOCKER_MINI_AGENT not set; skipping")
    if os.getenv("E2E_DOCKER_MODEL_READY", "0") != "1":
        pytest.skip("Tool-capable model not guaranteed; set E2E_DOCKER_MODEL_READY=1 to enable")

    host = os.getenv("MINI_AGENT_API_HOST", "127.0.0.1")
    port = int(os.getenv("MINI_AGENT_API_PORT", "8788"))
    if not _can(host, port):
        pytest.skip(f"mini-agent API not reachable on {host}:{port}")

    # Ask the model to issue two tool calls (exec_python) in one message.
    messages = [
        {"role": "system", "content": "You may call available tools to execute code."},
        {"role": "user", "content": (
            "Call the code execution tool twice in order: first print('A'), then print('B')."
            " Ensure both calls are separate."
        )},
    ]
    payload = {
        "messages": messages,
        "model": os.getenv("LITELLM_DEFAULT_CODE_MODEL", "codex-agent/mini"),
        "tool_backend": "local",
        "use_tools": True,
        "max_iterations": 2,
        "enable_repair": False,
        "max_total_seconds": 60,
    }

    url = f"http://{host}:{port}/agent/run"
    r = httpx.post(url, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    assert data.get("ok") is True
    # We cannot assert strict order without provider guarantees, but we expect at least one
    # assistant/tool pair. Presence of "last_tool_invocation" hints tool execution happened.
    inv = data.get("last_tool_invocation")
    msgs = data.get("messages", []) or []
    has_tool_reply = any(m.get("role") == "tool" for m in msgs if isinstance(m, dict))
    assert inv or has_tool_reply

