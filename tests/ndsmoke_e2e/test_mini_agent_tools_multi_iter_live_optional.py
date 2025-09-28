"""
Multi-iteration tool use under Docker (optional, gated).

Requires:
  - DOCKER_MINI_AGENT=1
  - E2E_DOCKER_MODEL_READY=1 (ensure the agent's model can produce a python code block)

Asserts:
  - ok True
  - Presence of either an "Observation from last tool run" message (repair path) OR non-empty final answer.
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
def test_mini_agent_docker_multi_iter_with_repair_optional():
    if os.getenv("DOCKER_MINI_AGENT", "0") != "1":
        pytest.skip("DOCKER_MINI_AGENT not set; skipping")
    if os.getenv("E2E_DOCKER_MODEL_READY", "0") != "1":
        pytest.skip("Model not guaranteed to produce code block; set E2E_DOCKER_MODEL_READY=1 to enable")

    host = os.getenv("MINI_AGENT_API_HOST", "127.0.0.1")
    port = int(os.getenv("MINI_AGENT_API_PORT", "8788"))
    if not _can(host, port):
        pytest.skip(f"mini-agent API not reachable on {host}:{port}")

    # Prompt the model to emit a Python code block; agent should auto-run it and optionally repair.
    messages = [
        {"role": "system", "content": "You may execute Python using available tools."},
        {"role": "user", "content": (
            "Write a Python function `inc(x)` that returns x+1. Return only a Python fenced code block."
            " Then correct it if it fails and print inc(2)."
        )},
    ]
    payload = {
        "messages": messages,
        "model": os.getenv("LITELLM_DEFAULT_CODE_MODEL", "codex-agent/mini"),
        "tool_backend": "local",
        "use_tools": True,
        "auto_run_code_on_code_block": True,
        "max_iterations": 3,
        "enable_repair": True,
    }

    url = f"http://{host}:{port}/agent/run"
    r = httpx.post(url, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    assert data.get("ok") is True
    msgs = data.get("messages", []) or []
    fa = (data.get("final_answer") or "").strip()
    has_obs = any(
        isinstance(m, dict) and isinstance(m.get("content"), str) and "Observation from last tool run" in m.get("content")
        for m in msgs
    )
    assert has_obs or len(fa) > 0

