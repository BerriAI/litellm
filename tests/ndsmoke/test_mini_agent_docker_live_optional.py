import os, asyncio, json, socket, pytest, httpx

TIMEOUT = int(os.getenv("NDSMOKE_TIMEOUT", "240"))


def _can_connect(host: str, port: int, timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.mark.ndsmoke
def test_mini_agent_docker_ollama_code_live_optional():
    if os.getenv('DOCKER_MINI_AGENT','0') != '1':
        pytest.skip('DOCKER_MINI_AGENT not set; skipping live docker test')
    host = os.getenv('MINI_AGENT_API_HOST','127.0.0.1')
    port = int(os.getenv('MINI_AGENT_API_PORT','8788'))
    if not _can_connect(host, port):
        pytest.skip(f'mini-agent API not reachable on {host}:{port}')

    model = os.getenv('LITELLM_DEFAULT_CODE_MODEL','ollama/glm4:latest')
    messages = [
        {"role":"system","content":"You may execute Python using available tools. If you produce code, run it and send stdout/stderr back. Iterate until tests pass."},
        {"role":"user","content":(
            "Implement a Python function `compress_runs(s: str) -> str` that compresses runs of repeated characters in a string"
            " using format like 'aaabbc' -> 'a3b2c1'. Handle empty strings and Unicode. Return only the code in a Python code block."
            " Then run a quick test that prints compress_runs('xxxyzzzz') and compress_runs('') to stdout."
        )},
    ]
    payload = {
        "messages": messages,
        "model": model,
        "tool_backend": "local",
        "use_tools": True,
        "auto_run_code_on_code_block": True,
        "max_iterations": 3,
        "enable_repair": True,
    }
    url=f"http://{host}:{port}/agent/run"
    r = httpx.post(url, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    assert data.get('ok') is True
    # Accept either a non-empty final answer or an observation from tool run
    fa = (data.get('final_answer') or '').strip()
    msgs = data.get('messages', []) or []
    has_obs = any(
        isinstance(m, dict) and isinstance(m.get('content'), str) and 'Observation from last tool run' in m.get('content')
        for m in msgs
    )
    assert (len(fa) > 0) or has_obs


@pytest.mark.ndsmoke
def test_mini_agent_docker_codex_code_live_optional():
    if os.getenv('DOCKER_MINI_AGENT','0') != '1':
        pytest.skip('DOCKER_MINI_AGENT not set; skipping live docker test')
    host = os.getenv('MINI_AGENT_API_HOST','127.0.0.1')
    port = int(os.getenv('MINI_AGENT_API_PORT','8788'))
    if not _can_connect(host, port):
        pytest.skip(f'mini-agent API not reachable on {host}:{port}')

    model = 'codex-agent/gpt-5'
    messages = [
        {"role":"system","content":"You may execute Python using available tools. If you produce code, run it and send stdout/stderr back. Iterate until tests pass."},
        {"role":"user","content":(
            "Implement a Python function `rotate_matrix_90(m)` that rotates an N x N matrix 90 degrees clockwise in-place."
            " Include a small main that prints the result on a sample matrix. Return only a Python code block."
        )},
    ]
    payload = {
        "messages": messages,
        "model": model,
        "tool_backend": "local",
        "use_tools": True,
        "auto_run_code_on_code_block": True,
        "max_iterations": 3,
        "enable_repair": True,
    }
    url=f"http://{host}:{port}/agent/run"
    r = httpx.post(url, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    assert data.get('ok') is True
    fa = (data.get('final_answer') or '').strip()
    assert len(fa) > 0
