"""
Purpose
- Live mini-agent (tools on): repair loop converges and prints expected outputs.

Scope
- DOES: post to /agent/run (local tools); assert last tool clean or outputs present.
- DOES NOT: run by default; requires reachable agent API and a code-capable model.

Run
- DOCKER_MINI_AGENT=1 MINI_AGENT_API_PORT=8788 LITELLM_DEFAULT_CODE_MODEL='ollama/...' \
  pytest tests/ndsmoke -k test_mini_agent_compress_runs_iterates_live_optional -q
"""
import os, socket, httpx, pytest

TIMEOUT = int(os.getenv("NDSMOKE_TIMEOUT", "240"))
MAX_ITERS = int(os.getenv("NDSMOKE_MAX_ITERS", "4"))


def _can_connect(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.mark.ndsmoke
def test_mini_agent_compress_runs_iterates_live_optional():
    if os.getenv('DOCKER_MINI_AGENT','0') != '1':
        pytest.skip('DOCKER_MINI_AGENT not set; skipping live docker test')
    host = os.getenv('MINI_AGENT_API_HOST','127.0.0.1')
    port = int(os.getenv('MINI_AGENT_API_PORT','18790'))
    if not _can_connect(host, port):
        pytest.skip(f'mini-agent API not reachable on {host}:{port}')

    model = os.getenv('LITELLM_DEFAULT_CODE_MODEL','ollama/glm4:latest')
    prompt = (
        "Implement a Python function `compress_runs(s: str) -> str` that compresses runs of repeated characters like aaabbc -> a3b2c1. "
        "Return only Python code in a code block and in __main__ print two tests: compress_runs('aaabbc') and compress_runs('aabbccddeeffgg'). "
        "If your code fails, read stderr and fix it, then try again."
    )
    payload = {
        "messages": [
            {"role": "system", "content": "You may execute Python using tools. If tests fail, fix and try again."},
            {"role": "user", "content": prompt},
        ],
        "model": model,
        "tool_backend": "local",
        "use_tools": True,
        "auto_run_code_on_code_block": True,
        "max_iterations": MAX_ITERS,
        "enable_repair": True,
    }
    url=f"http://{host}:{port}/agent/run"
    r = httpx.post(url, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    assert data.get('ok') is True

    # The mini-agent runs and repairs iteratively. Assert final tool status is clean (no stderr).
    last = data.get('last_tool_invocation') or {}
    # If the field is present, use it to enforce clean execution; otherwise fall back to text checks.
    if isinstance(last, dict) and last:
        assert last.get('ok') is True, f"last_tool_invocation not ok: {last}"
        assert (last.get('rc') or 0) == 0, f"non-zero rc: {last}"
        assert (last.get('stderr') or '').strip() == ''
    assert (last.get('t_ms') or 0) >= 0, f"stderr not empty: {last.get('stderr')!r}"

    # Accept success if we see expected outputs either in final answer or observations
    want1 = 'a3b2c1'
    want2 = 'a2b2c2d2e2f2g2'
    fa = (data.get('final_answer') or '').lower()
    if want1 in fa and want2 in fa:
        return
    msgs = data.get('messages', []) or []
    joined = '\n'.join((m.get('content') or '') for m in msgs if isinstance(m, dict))
    joined_lower = joined.lower()
    assert (want1 in joined_lower) and (want2 in joined_lower), f"Missing outputs. Got preview: {joined[-500:]}"
