"""
Purpose
- Live robustness: agent repairs a common glm4 Python bug (stray __main__) and prints target outputs.

Scope
- DOES: iterate with auto-run; assert expected prints appear in final/msgs.
- DOES NOT: run by default; model-dependent.

Run
- DOCKER_MINI_AGENT=1 MINI_AGENT_API_PORT=8788 LITELLM_DEFAULT_CODE_MODEL='ollama/glm4:latest' \
  pytest tests/ndsmoke -k test_mini_agent_glm4_bugfix_iterates_live_optional -q
"""
import os, socket, httpx, pytest

TIMEOUT = int(os.getenv("NDSMOKE_TIMEOUT", "240"))


def _can_connect(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.mark.ndsmoke
def test_mini_agent_glm4_bugfix_iterates_live_optional():
    """
    Start with a common glm4 code bug (stray `__main__` line) and ensure the mini-agent
    runs it, captures the failure, repairs, and prints expected outputs.
    """
    if os.getenv('DOCKER_MINI_AGENT','0') != '1':
        pytest.skip('DOCKER_MINI_AGENT not set; skipping live docker test')
    host = os.getenv('MINI_AGENT_API_HOST','127.0.0.1')
    port = int(os.getenv('MINI_AGENT_API_PORT','8788'))
    if not _can_connect(host, port):
        pytest.skip(f'mini-agent API not reachable on {host}:{port}')

    # Use glm4 chat by default to mirror the user scenario
    model = os.getenv('LITELLM_DEFAULT_CODE_MODEL','ollama/glm4:latest')

    buggy_code = '''```python
def compress_runs(s: str) -> str:
    compressed = []
    count = 1

    for i in range(1, len(s)):
        if s[i] == s[i-1]:
            count += 1
        else:
            compressed.append(s[i-1])
            compressed.append(str(count))
            count = 1

    # Add the last character and its count
    compressed.append(s[-1])
    compressed.append(str(count))

    return ''.join(compressed)

__main__
print(compress_runs("aaabbc"))  # Should print 'a3b2c1'
print(compress_runs("AABBBCCDAABBB"))  # Should print 'A1B3C2D1A3B3'
```'''

    prompt = (
        "Run the following Python code block as-is. If it fails, read stderr, fix the code (remove stray tokens, add proper __main__ guard), "
        "and try again until it prints the expected outputs. Only return Python in code blocks when proposing fixes.\n\n" + buggy_code
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
        "max_iterations": 4,
        "enable_repair": True,
    }
    url=f"http://{host}:{port}/agent/run"
    r = httpx.post(url, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    assert data.get('ok') is True

    want1 = 'a3b2c1'.lower()
    want2 = 'A1B3C2D1A3B3'.lower()
    fa = (data.get('final_answer') or '').lower()
    msgs = data.get('messages', []) or []
    joined = '\n'.join((m.get('content') or '') for m in msgs if isinstance(m, dict)).lower()

    assert (want1 in fa or want1 in joined) and (want2 in fa or want2 in joined), \
        f"Expected outputs not found. Preview: {joined[-500:]}"
