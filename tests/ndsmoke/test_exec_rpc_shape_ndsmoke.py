"""
Purpose
- Infra sanity: Exec RPC is reachable and always returns {ok, rc, stdout, stderr, t_ms}.

Scope
- DOES: post to /exec('python', 'print(1)'); assert t_ms present.
- DOES NOT: run by default; relies on reachable Exec RPC container.

Run
- DOCKER_MINI_AGENT=1 MINI_AGENT_EXEC_BASE=http://127.0.0.1:8792 \
  pytest tests/ndsmoke -k test_exec_rpc_shape_python_print1_optional -q
"""
import os, httpx, pytest, socket
from urllib.parse import urlsplit


def _can_connect(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.mark.ndsmoke
def test_exec_rpc_shape_python_print1_optional():
    if os.getenv('DOCKER_MINI_AGENT','0') != '1':
        pytest.skip('DOCKER_MINI_AGENT not set; skipping live ndsmoke')

    base = os.getenv('MINI_AGENT_EXEC_BASE','http://127.0.0.1:8792').rstrip('/')
    p = urlsplit(base)
    host = p.hostname or '127.0.0.1'
    port = p.port or (443 if p.scheme == 'https' else 80)
    if not _can_connect(host, port):
        pytest.skip(f'exec RPC not reachable on {host}:{port}')

    r = httpx.post(base + "/exec", json={"language":"python","code":"print(1)","timeout_sec":10.0}, timeout=15.0)
    r.raise_for_status()
    data = r.json()
    assert isinstance(data, dict)
    for k in ("ok","rc","stdout","stderr","t_ms"):
        assert k in data
    assert isinstance(data["ok"], bool)
    assert isinstance(data.get("rc"), (int,type(None)))
    assert isinstance(data.get("stdout"), str)
    assert isinstance(data.get("stderr"), str)
    assert isinstance(data.get("t_ms"), (int,float))
