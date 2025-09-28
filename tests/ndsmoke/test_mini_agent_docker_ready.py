"""
Docker mini-agent readiness smoke (optional).

Run:
  DOCKER_MINI_AGENT=1 pytest -q tests/ndsmoke/test_mini_agent_docker_ready.py -q
"""
import os, socket, pytest, httpx


def _can(host: str, port: int, t: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=t):
            return True
    except OSError:
        return False


@pytest.mark.ndsmoke
def test_mini_agent_docker_ready_optional():
    if os.getenv("DOCKER_MINI_AGENT", "0") != "1":
        pytest.skip("DOCKER_MINI_AGENT not set; skipping docker readiness smoke")
    host = os.getenv("MINI_AGENT_API_HOST", "127.0.0.1")
    port = int(os.getenv("MINI_AGENT_API_PORT", "8788"))
    if not _can(host, port):
        pytest.skip(f"mini-agent API not reachable on {host}:{port}")
    r = httpx.get(f"http://{host}:{port}/ready", timeout=5.0)
    assert r.status_code == 200
    assert (r.json() or {}).get("ok") is True

