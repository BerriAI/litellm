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
def test_exec_rpc_python_low_optional():
    host = os.getenv("EXEC_RPC_HOST", "127.0.0.1")
    port = int(os.getenv("EXEC_RPC_PORT", "8790"))
    if not _can(host, port):
        pytest.skip(f"exec-rpc not reachable on {host}:{port}")

    base = f"http://{host}:{port}"
    # health
    try:
        r = httpx.get(base + "/health", timeout=3.0)
        assert r.status_code == 200
    except Exception:
        pytest.skip("exec-rpc /health not responding as expected")

    # run simple python
    r = httpx.post(base + "/exec", json={"language": "python", "code": "print('OK')"}, timeout=10.0)
    r.raise_for_status()
    data = r.json()
    assert data.get("ok") is True
    assert "t_ms" in data and isinstance(data["t_ms"], (int, float))
    assert (data.get("stdout") or "").strip() == "OK"

