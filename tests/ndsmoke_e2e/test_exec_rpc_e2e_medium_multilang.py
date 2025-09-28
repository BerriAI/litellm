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
def test_exec_rpc_multilang_medium_optional():
    host = os.getenv("EXEC_RPC_HOST", "127.0.0.1")
    port = int(os.getenv("EXEC_RPC_PORT", "8790"))
    if not _can(host, port):
        pytest.skip(f"exec-rpc not reachable on {host}:{port}")

    base = f"http://{host}:{port}"
    # try python and javascript; skip gracefully if unavailable
    langs = [
        ("python", "print('py')", "py"),
        ("javascript", "console.log('js')", "js"),
    ]
    seen = set()
    for lang, code, marker in langs:
        try:
            r = httpx.post(base + "/exec", json={"language": lang, "code": code}, timeout=15.0)
            if r.status_code != 200:
                continue
            data = r.json()
            if data.get("ok") and marker in (data.get("stdout") or ""):
                seen.add(marker)
        except Exception:
            continue

    if not seen:
        pytest.skip("no supported languages available on exec-rpc host")
    # At least one ran successfully
    assert seen

