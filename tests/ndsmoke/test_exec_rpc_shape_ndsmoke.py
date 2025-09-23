import os, httpx, pytest, socket


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
    host = os.getenv('MINI_AGENT_EXEC_BASE','http://127.0.0.1:18790').rstrip('/')
    if not _can_connect('127.0.0.1', 8790):
        pytest.skip('exec RPC not reachable')
    r = httpx.post(host+"/exec", json={"language":"python","code":"print(1)","timeout_sec":10.0}, timeout=15.0)
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
