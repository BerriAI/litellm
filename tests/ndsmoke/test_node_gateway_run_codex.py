import os, time, subprocess
import pytest

@pytest.mark.ndsmoke
def test_node_gateway_run_codex_presence_and_501():
    if subprocess.call(['node','-v'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
        pytest.skip('node not installed')
    env = os.environ.copy()
    env['PORT'] = '8791'
    # Ensure codex not configured
    env.pop('CODEX_BIN', None)
    proc = subprocess.Popen(['node','local/mini_agent/node_tools_gateway/server.mjs'], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        import httpx
        ok=False
        for _ in range(50):
            time.sleep(0.1)
            try:
                r=httpx.get('http://127.0.0.1:8791/tools',timeout=1.0)
                if r.status_code==200:
                    ok=True; break
            except Exception:
                pass
        assert ok
        tools = httpx.get('http://127.0.0.1:8791/tools',timeout=2.0).json()
        names = [t.get('function',{}).get('name') for t in tools]
        assert 'run_codex' in names
        # Invoke without CODEX_BIN configured → 501
        payload={'name':'run_codex','arguments':{'args':['--version']}}
        r=httpx.post('http://127.0.0.1:8791/invoke', json=payload, timeout=5.0)
        assert r.status_code==501
    finally:
        proc.kill()
        try: proc.wait(timeout=2)
        except Exception: pass
