import os, json, time, subprocess, sys
import pytest

@pytest.mark.ndsmoke
def test_node_gateway_run_agent_basic(tmp_path):
    # Skip if node not available
    if subprocess.call(['node','-v'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
        pytest.skip('node not installed')
    # Start gateway
    env = os.environ.copy()
    env['PORT'] = '8790'
    proc = subprocess.Popen(['node','local/mini_agent/node_tools_gateway/server.mjs'], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        # wait for readiness
        t0=time.time()
        ok=False
        import httpx
        for _ in range(50):
            time.sleep(0.1)
            try:
                r=httpx.get('http://127.0.0.1:8790/tools',timeout=1.0)
                if r.status_code==200:
                    ok=True; break
            except Exception:
                pass
        assert ok, 'gateway not ready'
        # Call run_litellm_agent with local backend (echo through agent)
        payload={
            'name':'run_litellm_agent',
            'arguments':{
                'model':'noop',
                'messages':[{'role':'user','content':'Say hi and finish.'}],
                'max_iterations':1,
            }
        }
        r=httpx.post('http://127.0.0.1:8790/invoke', json=payload, timeout=30.0)
        assert r.status_code==200, r.text
        data=r.json()
        assert 'stopped_reason' in data and 'messages' in data
    finally:
        proc.kill()
        try: proc.wait(timeout=2)
        except Exception: pass
