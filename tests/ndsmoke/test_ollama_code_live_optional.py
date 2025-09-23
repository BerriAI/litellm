import os, socket, asyncio, pytest


def _ollama_host_port():
    host_env = os.getenv('OLLAMA_HOST')
    if host_env and ':' in host_env:
        h, p = host_env.rsplit(':', 1)
        try:
            return h, int(p)
        except Exception:
            pass
    return os.getenv('OLLAMA_HOST','127.0.0.1'), int(os.getenv('OLLAMA_PORT','11434'))


def _can_connect(host: str, port: int, timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.mark.ndsmoke
@pytest.mark.live_ollama
def test_ollama_code_model_live_optional():
    host, port = _ollama_host_port()
    if not _can_connect(host, port):
        pytest.skip(f'ollama not reachable on {host}:{port}')

    async def run():
        import httpx
        from litellm.router import Router

        # Prefer env LITELLM_DEFAULT_CODE_MODEL, else require glm4:latest
        preferred=[]
        env_code=os.getenv('LITELLM_DEFAULT_CODE_MODEL','')
        if env_code.startswith('ollama/'):
            preferred.append(env_code.split('ollama/',1)[1])
        preferred.append('glm4:latest')
        chosen=None
        try:
            tags=httpx.get(f'http://{host}:{port}/api/tags',timeout=1.0).json()
            names={m.get('name') for m in (tags.get('models') or [])}
            for name in preferred:
                if name in names:
                    chosen=f'ollama/{name}'; break
        except Exception:
            pass
        if not chosen:
            pytest.skip('no suitable ollama code model found; pull glm4:latest or set LITELLM_DEFAULT_CODE_MODEL')

        r = Router(model_list=[{"model_name":"m","litellm_params":{"model": chosen}}])
        prompt = 'Write a Python function sum_list(xs) that returns the sum of a list of ints.'
        try:
            resp = await r.acompletion(model='m', messages=[{"role":"user","content": prompt}], timeout=15)
        except Exception as e:
            import httpx
            if isinstance(e, httpx.ReadTimeout) or 'Timeout' in str(e):
                pytest.skip(f'ollama code request timeout: {e}')
            raise
        try:
            text = getattr(getattr(resp.choices[0],'message',{}),'content',None)
        except Exception:
            text = resp.get('choices',[{}])[0].get('message',{}).get('content')
        if not (isinstance(text,str) and len(text)>0):
            pytest.skip('ollama code model returned empty content; skipping')

    asyncio.run(run())
