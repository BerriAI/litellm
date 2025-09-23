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

from typing import List, Tuple


def _can_connect(host: str, port: int, timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.mark.ndsmoke
@pytest.mark.live_ollama
def test_ollama_batch_as_completed_optional():
    host, port = _ollama_host_port()
    if not _can_connect(host, port):
        pytest.skip(f'ollama not reachable on {host}:{port}')

    async def run():
        import httpx
        from litellm.router import Router
        from litellm.extras.batch import acompletion_as_completed

        # Pick a text model from tags
        preferred = [
            'granite3.3:8b','qwen3:8b','qwen2.5:7b','llama3.1:8b','mistral:7b','gemma2:9b'
        ]
        chosen = None
        try:
            tags = httpx.get(f'http://{host}:{port}/api/tags', timeout=1.0).json()
            names = { m.get('name') for m in (tags.get('models') or []) }
            for name in preferred:
                if name in names:
                    chosen = f'ollama/{name}'; break
        except Exception:
            pass
        if not chosen:
            env_text = os.getenv('LITELLM_DEFAULT_TEXT_MODEL','')
            env_model = os.getenv('LITELLM_DEFAULT_MODEL','')
            cand = env_text or env_model
            if cand.startswith('ollama/'):
                chosen = cand
            else:
                pytest.skip('no suitable ollama text model found')

        r = Router(model_list=[{"model_name":"m","litellm_params":{"model": chosen}}])

        prompts = [
            'Say hi in one word.',
            '2+2=?',
            'Name a primary color.'
        ]
        reqs = [{"model":"m","messages":[{"role":"user","content":p}]} for p in prompts]
        out: List[Tuple[int,str]] = []
        async for idx, resp in acompletion_as_completed(r, reqs, concurrency=3):
            try:
                text = getattr(getattr(resp.choices[0],'message',{}),'content',None)
            except Exception:
                text = resp.get('choices',[{}])[0].get('message',{}).get('content')
            assert isinstance(text, str) and len(text) > 0
            out.append((idx, text))
        assert len(out) == len(prompts)

    asyncio.run(run())
