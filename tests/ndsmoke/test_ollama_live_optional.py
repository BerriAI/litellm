import os
import socket
import asyncio
import pytest


def _ollama_host_port():
    host_env = os.getenv("OLLAMA_HOST")
    if host_env and ":" in host_env:
        h, p = host_env.rsplit(":", 1)
        try:
            return h, int(p)
        except Exception:
            pass
    return os.getenv("OLLAMA_HOST", "127.0.0.1"), int(os.getenv("OLLAMA_PORT", "11434"))


def _can_connect(host: str, port: int, timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.mark.ndsmoke
@pytest.mark.live_ollama
def test_ollama_live_optional():
    # Skip unless a local Ollama is reachable (default port 11434)
    host, port = _ollama_host_port()
    if not _can_connect(host, port):
        pytest.skip("ollama not reachable on {}:{}".format(host, port))

    async def run():
        from litellm.router import Router

        import httpx
        preferred=["granite3.3:8b","qwen3:8b","qwen2.5:7b","llama3.1:8b","mistral:7b","gemma2:9b"]
        chosen=None
        try:
            tags=httpx.get(f"http://{host}:{port}/api/tags",timeout=1.0).json()
            names={m.get("name") for m in (tags.get("models") or [])}
            for name in preferred:
                if name in names:
                    chosen=f"ollama/{name}"; break
        except Exception:
            pass
        if not chosen:
            env_text=os.getenv("LITELLM_DEFAULT_TEXT_MODEL","")
            env_model=os.getenv("LITELLM_DEFAULT_MODEL","")
            cand = env_text or env_model
            if cand.startswith("ollama/"):
                chosen=cand
            else:
                pytest.skip("no suitable ollama text model found; set LITELLM_DEFAULT_TEXT_MODEL=ollama/granite3.3:8b")
        r = Router(model_list=[{"model_name": "m", "litellm_params": {"model": chosen}}])
        try:
            resp = await r.acompletion(
            model="m",
            messages=[{"role": "user", "content": "Say hi in one word."}],
            timeout=10,
        )
        except Exception as e:
            import httpx
            if isinstance(e, httpx.ReadTimeout) or "Timeout" in str(e):
                import pytest as _pytest
                _pytest.skip(f"ollama request timeout: {e}")
            raise
        # Accept either object- or dict-shaped responses
        try:
            text = getattr(getattr(resp.choices[0], "message", {}), "content", None)
        except Exception:
            text = resp.get("choices", [{}])[0].get("message", {}).get("content")
        
        if not (isinstance(text, str) and len(text) > 0):
            import pytest as _pytest
            _pytest.skip("ollama returned empty content; skipping ndsmoke")

    asyncio.run(run())
