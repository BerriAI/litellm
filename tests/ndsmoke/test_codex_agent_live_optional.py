"""
Live optional ndsmoke for codex-agent via configured HTTP endpoint.

Skips unless:
- LITELLM_ENABLE_CODEX_AGENT=1
- CODEX_AGENT_API_BASE is set and responds on /v1/chat/completions
"""
import os
import socket
import urllib.parse
import httpx
import pytest


def _reachable(url: str, timeout: float = 0.5) -> bool:
    try:
        p = urllib.parse.urlparse(url)
        host = p.hostname or "127.0.0.1"
        port = p.port or (443 if p.scheme == "https" else 80)
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.mark.ndsmoke
@pytest.mark.asyncio
async def test_codex_agent_live_optional():
    if os.getenv("LITELLM_ENABLE_CODEX_AGENT", "") != "1":
        pytest.skip("codex-agent disabled (LITELLM_ENABLE_CODEX_AGENT!=1)")
    base = os.getenv("CODEX_AGENT_API_BASE")
    if not base or not _reachable(base):
        pytest.skip("CODEX_AGENT_API_BASE not set or not reachable")

    # quick sanity on endpoint
    try:
        r = httpx.post(base.rstrip("/") + "/v1/chat/completions", json={"model": "dummy", "messages": [{"role": "user", "content": "hi"}]}, timeout=2.0)
        assert r.status_code in (200, 400, 422)
    except Exception:
        pytest.skip("codex-agent endpoint not responding as expected")

    from litellm import Router

    router = Router(
        model_list=[
            {"model_name": "codex-agent-1", "litellm_params": {"model": "codex-agent/mini"}},
        ]
    )

    out = await router.acompletion(
        model="codex-agent-1",
        messages=[{"role": "user", "content": "Say hi and finish."}],
    )
    # Accept either text or OpenAI-like response
    text = getattr(out, "text", None)
    if isinstance(text, str) and text.strip():
        return
    content = None
    try:
        content = out.choices[0].message.content  # type: ignore[attr-defined]
    except Exception:
        content = None
    assert isinstance(content, str)
