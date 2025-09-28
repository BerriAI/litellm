import os, socket, urllib.parse
import pytest
from litellm import Router


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
@pytest.mark.e2e
@pytest.mark.asyncio
async def test_codex_agent_router_low_optional():
    if os.getenv("LITELLM_ENABLE_CODEX_AGENT", "") != "1":
        pytest.skip("codex-agent disabled (LITELLM_ENABLE_CODEX_AGENT!=1)")
    base = os.getenv("CODEX_AGENT_API_BASE")
    if not base or not _reachable(base):
        pytest.skip("CODEX_AGENT_API_BASE not set or not reachable")

    router = Router(
        model_list=[
            {"model_name": "codex-agent-1", "litellm_params": {
                "model": "codex-agent/mini",
                "api_base": base,
                "api_key": os.getenv("CODEX_AGENT_API_KEY", "")
            }},
        ]
    )

    out = await router.acompletion(
        model="codex-agent-1",
        messages=[{"role": "user", "content": "Say hello and finish."}],
    )
    text = getattr(out, "text", None)
    if isinstance(text, str) and text.strip():
        return
    try:
        content = out.choices[0].message.content  # type: ignore[attr-defined]
    except Exception:
        content = None
    assert isinstance(content, str) and content.strip()

