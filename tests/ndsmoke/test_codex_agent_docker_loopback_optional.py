"""
Docker codex-agent loopback smoke (optional).

Loops the codex-agent provider back to the mini-agent's OpenAI shim in Docker.

Run:
  DOCKER_MINI_AGENT=1 LITELLM_ENABLE_CODEX_AGENT=1 CODEX_AGENT_API_BASE=http://127.0.0.1:8788 \
    pytest -q tests/ndsmoke/test_codex_agent_docker_loopback_optional.py -q
"""
import os, socket, pytest
import litellm.llms.codex_agent  # ensure env-gated provider registers
from litellm import Router


def _can(host: str, port: int, t: float = 0.5) -> bool:
    import socket as _s
    try:
        with _s.create_connection((host, port), timeout=t):
            return True
    except OSError:
        return False


@pytest.mark.ndsmoke
@pytest.mark.e2e
def test_docker_codex_loopback_router_optional():
    if os.getenv("DOCKER_MINI_AGENT", "0") != "1":
        pytest.skip("DOCKER_MINI_AGENT not set; skipping codex loopback smoke")
    if os.getenv("MINI_AGENT_DOCKER_CODEX_LOOPBACK", "0") != "1":
        pytest.skip("codex loopback disabled; set MINI_AGENT_DOCKER_CODEX_LOOPBACK=1 to enable")
    base = os.getenv("CODEX_AGENT_API_BASE", "http://127.0.0.1:8788")
    import urllib.parse as _up
    p = _up.urlparse(base)
    host = p.hostname or "127.0.0.1"
    port = p.port or (443 if p.scheme == "https" else 80)
    if not _can(host, port):
        pytest.skip(f"codex loopback base not reachable: {base}")

    # Quick probe of the shim; if it doesn't accept a trivial payload, skip.
    import httpx
    try:
        _probe = httpx.post(
            base.rstrip("/") + "/v1/chat/completions",
            json={"model": "dummy", "messages": [{"role": "user", "content": "hi"}]},
            timeout=5.0,
        )
        if _probe.status_code >= 500:
            pytest.skip("agent shim not ready for loopback (500)")
    except Exception:
        pytest.skip("agent shim not reachable for loopback")

    r = Router(
        model_list=[
            {
                "model_name": "codex-agent-1",
                "litellm_params": {
                    "model": "codex-agent/mini",
                    "api_base": base,
                    "api_key": os.getenv("CODEX_AGENT_API_KEY", ""),
                },
            }
        ]
    )
    out = r.completion(
        model="codex-agent-1",
        messages=[{"role": "user", "content": "Say hello and finish."}],
    )
    content = getattr(getattr(out.choices[0], "message", {}), "content", None)
    assert isinstance(content, str) and len(content.strip()) > 0
