"""
Codex-agent timeout/mapping smoke (optional).

Requires:
  - DOCKER_MINI_AGENT=1 (to target the dockerized agent shim)
  - MINI_AGENT_OPENAI_SHIM_DELAY_MS set in the agent-api environment externally (e.g., 500)

This test verifies that when the shim is slow, using a small timeout on the provider path
results in a handled error (no crash in Router) and a mapped APIConnectionError.
"""
import os, socket, pytest
from litellm import Router


def _can(host: str, port: int, t: float = 0.5) -> bool:
    import socket as _s
    try:
        with _s.create_connection((host, port), timeout=t):
            return True
    except OSError:
        return False


@pytest.mark.ndsmoke
def test_codex_agent_timeout_optional():
    if os.getenv("DOCKER_MINI_AGENT", "0") != "1":
        pytest.skip("DOCKER_MINI_AGENT not set; skipping")
    base = os.getenv("CODEX_AGENT_API_BASE", "http://127.0.0.1:8788")
    import urllib.parse as _up
    p = _up.urlparse(base)
    host = p.hostname or "127.0.0.1"
    port = p.port or (443 if p.scheme == "https" else 80)
    if not _can(host, port):
        pytest.skip(f"codex base not reachable: {base}")

    # Ensure provider is enabled
    if os.getenv("LITELLM_ENABLE_CODEX_AGENT", "") != "1":
        pytest.skip("codex-agent disabled (LITELLM_ENABLE_CODEX_AGENT!=1)")

    # Use a very small timeout to trigger mapping when shim is slow
    r = Router(
        model_list=[
            {
                "model_name": "codex-agent-1",
                "litellm_params": {
                    "model": "codex-agent/mini",
                    "api_base": base,
                    "api_key": os.getenv("CODEX_AGENT_API_KEY", ""),
                    "timeout": 0.1,
                },
            }
        ]
    )
    try:
        _ = r.completion(
            model="codex-agent-1",
            messages=[{"role": "user", "content": "Say hello and finish."}],
        )
    except Exception as e:  # APIConnectionError expected when delayed
        assert "APIConnectionError" in str(type(e)) or "timeout" in str(e).lower()
    else:
        pytest.skip("Timeout did not trigger; ensure MINI_AGENT_OPENAI_SHIM_DELAY_MS is set")

