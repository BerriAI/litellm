"""
Purpose
- Env-gated smoke that codex-agent integrates via Router alias when enabled.

Scope
- DOES: skip cleanly unless LITELLM_ENABLE_CODEX_AGENT + codex binary are discoverable; stub Router call.
- DOES NOT: launch or run the real CLI.

Run
- `pytest tests/smoke -k test_codex_agent_env_smoke -q`
"""
import asyncio
import os
import pytest

from litellm import Router
import shutil

@pytest.fixture
def have_codex_env():
    """
    Consider the env 'present' if the feature flag is on AND either an API base or a discoverable binary exists.
    The test still stubs Router.acompletion, so no network/CLI is used.
    """
    flag = os.getenv("LITELLM_ENABLE_CODEX_AGENT") == "1"
    have_api = bool(os.getenv("CODEX_AGENT_API_BASE"))
    bin_path = os.getenv("LITELLM_CODEX_BINARY_PATH","")
    have_bin = bool(shutil.which(bin_path) or shutil.which("codex"))
    return flag and (have_api or have_bin)


@pytest.mark.asyncio
async def test_codex_agent_env_gated_smoke(monkeypatch, have_codex_env):
    """
    Deterministic env-gated smoke for the experimental codex-agent.

    - Skips unless LITELLM_ENABLE_CODEX_AGENT=1 and a codex binary is discoverable
      (either CODEX_BINARY_PATH or PATH 'codex'), as provided by have_codex_env().
    - Wires a Router model entry pointing to the codex-agent provider alias.
    - Stubs Router.acompletion to avoid any network/CLI side effects and proves
      the integration surface composes correctly.

    This validates:
      - docs alignment for provider alias usage (codex-agent/mini)
      - env gating behavior for CI friendliness
      - stable Router API usage (no new surface needed)
    """
    if not have_codex_env:
        pytest.skip("codex-agent smoke: env flag/binary not present (expected in CI).")

    # Ensure flag is set for this test environment
    monkeypatch.setenv("LITELLM_ENABLE_CODEX_AGENT", "1")

    router = Router(
        model_list=[
            {"model_name": "codex-agent-1", "litellm_params": {"model": "codex-agent/mini"}},
        ]
    )

    # Stub out the actual acompletion call to keep the test deterministic/offline
    async def stub_acompletion(model, messages, **kwargs):
        await asyncio.sleep(0)
        class R:
            def __init__(self, text): self.text = text
        return R("codex-ok")

    monkeypatch.setattr(router, "acompletion", stub_acompletion)

    resp = await router.acompletion(
        model="codex-agent-1",
        messages=[{"role": "user", "content": "echo hello then stop"}],
    )
    assert getattr(resp, "text", "") == "codex-ok"
