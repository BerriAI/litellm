"""
Boat, Tensorlake, Blaxel and Modal sandboxes - end-to-end integration tests.

These tests make REAL calls to each provider and are skipped unless that provider's credentials are set.

Run only these tests:
    pytest tests/integration/sandbox/test_sandbox_providers.py -v
"""

import os
from typing import Final

import pytest

import litellm

PROVIDER_ENV: Final = {
    "boat": ("BOAT_API_KEY",),
    "tensorlake": ("TENSORLAKE_API_KEY",),
    "blaxel": ("BL_API_KEY",),
    "modal": ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"),
}
PROVIDERS: Final = [
    pytest.param(
        provider,
        marks=pytest.mark.skipif(
            any(name not in os.environ for name in names), reason=f"needs real {' and '.join(names)}"
        ),
    )
    for provider, names in PROVIDER_ENV.items()
]


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.asyncio
async def test_integration_lifecycle_and_cost(provider):
    container = await litellm.acreate_sandbox(provider=provider, timeout=120)
    try:
        ok = await litellm.arun_code(provider=provider, container=container, code="print(sum(range(10)))")
        failed = await litellm.arun_code(
            provider=provider, container=container, code="echo out; echo err >&2; exit 3", language="bash"
        )
    finally:
        assert await litellm.adelete_sandbox(provider=provider, container=container)
    cost = await litellm.asandbox_cost(provider=provider, container=container)

    assert ok.stdout.strip() == "45"
    assert ok.error is None
    assert (failed.stdout.strip(), failed.stderr.strip(), failed.exit_code) == ("out", "err", 3)
    assert cost.source == ("provider" if provider == "boat" else "estimate")
    assert cost.cost_usd is not None and cost.cost_usd >= 0


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.asyncio
async def test_integration_ephemeral_reports_cost(provider):
    result = await litellm.acode_interpreter_tool(provider=provider, code="print(6*7)", timeout=120)

    assert result.stdout.strip() == "42"
    assert result._hidden_params["response_cost"] >= 0
