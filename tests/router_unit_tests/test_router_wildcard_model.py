from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest

from litellm import Router


@pytest.mark.asyncio
async def test_wildcard_deployment_preserves_requested_model() -> None:
    """Provider wildcard credentials must not replace the concrete request model."""
    router = Router(model_list=[])
    deployment = {
        "model_name": "openai/*",
        "litellm_params": {"model": "openai/*", "api_key": "sk-test"},
        "model_info": {"id": "wildcard-deployment"},
    }
    original_function = AsyncMock(return_value="response")

    with (
        patch.object(
            router,
            "async_get_available_deployment",
            new=AsyncMock(return_value=deployment),
        ),
        patch.object(router, "async_routing_strategy_pre_call_checks", new=AsyncMock()),
        patch.object(router, "_get_client", return_value=None),
    ):
        result = await router._ageneric_api_call_with_fallbacks_helper(
            model="openai/gpt-5.3-codex",
            original_generic_function=original_function,
            messages=[{"role": "user", "content": "ping"}],
        )

    assert result == "response"
    assert original_function.await_args.kwargs["model"] == "openai/gpt-5.3-codex"
