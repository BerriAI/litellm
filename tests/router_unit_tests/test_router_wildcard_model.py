from unittest.mock import AsyncMock

import pytest

from litellm import Router


@pytest.mark.asyncio
async def test_wildcard_deployment_preserves_requested_model() -> None:
    """Provider wildcard credentials must not replace the concrete request model."""
    router = Router(
        model_list=[
            {
                "model_name": "openai/*",
                "litellm_params": {"model": "openai/*", "api_key": "sk-test"},
                "model_info": {"id": "wildcard-deployment"},
            }
        ]
    )
    original_function = AsyncMock(return_value="response")

    result = await router._ageneric_api_call_with_fallbacks_helper(
        model="openai/gpt-5.3-codex",
        original_generic_function=original_function,
        messages=[{"role": "user", "content": "ping"}],
    )

    assert result == "response"
    assert original_function.await_args.kwargs["model"] == "openai/gpt-5.3-codex"
