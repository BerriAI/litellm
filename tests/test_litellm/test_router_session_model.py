from unittest.mock import AsyncMock

import pytest

from litellm import Router


@pytest.mark.asyncio
async def test_ageneric_api_call_with_fallbacks_resolves_session_model():
    """Realtime session.model must use the picked deployment, not the caller alias."""
    mock_function = AsyncMock()
    mock_function.__name__ = "acreate_realtime_client_secret"
    mock_function.return_value = {"value": "ek_test"}

    router = Router(
        model_list=[
            {
                "model_name": "gpt-realtime-2-1-mini",
                "litellm_params": {
                    "model": "azure/gpt-realtime-2-1-mini-deployment",
                    "api_key": "fake-api-key",
                    "api_base": "https://fake.openai.azure.com",
                },
            }
        ]
    )

    await router._ageneric_api_call_with_fallbacks(
        model="gpt-realtime-2-1-mini",
        original_function=mock_function,
        session={"type": "realtime", "model": "gpt-realtime-2-1-mini"},
    )

    call_kwargs = mock_function.call_args.kwargs
    assert call_kwargs["model"] == "azure/gpt-realtime-2-1-mini-deployment"
    assert call_kwargs["session"]["model"] == "azure/gpt-realtime-2-1-mini-deployment"
