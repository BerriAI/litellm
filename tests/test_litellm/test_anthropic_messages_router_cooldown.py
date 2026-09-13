from unittest.mock import AsyncMock, patch

import httpx
import pytest

import litellm


@pytest.mark.asyncio
async def test_anthropic_messages_uses_deployment_cooldown_time():
    failing_deployment_id = "messages-deployment-rate-limited"

    router = litellm.Router(
        model_list=[
            {
                "model_name": "messages-model",
                "litellm_params": {
                    "model": "anthropic/claude-3-5-sonnet-20240620",
                    "api_key": "mock-api-key-1",
                    "cooldown_time": 0,
                },
                "model_info": {"id": failing_deployment_id},
            },
            {
                "model_name": "messages-model",
                "litellm_params": {
                    "model": "anthropic/claude-3-5-sonnet-20240620",
                    "api_key": "mock-api-key-2",
                },
                "model_info": {"id": "messages-deployment-healthy"},
            },
        ],
        num_retries=0,
        cooldown_time=60,
    )

    rate_limit_error = litellm.RateLimitError(
        message="upstream throttled",
        llm_provider="anthropic",
        model="anthropic/claude-3-5-sonnet-20240620",
        response=httpx.Response(
            status_code=429,
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        ),
    )

    def pin_to_failing_deployment(seq):
        for deployment in seq:
            if deployment["model_info"]["id"] == failing_deployment_id:
                return deployment
        return seq[0]

    with (
        patch(
            "litellm.llms.custom_httpx.llm_http_handler.BaseLLMHTTPHandler.async_anthropic_messages_handler",
            new_callable=AsyncMock,
            side_effect=rate_limit_error,
        ),
        patch(
            "litellm.router_strategy.simple_shuffle.random.choice",
            side_effect=pin_to_failing_deployment,
        ),
        patch("litellm.router._set_cooldown_deployments") as mock_set_cooldown,
    ):
        with pytest.raises(litellm.RateLimitError):
            await router.aanthropic_messages(
                model="messages-model",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=16,
            )

    mock_set_cooldown.assert_called_once()
    assert mock_set_cooldown.call_args.kwargs["deployment"] == failing_deployment_id
    assert mock_set_cooldown.call_args.kwargs["time_to_cooldown"] == 0
