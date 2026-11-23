"""
Regression: Responses API router must register cooldowns on deployment
failures. Previously the Responses API path built ``litellm_params`` without
``model_info``, so ``Router.deployment_callback_on_failure`` exited early via
the "No model_info found" branch and the failing deployment was never added
to the cooldown set.
"""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

import litellm


@pytest.mark.asyncio
async def test_responses_api_rate_limit_marks_deployment_for_cooldown():
    deployment_ids = {"mock-api-key-1": "deployment-a", "mock-api-key-2": "deployment-b"}

    router = litellm.Router(
        model_list=[
            {
                "model_name": "openai.gpt-5.1-codex",
                "litellm_params": {
                    "model": "openai/gpt-5.1-codex",
                    "api_key": "mock-api-key-1",
                },
                "model_info": {"id": "deployment-a"},
            },
            {
                "model_name": "openai.gpt-5.1-codex",
                "litellm_params": {
                    "model": "openai/gpt-5.1-codex",
                    "api_key": "mock-api-key-2",
                },
                "model_info": {"id": "deployment-b"},
            },
        ],
        num_retries=0,
        cooldown_time=60,
    )

    rate_limit_error = litellm.RateLimitError(
        message="upstream throttled",
        llm_provider="openai",
        model="openai/gpt-5.1-codex",
        response=httpx.Response(
            status_code=429,
            request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
        ),
    )

    called_api_keys = []

    async def fail_and_record(*args, **kwargs):
        called_api_keys.append(kwargs["litellm_params"].api_key)
        raise rate_limit_error

    with patch(
        "litellm.llms.custom_httpx.llm_http_handler.BaseLLMHTTPHandler.async_response_api_handler",
        new_callable=AsyncMock,
        side_effect=fail_and_record,
    ):
        with pytest.raises(litellm.RateLimitError):
            await router.aresponses(
                model="openai.gpt-5.1-codex",
                input="hi",
            )

    # num_retries=0: exactly one deployment took the request and failed.
    (called_api_key,) = called_api_keys
    failed_id = deployment_ids[called_api_key]
    cooldown_ids = [
        model_id
        for model_id, _ in await router.cooldown_cache.async_get_active_cooldowns(
            model_ids=list(deployment_ids.values()), parent_otel_span=None
        )
    ]
    assert cooldown_ids == [failed_id], (
        f"Responses API failure callback did not register cooldown for {failed_id!r}; cooldown set was {cooldown_ids}"
    )
