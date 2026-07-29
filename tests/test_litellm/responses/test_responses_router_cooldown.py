"""
Regression: Responses API router must register cooldowns on deployment
failures. Previously the Responses API path built ``litellm_params`` without
``model_info``, so ``Router.deployment_callback_on_failure`` exited early via
the "No model_info found" branch and the failing deployment was never added
to the cooldown set.
"""

import os
import sys
from unittest.mock import AsyncMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.router_utils.cooldown_handlers import _async_get_cooldown_deployments


@pytest.mark.asyncio
async def test_responses_api_rate_limit_marks_deployment_for_cooldown():
    failing_deployment_id = "deployment-rate-limited"

    router = litellm.Router(
        model_list=[
            {
                "model_name": "openai.gpt-5.1-codex",
                "litellm_params": {
                    "model": "openai/gpt-5.1-codex",
                    "api_key": "mock-api-key-1",
                },
                "model_info": {"id": failing_deployment_id},
            },
            {
                "model_name": "openai.gpt-5.1-codex",
                "litellm_params": {
                    "model": "openai/gpt-5.1-codex",
                    "api_key": "mock-api-key-2",
                },
                "model_info": {"id": "deployment-healthy"},
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

    def pin_to_failing_deployment(seq):
        for d in seq:
            if d["model_info"]["id"] == failing_deployment_id:
                return d
        return seq[0]

    with (
        patch(
            "litellm.llms.custom_httpx.llm_http_handler.BaseLLMHTTPHandler.async_response_api_handler",
            new_callable=AsyncMock,
            side_effect=rate_limit_error,
        ),
        patch(
            "litellm.router_strategy.simple_shuffle.random.choice",
            side_effect=pin_to_failing_deployment,
        ),
    ):
        with pytest.raises(litellm.RateLimitError):
            await router.aresponses(
                model="openai.gpt-5.1-codex",
                input="hi",
            )

    cooldown_ids = await _async_get_cooldown_deployments(
        litellm_router_instance=router, parent_otel_span=None
    )
    assert failing_deployment_id in cooldown_ids, (
        f"Responses API failure callback did not register cooldown for "
        f"{failing_deployment_id!r}; cooldown set was {cooldown_ids}"
    )
