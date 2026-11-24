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
from litellm.llms.custom_httpx import llm_http_handler
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.router_utils.cooldown_handlers import async_get_cooldown_deployments


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

    cooldown_ids = await async_get_cooldown_deployments(
        litellm_router_instance=router, parent_otel_span=None
    )
    assert failing_deployment_id in cooldown_ids, (
        f"Responses API failure callback did not register cooldown for "
        f"{failing_deployment_id!r}; cooldown set was {cooldown_ids}"
    )



@pytest.mark.asyncio
async def test_openrouter_responses_connection_failure_cools_down_the_deployment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse_connection(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused by the test transport", request=request)

    class RefusingHandler(AsyncHTTPHandler):
        """Also refuses on the fresh client the handler builds to retry a connection error."""

        def create_client(self, *_args: object, **_kwargs: object) -> httpx.AsyncClient:
            return httpx.AsyncClient(transport=httpx.MockTransport(refuse_connection))

    # The router does not forward a caller's client, so the handler's client factory supplies it.
    monkeypatch.setattr(llm_http_handler, "get_async_httpx_client", lambda **_kwargs: RefusingHandler())
    deployment_ids = ("deployment-a", "deployment-b")
    router = litellm.Router(
        model_list=[
            {
                "model_name": "openrouter-responses",
                "litellm_params": {
                    "model": "openrouter/openai/gpt-6.2-sol",
                    "api_key": "sk-or-test",
                    "api_base": "https://backend.example.test/api/v1",
                },
                "model_info": {"id": deployment_id},
            }
            for deployment_id in deployment_ids
        ],
        num_retries=0,
        allowed_fails=0,
        cooldown_time=60,
    )

    with pytest.raises(litellm.APIError, match="refused by the test transport"):
        await router.aresponses(model="openrouter-responses", input="hi")

    cooldowns = await router.cooldown_cache.async_get_active_cooldowns(
        model_ids=list(deployment_ids), parent_otel_span=None
    )
    assert len(cooldowns) == 1
