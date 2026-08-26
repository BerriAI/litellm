from unittest.mock import AsyncMock, patch

import pytest

from litellm.proxy.route_llm_request import _get_available_fallback_request
from litellm.router_utils.common_utils import filter_team_based_models
from litellm.router_utils.fallback_event_handlers import run_async_fallback


class _PreflightRouter:
    max_fallbacks = 5
    model_group_alias = None

    def __init__(self) -> None:
        self.fallbacks = [
            {
                "public-model": [
                    {
                        "model": "fallback-model",
                        "metadata": {"user_api_key_auth": "attacker-auth"},
                    }
                ]
            }
        ]
        self.model_list = [
            {
                "model_name": "fallback-model",
                "model_info": {"id": "fallback-deployment"},
            }
        ]
        self.request_kwargs = None

    async def async_pre_routing_hook(self, **kwargs):
        return None

    async def async_get_healthy_deployments(self, *, request_kwargs, **kwargs):
        self.request_kwargs = request_kwargs
        return [{"model_info": {"id": "fallback-deployment"}}]


@pytest.mark.asyncio
async def test_fallback_preflight_preserves_authenticated_api_key_context():
    router = _PreflightRouter()
    trusted_auth = object()

    fallback_request = await _get_available_fallback_request(
        llm_router=router,
        model_name="public-model",
        team_id=None,
        request_data={
            "metadata": {"user_api_key_auth": trusted_auth},
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert fallback_request is not None
    assert router.request_kwargs["metadata"]["user_api_key_auth"] is trusted_auth
    assert fallback_request["metadata"]["user_api_key_auth"] is trusted_auth


@pytest.mark.asyncio
async def test_fallback_preflight_strips_injected_api_key_context_when_none_authenticated():
    router = _PreflightRouter()

    fallback_request = await _get_available_fallback_request(
        llm_router=router,
        model_name="public-model",
        team_id=None,
        request_data={"messages": [{"role": "user", "content": "Hello"}]},
    )

    assert fallback_request is not None
    assert "user_api_key_auth" not in router.request_kwargs.get("metadata", {})
    assert "user_api_key_auth" not in fallback_request.get("metadata", {})


def test_retry_scoped_deployment_exclusions_survive_router_pop_until_fallback_advances():
    deployments = [
        {"model_info": {"id": "validated-deployment"}},
        {"model_info": {"id": "unvalidated-deployment"}},
    ]
    request_kwargs = {
        "fallback_depth": 1,
        "_excluded_deployment_ids": ["unvalidated-deployment"],
    }

    first_lookup = filter_team_based_models(deployments, request_kwargs)
    assert [item["model_info"]["id"] for item in first_lookup] == ["validated-deployment"]

    # Router health selection consumes the public exclusion key after team
    # filtering. A retry at the same fallback depth must restore it.
    request_kwargs.pop("_excluded_deployment_ids")
    retry_lookup = filter_team_based_models(deployments, request_kwargs)
    assert [item["model_info"]["id"] for item in retry_lookup] == ["validated-deployment"]
    assert request_kwargs["_excluded_deployment_ids"] == ["unvalidated-deployment"]

    # Moving to the next trusted fallback increments fallback_depth. The old
    # target's exclusions must no longer constrain that new fallback.
    request_kwargs.pop("_excluded_deployment_ids")
    request_kwargs["fallback_depth"] = 2
    next_fallback_lookup = filter_team_based_models(deployments, request_kwargs)
    assert [item["model_info"]["id"] for item in next_fallback_lookup] == [
        "validated-deployment",
        "unvalidated-deployment",
    ]


class _RuntimeFallbackRouter:
    model_group_alias = None

    def __init__(self) -> None:
        self.request_kwargs = None

    def log_retry(self, kwargs, e):
        return kwargs

    async def async_function_with_fallbacks(self, *args, **kwargs):
        self.request_kwargs = kwargs
        return "fallback response"


@pytest.mark.asyncio
async def test_runtime_fallback_preserves_authenticated_api_key_context():
    router = _RuntimeFallbackRouter()
    trusted_auth = object()

    async def _original_function():
        return None

    with (
        patch(
            "litellm.router_utils.fallback_event_handlers.add_fallback_headers_to_response",
            side_effect=lambda response, **kwargs: response,
        ),
        patch(
            "litellm.router_utils.fallback_event_handlers.log_success_fallback_event",
            new=AsyncMock(),
        ),
    ):
        response = await run_async_fallback(
            litellm_router=router,
            fallback_model_group=[
                {
                    "model": "fallback-model",
                    "metadata": {"user_api_key_auth": "attacker-auth"},
                }
            ],
            original_model_group="primary-model",
            original_exception=RuntimeError("primary failed"),
            max_fallbacks=1,
            fallback_depth=0,
            original_function=_original_function,
            metadata={"user_api_key_auth": trusted_auth},
        )

    assert response == "fallback response"
    assert router.request_kwargs["metadata"]["user_api_key_auth"] is trusted_auth
