from unittest.mock import AsyncMock

import pytest

from litellm.proxy.route_llm_request import _get_available_fallback_request


class _ListFallbackRouter:
    max_fallbacks = 5
    model_group_alias = {}

    def __init__(self, fallbacks, healthy_models):
        self.fallbacks = fallbacks
        self.healthy_models = set(healthy_models)
        self.model_list = [
            {"model_info": {"id": "primary-deployment"}},
            {"model_info": {"id": "first-deployment"}},
            {"model_info": {"id": "second-deployment"}},
            {"model_info": {"id": "later-deployment"}},
        ]
        self.async_pre_routing_hook = AsyncMock(return_value=None)

    async def async_get_healthy_deployments(self, *, model, **kwargs):
        if model not in self.healthy_models:
            return []
        deployment_id = {
            "first-model": "first-deployment",
            "second-model": "second-deployment",
            "later-model": "later-deployment",
        }[model]
        return [{"model_info": {"id": deployment_id}}]


@pytest.mark.asyncio
async def test_list_valued_direct_fallback_uses_first_healthy_candidate():
    router = _ListFallbackRouter(
        fallbacks=[{"model": ["first-model", "second-model"], "temperature": 0.2}],
        healthy_models={"second-model"},
    )

    request = await _get_available_fallback_request(
        llm_router=router,
        model_name="blocked-model",
        team_id=None,
        request_data={"messages": [{"role": "user", "content": "hello"}]},
    )

    assert request is not None
    assert request["model"] == "second-model"
    assert request["temperature"] == 0.2
    assert request["fallbacks"] == []
    assert [call.kwargs["model"] for call in router.async_pre_routing_hook.await_args_list] == [
        "first-model",
        "second-model",
    ]


@pytest.mark.asyncio
async def test_list_valued_direct_fallback_preserves_remaining_candidates_in_trusted_tail():
    router = _ListFallbackRouter(
        fallbacks=[
            {"model": ["first-model", "second-model"], "temperature": 0.2},
            "later-model",
        ],
        healthy_models={"first-model", "second-model", "later-model"},
    )

    request = await _get_available_fallback_request(
        llm_router=router,
        model_name="blocked-model",
        team_id=None,
        request_data={"messages": [{"role": "user", "content": "hello"}]},
    )

    assert request is not None
    assert request["model"] == "first-model"
    assert request["fallbacks"] == [
        {"model": "second-model", "temperature": 0.2},
        "later-model",
    ]


@pytest.mark.asyncio
async def test_later_list_valued_fallback_is_expanded_in_trusted_tail():
    router = _ListFallbackRouter(
        fallbacks=[
            {"model": ["first-model"], "temperature": 0.2},
            {"model": ["later-model"]},
        ],
        healthy_models={"first-model", "later-model"},
    )

    request = await _get_available_fallback_request(
        llm_router=router,
        model_name="blocked-model",
        team_id=None,
        request_data={"messages": [{"role": "user", "content": "hello"}]},
    )

    assert request is not None
    assert request["model"] == "first-model"
    assert request["fallbacks"] == [{"model": "later-model"}]
