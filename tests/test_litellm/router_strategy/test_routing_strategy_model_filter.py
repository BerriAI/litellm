"""
Tests for `routing_strategy_model_filter` — restricts the configured routing
strategy to a list of model names. Other models fall back to simple-shuffle.
"""

import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm import Router
from litellm.types.router import PreRoutingHookResponse


def _build_router(strategy: str, model_filter):
    return Router(
        model_list=[
            {
                "model_name": "filtered-model",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_key": "sk-test-1",
                    "api_base": "https://example.invalid",
                },
                "model_info": {"id": "deploy-1"},
            },
            {
                "model_name": "filtered-model",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "sk-test-2",
                    "api_base": "https://example.invalid",
                },
                "model_info": {"id": "deploy-2"},
            },
            {
                "model_name": "other-model",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_key": "sk-test-3",
                    "api_base": "https://example.invalid",
                },
                "model_info": {"id": "deploy-3"},
            },
        ],
        routing_strategy=strategy,
        routing_strategy_model_filter=model_filter,
    )


def test_effective_strategy_no_filter_returns_configured():
    router = _build_router("latency-based-routing", model_filter=None)
    assert router._get_effective_routing_strategy("anything") == "latency-based-routing"


def test_effective_strategy_model_in_filter_returns_configured():
    router = _build_router(
        "latency-based-routing", model_filter=["filtered-model", "other"]
    )
    assert (
        router._get_effective_routing_strategy("filtered-model")
        == "latency-based-routing"
    )


def test_effective_strategy_model_not_in_filter_falls_back():
    router = _build_router("latency-based-routing", model_filter=["filtered-model"])
    assert router._get_effective_routing_strategy("other-model") == "simple-shuffle"


def test_effective_strategy_empty_filter_falls_back_for_all():
    router = _build_router("latency-based-routing", model_filter=[])
    assert router._get_effective_routing_strategy("filtered-model") == "simple-shuffle"
    assert router._get_effective_routing_strategy("other-model") == "simple-shuffle"


@pytest.mark.asyncio
async def test_async_dispatch_uses_strategy_for_filtered_model():
    router = _build_router("latency-based-routing", model_filter=["filtered-model"])

    with (
        patch.object(
            router.lowestlatency_logger,
            "async_get_available_deployments",
            wraps=router.lowestlatency_logger.async_get_available_deployments,
        ) as latency_spy,
        patch(
            "litellm.router.simple_shuffle", wraps=litellm.router.simple_shuffle
        ) as shuffle_spy,
    ):
        await router.async_get_available_deployment(
            model="filtered-model", request_kwargs={}
        )

        assert latency_spy.called, "latency strategy should run for filtered model"
        assert (
            not shuffle_spy.called
        ), "simple_shuffle should not run for filtered model"


@pytest.mark.asyncio
async def test_async_dispatch_falls_back_to_simple_shuffle_for_other_models():
    router = _build_router("latency-based-routing", model_filter=["filtered-model"])

    with (
        patch.object(
            router.lowestlatency_logger,
            "async_get_available_deployments",
            wraps=router.lowestlatency_logger.async_get_available_deployments,
        ) as latency_spy,
        patch(
            "litellm.router.simple_shuffle", wraps=litellm.router.simple_shuffle
        ) as shuffle_spy,
    ):
        await router.async_get_available_deployment(
            model="other-model", request_kwargs={}
        )

        assert shuffle_spy.called, "simple_shuffle should run for unfiltered model"
        assert (
            not latency_spy.called
        ), "latency strategy should not run for unfiltered model"


def test_sync_dispatch_falls_back_to_simple_shuffle_for_other_models():
    router = _build_router("latency-based-routing", model_filter=["filtered-model"])

    with (
        patch.object(
            router.lowestlatency_logger,
            "get_available_deployments",
            wraps=router.lowestlatency_logger.get_available_deployments,
        ) as latency_spy,
        patch(
            "litellm.router.simple_shuffle", wraps=litellm.router.simple_shuffle
        ) as shuffle_spy,
    ):
        router.get_available_deployment(model="other-model")

        assert shuffle_spy.called
        assert not latency_spy.called


@pytest.mark.asyncio
async def test_async_filter_recalculated_after_pre_routing_hook_rewrites_model():
    """
    The pre-routing hook can replace the incoming model name. The filter must
    key off the post-hook model so the strategy reflects the actual deployment
    being routed to.
    """
    router = _build_router("latency-based-routing", model_filter=["filtered-model"])

    # Hook rewrites filtered-model -> other-model. Effective strategy should
    # therefore become simple-shuffle, not latency-based-routing.
    router.async_pre_routing_hook = AsyncMock(
        return_value=PreRoutingHookResponse(model="other-model", messages=None)
    )

    with (
        patch.object(
            router.lowestlatency_logger,
            "async_get_available_deployments",
            wraps=router.lowestlatency_logger.async_get_available_deployments,
        ) as latency_spy,
        patch(
            "litellm.router.simple_shuffle", wraps=litellm.router.simple_shuffle
        ) as shuffle_spy,
    ):
        await router.async_get_available_deployment(
            model="filtered-model", request_kwargs={}
        )

        assert shuffle_spy.called, "simple_shuffle should run for the post-hook model"
        assert (
            not latency_spy.called
        ), "latency strategy should not run since post-hook model is not in filter"


@pytest.mark.asyncio
async def test_async_filter_recalculated_when_hook_rewrites_into_filter():
    """
    Inverse direction: hook rewrites a non-filtered model into a filtered one.
    The configured strategy should then apply, not the simple-shuffle fallback.
    """
    router = _build_router("latency-based-routing", model_filter=["filtered-model"])

    router.async_pre_routing_hook = AsyncMock(
        return_value=PreRoutingHookResponse(model="filtered-model", messages=None)
    )

    with (
        patch.object(
            router.lowestlatency_logger,
            "async_get_available_deployments",
            wraps=router.lowestlatency_logger.async_get_available_deployments,
        ) as latency_spy,
        patch(
            "litellm.router.simple_shuffle", wraps=litellm.router.simple_shuffle
        ) as shuffle_spy,
    ):
        await router.async_get_available_deployment(
            model="other-model", request_kwargs={}
        )

        assert latency_spy.called, "latency strategy should run for post-hook model"
        assert (
            not shuffle_spy.called
        ), "simple_shuffle should not run since post-hook model is in filter"


def test_filter_propagates_through_update_settings():
    router = _build_router("latency-based-routing", model_filter=None)
    assert router.routing_strategy_model_filter is None

    router.update_settings(routing_strategy_model_filter=["filtered-model"])
    assert router.routing_strategy_model_filter == ["filtered-model"]
    assert router._get_effective_routing_strategy("other-model") == "simple-shuffle"
