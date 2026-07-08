import pytest
from unittest.mock import AsyncMock

from litellm import Router
from litellm.router_strategy.lar1_routing import (
    LAR1RoutingStrategy,
    _normalize_thresholds,
    _parse_lar1_metadata,
    apply_lar1_routing_strategy,
    lar1_thresholds_from_args,
    DEFAULT_THRESHOLDS,
)


def _model_list():
    return [
        {
            "model_name": "agent-router",
            "litellm_params": {
                "model": "openai/gpt-4o",
                "api_key": "fake-key",
            },
            "model_info": {"id": "cloud-smart", "type": "cloud-smart"},
        },
        {
            "model_name": "agent-router",
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "api_key": "fake-key",
            },
            "model_info": {"id": "cloud-fast", "type": "cloud-fast"},
        },
        {
            "model_name": "agent-router",
            "litellm_params": {
                "model": "ollama/qwythos",
                "api_key": "fake-key",
            },
            "model_info": {"id": "local", "type": "local"},
        },
        {
            "model_name": "agent-router",
            "litellm_params": {
                "model": "ollama/mythos",
                "api_key": "fake-key",
            },
            "model_info": {"id": "deep", "type": "deep"},
        },
    ]


def _create_test_router():
    return Router(model_list=_model_list())


@pytest.mark.asyncio
async def test_low_confidence_routes_to_cloud_smart():
    router = _create_test_router()
    strategy = LAR1RoutingStrategy(router)

    result = await strategy.async_get_available_deployment(
        model="agent-router",
        request_kwargs={"metadata": {"lar1": {"confidence": 0.2}}},
    )
    assert result["model_info"]["type"] == "cloud-smart"


@pytest.mark.asyncio
async def test_high_confidence_routes_to_deep():
    router = _create_test_router()
    strategy = LAR1RoutingStrategy(router)

    result = await strategy.async_get_available_deployment(
        model="agent-router",
        request_kwargs={"metadata": {"lar1": {"confidence": 0.8}}},
    )
    assert result["model_info"]["type"] == "deep"


@pytest.mark.asyncio
async def test_unverified_evidence_fallback():
    router = _create_test_router()
    strategy = LAR1RoutingStrategy(router)

    result = await strategy.async_get_available_deployment(
        model="agent-router",
        request_kwargs={
            "metadata": {
                "lar1": {"confidence": 0.9, "evidence": ["UNVERIFIED"]},
            }
        },
    )
    assert result["model_info"]["type"] == "cloud-smart"


@pytest.mark.asyncio
async def test_custom_thresholds():
    router = _create_test_router()
    custom_strategy = LAR1RoutingStrategy(
        router,
        thresholds={"low": 0.1, "medium": 0.3, "high": 0.9},
    )

    result = await custom_strategy.async_get_available_deployment(
        model="agent-router",
        request_kwargs={"metadata": {"lar1": {"confidence": 0.85}}},
    )
    assert result["model_info"]["type"] == "local"


@pytest.mark.asyncio
async def test_mem_time_routes_to_cloud_fast():
    router = _create_test_router()
    strategy = LAR1RoutingStrategy(router)

    result = await strategy.async_get_available_deployment(
        model="agent-router",
        request_kwargs={
            "metadata": {
                "lar1": {"confidence": 0.9, "time": "MEM"},
            }
        },
    )
    assert result["model_info"]["type"] == "cloud-fast"


@pytest.mark.asyncio
async def test_invalid_confidence_falls_back_to_local():
    router = _create_test_router()
    strategy = LAR1RoutingStrategy(router)

    result = await strategy.async_get_available_deployment(
        model="agent-router",
        request_kwargs={
            "metadata": {"lar1": {"confidence": "not-a-number"}},
        },
    )
    assert result["model_info"]["type"] == "local"


@pytest.mark.asyncio
async def test_no_metadata_defaults_to_local():
    router = _create_test_router()
    strategy = LAR1RoutingStrategy(router)

    result = await strategy.async_get_available_deployment(
        model="agent-router",
        request_kwargs={},
    )
    assert result["model_info"]["type"] == "local"


@pytest.mark.asyncio
async def test_router_init_with_lar1_routing_strategy():
    router = Router(
        model_list=_model_list(),
        routing_strategy="lar1",
        routing_strategy_args={
            "confidence_threshold_low": 0.1,
            "confidence_threshold_medium": 0.3,
            "confidence_threshold_high": 0.9,
        },
    )

    result = await router.async_get_available_deployment(
        model="agent-router",
        request_kwargs={"metadata": {"lar1": {"confidence": 0.85}}},
    )
    assert result["model_info"]["type"] == "local"
    assert router.routing_strategy == "lar1"


@pytest.mark.asyncio
async def test_router_init_lar1_default_thresholds():
    router = Router(model_list=_model_list(), routing_strategy="lar1")

    result = await router.async_get_available_deployment(
        model="agent-router",
        request_kwargs={"metadata": {"lar1": {"confidence": 0.4}}},
    )
    assert result["model_info"]["type"] == "cloud-fast"
    assert router.routing_strategy == "lar1"


@pytest.mark.asyncio
async def test_mid_confidence_routes_to_cloud_fast():
    router = _create_test_router()
    strategy = LAR1RoutingStrategy(router)

    result = await strategy.async_get_available_deployment(
        model="agent-router",
        request_kwargs={"metadata": {"lar1": {"confidence": 0.4}}},
    )
    assert result["model_info"]["type"] == "cloud-fast"


@pytest.mark.asyncio
async def test_no_router_returns_none():
    strategy = LAR1RoutingStrategy()

    result = await strategy.async_get_available_deployment(model="agent-router")
    assert result is None


@pytest.mark.asyncio
async def test_request_kwargs_none_uses_defaults():
    router = _create_test_router()
    strategy = LAR1RoutingStrategy(router)

    result = await strategy.async_get_available_deployment(
        model="agent-router",
        request_kwargs=None,
    )
    assert result["model_info"]["type"] == "local"


@pytest.mark.asyncio
async def test_invalid_lar1_metadata_type_uses_defaults(caplog):
    router = _create_test_router()
    strategy = LAR1RoutingStrategy(router)

    with caplog.at_level("WARNING"):
        result = await strategy.async_get_available_deployment(
            model="agent-router",
            request_kwargs={"metadata": {"lar1": "not-a-dict"}},
        )

    assert result["model_info"]["type"] == "local"
    assert "Invalid lar1 metadata type" in caplog.text


@pytest.mark.asyncio
async def test_confirmed_evidence_routes_by_confidence():
    router = _create_test_router()
    strategy = LAR1RoutingStrategy(router)

    result = await strategy.async_get_available_deployment(
        model="agent-router",
        request_kwargs={
            "metadata": {
                "lar1": {"confidence": 0.8, "evidence": ["CONFIRMED"]},
            }
        },
    )
    assert result["model_info"]["type"] == "deep"


@pytest.mark.asyncio
async def test_empty_healthy_deployments_returns_none():
    router = _create_test_router()
    strategy = LAR1RoutingStrategy(router)
    router.async_get_healthy_deployments = AsyncMock(return_value=[])

    result = await strategy.async_get_available_deployment(
        model="agent-router",
        request_kwargs={"metadata": {"lar1": {"confidence": 0.8}}},
    )
    assert result is None


@pytest.mark.asyncio
async def test_specific_deployment_dict_short_circuit():
    deployment = {
        "model_info": {"type": "deep"},
        "litellm_params": {"model": "ollama/mythos"},
    }
    router = _create_test_router()
    strategy = LAR1RoutingStrategy(router)
    router.async_get_healthy_deployments = AsyncMock(return_value=deployment)

    result = await strategy.async_get_available_deployment(
        model="agent-router",
        specific_deployment=True,
        request_kwargs={},
    )
    assert result == deployment


@pytest.mark.asyncio
async def test_non_dict_healthy_deployment_returns_none():
    router = _create_test_router()
    strategy = LAR1RoutingStrategy(router)
    router.async_get_healthy_deployments = AsyncMock(return_value=[None])

    result = await strategy.async_get_available_deployment(
        model="agent-router",
        request_kwargs={},
    )
    assert result is None


def test_parse_lar1_metadata_defaults_when_missing():
    metadata = _parse_lar1_metadata({})
    assert metadata.confidence == 0.5
    assert metadata.time.value == "NOW"


def test_select_deployment_empty_list():
    strategy = LAR1RoutingStrategy()
    selected, exact_match = strategy._select_deployment("local", [])
    assert selected is None
    assert exact_match is False


def test_select_deployment_skips_non_dict_entries():
    strategy = LAR1RoutingStrategy()
    deployment = {"model_info": {"type": "local"}}
    selected, exact_match = strategy._select_deployment(
        "local",
        ["skip-me", deployment],
    )
    assert selected == deployment
    assert exact_match is True


def test_select_deployment_fallback_uses_first_dict():
    strategy = LAR1RoutingStrategy()
    deployment = {"model_info": {"type": "local"}}
    selected, exact_match = strategy._select_deployment(
        "cloud-smart",
        ["skip-me", deployment],
    )
    assert selected == deployment
    assert exact_match is False


def test_select_deployment_all_non_dict_returns_none():
    strategy = LAR1RoutingStrategy()
    selected, exact_match = strategy._select_deployment("local", ["a", None])
    assert selected is None
    assert exact_match is False


def test_lar1_thresholds_from_args_uses_defaults():
    assert lar1_thresholds_from_args({}) == DEFAULT_THRESHOLDS


def test_lar1_thresholds_from_args_ignores_invalid_values():
    assert lar1_thresholds_from_args(
        {
            "confidence_threshold_low": "bad",
            "confidence_threshold_medium": 0.4,
            "confidence_threshold_high": 0.8,
        }
    ) == {"low": 0.3, "medium": 0.4, "high": 0.8}


@pytest.mark.asyncio
async def test_update_settings_switches_to_lar1_routing():
    router = Router(model_list=_model_list(), routing_strategy="simple-shuffle")
    router.update_settings(
        routing_strategy="lar1",
        routing_strategy_args={
            "confidence_threshold_low": 0.1,
            "confidence_threshold_medium": 0.3,
            "confidence_threshold_high": 0.9,
        },
    )

    result = await router.async_get_available_deployment(
        model="agent-router",
        request_kwargs={"metadata": {"lar1": {"confidence": 0.85}}},
    )
    assert router.routing_strategy == "lar1"
    assert result["model_info"]["type"] == "local"


def test_update_settings_switching_from_lar1_restores_default_selectors():
    router = Router(model_list=_model_list(), routing_strategy="lar1")

    router.update_settings(routing_strategy="simple-shuffle")

    assert router.routing_strategy == "simple-shuffle"
    assert "get_available_deployment" not in router.__dict__
    assert "async_get_available_deployment" not in router.__dict__
    result = router.get_available_deployment(model="agent-router")
    assert result["model_name"] == "agent-router"


@pytest.mark.asyncio
async def test_update_settings_routing_strategy_args_relinks_lar1():
    router = Router(model_list=_model_list(), routing_strategy="lar1")

    router.update_settings(
        routing_strategy_args={
            "confidence_threshold_low": 0.1,
            "confidence_threshold_medium": 0.3,
            "confidence_threshold_high": 0.9,
        },
    )

    result = await router.async_get_available_deployment(
        model="agent-router",
        request_kwargs={"metadata": {"lar1": {"confidence": 0.85}}},
    )
    assert result["model_info"]["type"] == "local"


def test_apply_lar1_routing_strategy_wires_custom_selector():
    router = Router(model_list=_model_list(), routing_strategy="simple-shuffle")
    apply_lar1_routing_strategy(router, {"confidence_threshold_high": 0.9})
    assert router.routing_strategy == "lar1"
    with pytest.raises(NotImplementedError, match="async routing"):
        router.get_available_deployment(model="agent-router")


def test_apply_lar1_invalid_thresholds_leaves_router_unchanged():
    router = Router(model_list=_model_list(), routing_strategy="simple-shuffle")

    with pytest.raises(ValueError, match="LAR-1 thresholds must satisfy"):
        apply_lar1_routing_strategy(
            router,
            {
                "confidence_threshold_low": 0.9,
                "confidence_threshold_medium": 0.5,
                "confidence_threshold_high": 0.7,
            },
        )

    assert router.routing_strategy == "simple-shuffle"
    assert "async_get_available_deployment" not in router.__dict__
    result = router.get_available_deployment(model="agent-router")
    assert result["model_name"] == "agent-router"


def test_invalid_threshold_order_raises():
    with pytest.raises(ValueError, match="LAR-1 thresholds must satisfy"):
        _normalize_thresholds({"low": 0.5, "medium": 0.3, "high": 0.7})


def test_get_available_deployment_raises_not_implemented():
    strategy = LAR1RoutingStrategy()
    with pytest.raises(NotImplementedError, match="async routing"):
        strategy.get_available_deployment(model="agent-router")


@pytest.mark.asyncio
async def test_missing_target_type_falls_back_with_warning(caplog):
    router = Router(
        model_list=[
            {
                "model_name": "agent-router",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_key": "fake-key",
                },
                "model_info": {"id": "only-local", "type": "local"},
            }
        ]
    )
    strategy = LAR1RoutingStrategy(router)

    with caplog.at_level("WARNING"):
        result = await strategy.async_get_available_deployment(
            model="agent-router",
            request_kwargs={"metadata": {"lar1": {"confidence": 0.2}}},
        )

    assert result["model_info"]["type"] == "local"
    assert "No deployment for type 'cloud-smart'" in caplog.text
