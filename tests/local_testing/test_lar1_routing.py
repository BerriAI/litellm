import pytest
from litellm import Router
from litellm.router_strategy.lar1_routing import LAR1RoutingStrategy


def _model_list():
    return [
        {
            "model_name": "gpt-4",
            "litellm_params": {
                "model": "openai/gpt-4o",
                "api_key": "fake-key",
            },
            "model_info": {"id": "cloud-smart", "type": "cloud-smart"},
        },
        {
            "model_name": "gpt-4-mini",
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "api_key": "fake-key",
            },
            "model_info": {"id": "cloud-fast", "type": "cloud-fast"},
        },
        {
            "model_name": "local-qwythos",
            "litellm_params": {
                "model": "ollama/qwythos",
                "api_key": "fake-key",
            },
            "model_info": {"id": "local", "type": "local"},
        },
        {
            "model_name": "local-mythos",
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
        model="gpt-4",
        request_kwargs={"metadata": {"lar1": {"confidence": 0.2}}},
    )
    assert result["model_info"]["type"] == "cloud-smart"


@pytest.mark.asyncio
async def test_high_confidence_routes_to_deep():
    router = _create_test_router()
    strategy = LAR1RoutingStrategy(router)

    result = await strategy.async_get_available_deployment(
        model="gpt-4",
        request_kwargs={"metadata": {"lar1": {"confidence": 0.8}}},
    )
    assert result["model_info"]["type"] == "deep"


@pytest.mark.asyncio
async def test_unverified_evidence_fallback():
    router = _create_test_router()
    strategy = LAR1RoutingStrategy(router)

    result = await strategy.async_get_available_deployment(
        model="gpt-4",
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
        model="gpt-4",
        request_kwargs={"metadata": {"lar1": {"confidence": 0.85}}},
    )
    assert result["model_info"]["type"] == "local"


@pytest.mark.asyncio
async def test_mem_time_routes_to_cloud_fast():
    router = _create_test_router()
    strategy = LAR1RoutingStrategy(router)

    result = await strategy.async_get_available_deployment(
        model="gpt-4",
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
        model="gpt-4",
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
        model="gpt-4",
        request_kwargs={},
    )
    assert result["model_info"]["type"] == "local"


@pytest.mark.asyncio
async def test_router_init_with_lar1_routing_strategy():
    router = Router(
        model_list=_model_list(),
        routing_strategy="lar1",
        lar1_settings={
            "confidence_threshold_low": 0.1,
            "confidence_threshold_medium": 0.3,
            "confidence_threshold_high": 0.9,
        },
    )

    result = await router.async_get_available_deployment(
        model="gpt-4",
        request_kwargs={"metadata": {"lar1": {"confidence": 0.85}}},
    )
    assert result["model_info"]["type"] == "local"
    assert router.routing_strategy == "lar1"
