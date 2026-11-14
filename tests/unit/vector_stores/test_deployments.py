from collections.abc import Iterator, Mapping
from typing import Final

import aiohttp
import httpx
import pytest

import litellm
from litellm.proxy._types import ProxyException
from litellm.vector_stores.deployments import (
    Failed,
    Served,
    VectorStoreDeployment,
    classify_error,
    parse_vector_store_deployments,
    primary_connection,
    vector_store_deployment_cooldowns,
    vector_store_deployment_problems,
    vector_store_deployment_registry,
    with_failover,
)

_EASTUS: Final = {
    "deployment_name": "kb-search",
    "deployment_id": "eastus",
    "litellm_params": {
        "custom_llm_provider": "azure_ai",
        "api_base": "https://eastus.search.windows.net",
        "api_key": "key-eastus",
        "order": 1,
        "cooldown_time": 30,
    },
}
_SCUS: Final = {
    "deployment_name": "kb-search",
    "deployment_id": "scus",
    "litellm_params": {
        "custom_llm_provider": "azure_ai",
        "api_base": "https://scus.search.windows.net",
        "api_key": "key-scus",
        "order": 2,
    },
}


@pytest.fixture(autouse=True)
def kb_search_group() -> Iterator[None]:
    vector_store_deployment_registry.load(
        parse_vector_store_deployments({"vector_store_deployments": [_SCUS, _EASTUS]})
    )
    vector_store_deployment_cooldowns.in_memory_cache.flush_cache()
    yield
    vector_store_deployment_registry.load(())
    vector_store_deployment_cooldowns.in_memory_cache.flush_cache()


def _store(**params: object) -> Mapping[str, object]:
    return {
        "vector_store_name": "claims",
        "litellm_params": {
            "vector_store_id": "claims-docs",
            "custom_llm_provider": "azure_ai",
            "vector_store_deployment": "kb-search",
            **params,
        },
    }


def _scripted(outcomes: Mapping[str, str]):
    """An attempt that serves or fails per deployment, recording the order it was called in."""
    calls: list[str] = []

    async def attempt(deployment: VectorStoreDeployment):
        calls.append(deployment.deployment_id)
        match outcomes[deployment.deployment_id]:
            case "ok":
                return Served(deployment=deployment, value=f"from {deployment.deployment_id}", failed_over=0)
            case "503":
                return Failed(deployment=deployment, retryable=True, cools_down=True, error=RuntimeError("503"))
            case "404":
                return Failed(deployment=deployment, retryable=True, cools_down=False, error=RuntimeError("404"))
            case "400":
                return Failed(deployment=deployment, retryable=False, cools_down=False, error=RuntimeError("400"))
        raise AssertionError(outcomes[deployment.deployment_id])

    return attempt, calls


def test_parse_orders_by_order_and_keeps_routing_settings_out_of_the_connection():
    deployments = vector_store_deployment_registry.group("kb-search")

    assert [d.deployment_id for d in deployments] == ["eastus", "scus"]
    assert deployments[0].cooldown_time == 30
    assert "order" not in deployments[0].connection and "cooldown_time" not in deployments[0].connection


@pytest.mark.asyncio
async def test_a_healthy_primary_serves_and_the_secondary_is_untouched():
    attempt, calls = _scripted({"eastus": "ok", "scus": "ok"})

    outcome = await with_failover("kb-search", attempt)

    assert isinstance(outcome, Served) and outcome.value == "from eastus" and outcome.failed_over == 0
    assert calls == ["eastus"]


@pytest.mark.asyncio
async def test_a_retryable_failure_fails_over_in_the_same_read_and_cools_the_primary_down():
    attempt, calls = _scripted({"eastus": "503", "scus": "ok"})
    first = await with_failover("kb-search", attempt)

    attempt_again, calls_again = _scripted({"eastus": "ok", "scus": "ok"})
    second = await with_failover("kb-search", attempt_again)

    assert isinstance(first, Served) and first.value == "from scus" and first.failed_over == 1
    assert calls == ["eastus", "scus"]
    assert isinstance(second, Served) and second.value == "from scus"
    assert calls_again == ["scus"]


@pytest.mark.asyncio
async def test_a_missing_index_fails_over_without_cooling_the_service_down():
    attempt, _ = _scripted({"eastus": "404", "scus": "ok"})
    await with_failover("kb-search", attempt)

    attempt_again, calls_again = _scripted({"eastus": "ok", "scus": "ok"})
    await with_failover("kb-search", attempt_again)

    assert calls_again == ["eastus"]


@pytest.mark.asyncio
async def test_a_non_retryable_failure_is_returned_without_trying_the_secondary():
    attempt, calls = _scripted({"eastus": "400", "scus": "ok"})

    outcome = await with_failover("kb-search", attempt)

    assert isinstance(outcome, Failed) and str(outcome.error) == "400"
    assert calls == ["eastus"]


@pytest.mark.asyncio
async def test_when_every_deployment_fails_the_last_failure_comes_back_and_cooled_ones_are_still_tried():
    attempt, calls = _scripted({"eastus": "503", "scus": "503"})
    first = await with_failover("kb-search", attempt)
    attempt_again, calls_again = _scripted({"eastus": "ok", "scus": "ok"})
    second = await with_failover("kb-search", attempt_again)

    assert isinstance(first, Failed) and calls == ["eastus", "scus"]
    assert isinstance(second, Served) and calls_again == ["eastus"]


def test_a_valid_grouped_store_has_no_problems():
    config = {"vector_store_deployments": [_EASTUS, _SCUS], "vector_store_registry": [_store()]}

    assert vector_store_deployment_problems(config) == ()


@pytest.mark.parametrize(
    ("deployments", "store", "problem"),
    [
        ([_EASTUS], _store(vector_store_deployment="missing"), "'missing', which is not defined"),
        ([_EASTUS], _store(custom_llm_provider="milvus"), "has custom_llm_provider 'milvus'"),
        ([_EASTUS], _store(api_base="https://eastus.search.windows.net"), "['api_base']"),
        ([_EASTUS], _store(litellm_credential_name="cred"), "['litellm_credential_name']"),
        (
            [_EASTUS, {**_SCUS, "litellm_params": {**_SCUS["litellm_params"], "custom_llm_provider": "milvus"}}],
            _store(),
            "mixes providers ['azure_ai', 'milvus']",
        ),
        ([_EASTUS, {**_SCUS, "deployment_id": "eastus"}], _store(), "repeats deployment_id 'eastus'"),
    ],
)
def test_a_config_that_cannot_serve_is_reported(deployments, store, problem):
    config = {"vector_store_deployments": deployments, "vector_store_registry": [store]}

    assert any(problem in found for found in vector_store_deployment_problems(config))


def test_primary_connection_puts_a_grouped_store_on_its_lowest_order_deployment():
    params = primary_connection({"vector_store_id": "claims-docs", "vector_store_deployment": "kb-search"})

    assert dict(params) == {
        "vector_store_id": "claims-docs",
        "custom_llm_provider": "azure_ai",
        "api_base": "https://eastus.search.windows.net",
        "api_key": "key-eastus",
    }


_EASTUS_SEARCH: Final = "https://eastus.search.windows.net/indexes/claims-docs/docs/search"
_SCUS_SEARCH: Final = "https://scus.search.windows.net/indexes/claims-docs/docs/search"
_HIT: Final = {"value": [{"@search.score": 0.9, "id": "doc-1", "content": "served by scus"}]}


@pytest.fixture
def httpx_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)


async def _grouped_asearch(**request: object):
    return await litellm.vector_stores.asearch(
        vector_store_id="claims-docs",
        query="claims",
        custom_llm_provider="azure_ai",
        vector_store_deployment="kb-search",
        litellm_embedding_model="openai/text-embedding-3-small",
        litellm_embedding_config={"api_key": "sk-fake", "mock_response": [0.1, 0.2, 0.3]},
        **request,
    )


@pytest.mark.asyncio
async def test_asearch_on_a_group_fails_over_and_reports_the_serving_deployment(respx_mock, httpx_transport):
    eastus = respx_mock.post(url__startswith=_EASTUS_SEARCH).mock(return_value=httpx.Response(503, json={}))
    scus = respx_mock.post(url__startswith=_SCUS_SEARCH).mock(return_value=httpx.Response(200, json=_HIT))

    response = await _grouped_asearch()

    assert eastus.call_count == 1 and scus.call_count == 1
    assert scus.calls.last.request.headers["api-key"] == "key-scus"
    assert response["data"][0]["content"][0]["text"] == "served by scus"
    hidden = response["_hidden_params"]
    assert hidden["api_base"] == "https://scus.search.windows.net"
    assert hidden["additional_headers"]["x-litellm-vector-store-deployment"] == "scus"
    assert hidden["additional_headers"]["x-litellm-attempted-fallbacks"] == "1"


@pytest.mark.asyncio
async def test_asearch_on_a_group_keeps_the_deployments_connection_over_the_callers(respx_mock, httpx_transport):
    eastus = respx_mock.post(url__startswith=_EASTUS_SEARCH).mock(return_value=httpx.Response(200, json=_HIT))
    attacker = respx_mock.post(url__startswith="https://attacker.example").mock(return_value=httpx.Response(200))

    await _grouped_asearch(api_base="https://attacker.example", api_key="caller-key")

    assert not attacker.called
    assert eastus.calls.last.request.headers["api-key"] == "key-eastus"


@pytest.mark.asyncio
async def test_a_cooldown_entry_past_its_end_no_longer_holds_the_primary_back():
    import time

    vector_store_deployment_cooldowns.in_memory_cache.set_cache(
        "vector_store_deployment_cooldown:kb-search:eastus", time.time() - 1, ttl=600
    )
    attempt, calls = _scripted({"eastus": "ok", "scus": "ok"})

    await with_failover("kb-search", attempt)

    assert calls == ["eastus"]


@pytest.mark.asyncio
async def test_allowed_fails_holds_the_cooldown_until_that_many_failures():
    eastus_tolerant = {**_EASTUS, "litellm_params": {**_EASTUS["litellm_params"], "allowed_fails": 2}}
    vector_store_deployment_registry.load(
        parse_vector_store_deployments({"vector_store_deployments": [_SCUS, eastus_tolerant]})
    )
    orders: list[list[str]] = []
    for outcomes in ({"eastus": "503", "scus": "ok"}, {"eastus": "503", "scus": "ok"}, {"eastus": "ok", "scus": "ok"}):
        attempt, calls = _scripted(outcomes)
        await with_failover("kb-search", attempt)
        orders.append(calls)

    assert orders == [["eastus", "scus"], ["eastus", "scus"], ["scus"]]


@pytest.fixture
def counted_embeddings(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    queries: list[str] = []
    real_aembedding = litellm.aembedding

    async def aembedding(**kwargs: object):
        queries.append(str(kwargs["input"]))
        return await real_aembedding(**kwargs)

    monkeypatch.setattr(litellm, "aembedding", aembedding)
    return queries


@pytest.mark.asyncio
async def test_a_failover_reuses_the_query_embedding(respx_mock, httpx_transport, counted_embeddings):
    respx_mock.post(url__startswith=_EASTUS_SEARCH).mock(return_value=httpx.Response(503, json={}))
    scus = respx_mock.post(url__startswith=_SCUS_SEARCH).mock(return_value=httpx.Response(200, json=_HIT))

    await _grouped_asearch()

    assert scus.call_count == 1
    assert len(counted_embeddings) == 1


@pytest.mark.asyncio
async def test_a_failed_embedding_stops_the_search_without_cooling_a_deployment_down(
    respx_mock, httpx_transport, monkeypatch: pytest.MonkeyPatch
):
    async def aembedding(**kwargs: object):
        raise litellm.RateLimitError(message="embedding quota", llm_provider="openai", model="text-embedding-3-small")

    monkeypatch.setattr(litellm, "aembedding", aembedding)
    search = respx_mock.post(url__startswith="https://").mock(return_value=httpx.Response(200, json=_HIT))

    with pytest.raises(litellm.RateLimitError):
        await _grouped_asearch()

    assert not search.called
    assert vector_store_deployment_cooldowns.in_memory_cache.cache_dict == {}


def test_an_unreachable_host_behind_a_passthrough_error_fails_over_and_cools_down():
    deployment = vector_store_deployment_registry.group("kb-search")[0]
    try:
        try:
            raise aiohttp.ClientConnectionError("Cannot connect to host")
        except aiohttp.ClientConnectionError:
            raise ProxyException(message="Cannot connect to host", type="internal_server_error", param=None, code=500)
    except ProxyException as passthrough_error:
        failure = classify_error(deployment, passthrough_error)

    assert failure.retryable and failure.cools_down


class _UnreachableRedis:
    async def async_batch_get_cache(self, key_list: list[str]) -> list[object]:
        raise ConnectionError("redis circuit open")


@pytest.mark.asyncio
async def test_a_redis_outage_falls_back_to_this_workers_cooldowns(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(vector_store_deployment_cooldowns, "redis_cache", _UnreachableRedis())
    attempt, calls = _scripted({"eastus": "ok", "scus": "ok"})

    outcome = await with_failover("kb-search", attempt)

    assert isinstance(outcome, Served) and outcome.value == "from eastus"
    assert calls == ["eastus"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("cooldown_time", "expiries"), [(0.5, [1]), (0, [])])
async def test_a_cooldown_expiry_is_whole_seconds_and_a_zero_cooldown_records_none(
    monkeypatch: pytest.MonkeyPatch, cooldown_time: float, expiries: list[int]
):
    eastus = {**_EASTUS, "litellm_params": {**_EASTUS["litellm_params"], "cooldown_time": cooldown_time}}
    vector_store_deployment_registry.load(parse_vector_store_deployments({"vector_store_deployments": [_SCUS, eastus]}))
    recorded: list[object] = []
    real_set = vector_store_deployment_cooldowns.async_set_cache

    async def set_cache(key: str, value: object, **kwargs: object) -> None:
        recorded.append(kwargs.get("ttl"))
        await real_set(key, value, **kwargs)

    monkeypatch.setattr(vector_store_deployment_cooldowns, "async_set_cache", set_cache)
    attempt, _ = _scripted({"eastus": "503", "scus": "ok"})

    await with_failover("kb-search", attempt)

    assert recorded == expiries


def test_sync_search_on_a_group_fails_over_like_asearch(respx_mock, httpx_transport):
    eastus = respx_mock.post(url__startswith=_EASTUS_SEARCH).mock(return_value=httpx.Response(503, json={}))
    scus = respx_mock.post(url__startswith=_SCUS_SEARCH).mock(return_value=httpx.Response(200, json=_HIT))

    response = litellm.vector_stores.search(
        vector_store_id="claims-docs",
        query="claims",
        custom_llm_provider="azure_ai",
        vector_store_deployment="kb-search",
        litellm_embedding_model="openai/text-embedding-3-small",
        litellm_embedding_config={"api_key": "sk-fake", "mock_response": [0.1, 0.2, 0.3]},
    )

    assert eastus.call_count == 1 and scus.call_count == 1
    assert response["data"][0]["content"][0]["text"] == "served by scus"
    assert response["_hidden_params"]["additional_headers"]["x-litellm-vector-store-deployment"] == "scus"
