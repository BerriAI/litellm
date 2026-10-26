import datetime
import json
from collections.abc import Mapping
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.search.cost_calculator import search_provider_cost_per_query

EXA_SEARCH_URL: Final = "https://api.exa.ai/search"
TAVILY_SEARCH_URL: Final = "https://api.tavily.com/search"
LINKUP_SEARCH_URL: Final = "https://api.linkup.so/v1/search"
SEARCH_MODE_PRICES: Final = {
    "tavily/search": 0.002,
    "tavily/search-advanced": 0.007,
    "tavily/search-fast": 0.001,
    "linkup/search": 0.003,
    "linkup/search-deep": 0.011,
    "perplexity/search": 0.004,
    "perplexity/search-advanced": 0.009,
    "tavily/negotiated-search": 0.004,
}


@pytest.fixture
def _local_model_cost_map(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "model_cost", GetModelCostMap.load_local_model_cost_map())


@pytest.fixture
def _search_mode_prices(_local_model_cost_map: None, monkeypatch: pytest.MonkeyPatch) -> None:
    for model, price in SEARCH_MODE_PRICES.items():
        monkeypatch.setitem(
            litellm.model_cost,
            model,
            {"input_cost_per_query": price, "litellm_provider": model.split("/", 1)[0], "mode": "search"},
        )


def _proxy_style_logging_obj() -> Logging:
    """The proxy builds the logging object from the request body before the router adds the search tool's
    litellm_params, so the per-tool price only reaches cost tracking if the search call itself records it."""
    return Logging(
        model="exa-search",
        messages=[],
        stream=False,
        call_type="search",
        start_time=datetime.datetime.now(),
        litellm_call_id="search-tool-price",
        function_id="f",
    )


@pytest.mark.usefixtures("_local_model_cost_map")
def test_search_bills_search_tool_input_cost_per_query(respx_mock: respx.MockRouter) -> None:
    respx_mock.post(EXA_SEARCH_URL).respond(json={"results": [{"title": "t", "url": "https://example.com"}]})
    map_price, _ = search_provider_cost_per_query(model="exa_ai/search", custom_llm_provider="exa_ai")
    configured_price: Final = 0.05
    assert map_price not in (0.0, configured_price)

    map_priced: Final = litellm.search(
        query="litellm release notes",
        search_provider="exa_ai",
        api_key="test-exa-key",
        litellm_logging_obj=_proxy_style_logging_obj(),
    )
    tool_priced: Final = litellm.search(
        query="litellm release notes",
        search_provider="exa_ai",
        api_key="test-exa-key",
        input_cost_per_query=configured_price,
        litellm_logging_obj=_proxy_style_logging_obj(),
    )

    assert map_priced._hidden_params["response_cost"] == pytest.approx(map_price)
    assert tool_priced._hidden_params["response_cost"] == pytest.approx(configured_price)
    assert respx_mock.calls.last.request.content.find(b"input_cost_per_query") == -1


@pytest.mark.usefixtures("_local_model_cost_map")
def test_unpriced_fallback_search_tool_is_billed_at_its_own_price_not_the_failed_primarys(
    respx_mock: respx.MockRouter,
) -> None:
    respx_mock.post(EXA_SEARCH_URL).mock(
        side_effect=(
            httpx.Response(500, json={"error": "primary down"}),
            httpx.Response(200, json={"results": [{"title": "t", "url": "https://example.com"}]}),
        )
    )
    map_price, _ = search_provider_cost_per_query(model="exa_ai/search", custom_llm_provider="exa_ai")
    primary_price: Final = 0.05
    assert map_price != primary_price
    shared_by_every_fallback_attempt: Final = _proxy_style_logging_obj()

    with pytest.raises(litellm.InternalServerError):
        litellm.search(
            query="litellm release notes",
            search_provider="exa_ai",
            api_key="test-exa-key",
            input_cost_per_query=primary_price,
            litellm_logging_obj=shared_by_every_fallback_attempt,
        )
    fallback: Final = litellm.search(
        query="litellm release notes",
        search_provider="exa_ai",
        api_key="test-exa-key",
        litellm_logging_obj=shared_by_every_fallback_attempt,
    )

    assert fallback._hidden_params["response_cost"] == pytest.approx(map_price)


@pytest.mark.usefixtures("_search_mode_prices")
@pytest.mark.parametrize(
    ("provider", "optional_params", "expected"),
    [
        ("tavily", {"search_depth": "advanced"}, 0.007),
        ("tavily", {"search_depth": "fast"}, 0.001),
        ("tavily", {"search_depth": "basic"}, 0.002),
        ("tavily", {"search_depth": "ultra-fast"}, 0.002),
        ("tavily", {}, 0.002),
        ("tavily", None, 0.002),
        ("tavily", {"search_depth": ["advanced"]}, 0.002),
        ("tavily", {"depth": "advanced"}, 0.002),
        ("linkup", {"depth": "deep"}, 0.011),
        ("linkup", {"depth": "standard"}, 0.003),
        ("linkup", {}, 0.003),
        ("linkup", {"search_depth": "deep"}, 0.003),
        ("perplexity", {"search_depth": "advanced", "depth": "advanced"}, 0.004),
    ],
    ids=[
        "tavily-advanced",
        "tavily-fast",
        "tavily-basic-has-no-entry",
        "tavily-mode-without-entry",
        "tavily-no-mode",
        "tavily-no-params",
        "tavily-non-string-mode",
        "tavily-ignores-linkup-param",
        "linkup-deep",
        "linkup-standard-has-no-entry",
        "linkup-no-mode",
        "linkup-ignores-tavily-param",
        "provider-without-mode-param-keeps-base-entry",
    ],
)
def test_search_cost_is_the_price_entry_of_the_requested_mode(
    provider: str, optional_params: Mapping[str, object] | None, expected: float
) -> None:
    cost: Final = search_provider_cost_per_query(
        model=f"{provider}/search", custom_llm_provider=provider, optional_params=optional_params
    )
    assert cost == pytest.approx((expected, 0.0))


@pytest.mark.usefixtures("_search_mode_prices")
def test_a_registered_search_model_keeps_its_own_price_for_an_advanced_search() -> None:
    cost: Final = search_provider_cost_per_query(
        model="tavily/negotiated-search", custom_llm_provider="tavily", optional_params={"search_depth": "advanced"}
    )
    assert cost == pytest.approx((0.004, 0.0))


@pytest.fixture
def _loaded_catalog_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        litellm,
        "model_cost",
        {key: dict(entry) for key, entry in GetModelCostMap.loaded_model_cost_map().items()},
    )


MODE_CASES: Final = (
    pytest.param("tavily", "tavily/search", {"search_depth": "advanced"}, id="tavily"),
    pytest.param("tavily", "tavily/tavily/search", {"search_depth": "advanced"}, id="tavily-sdk-spelling"),
    pytest.param("linkup", "linkup/search", {"depth": "deep"}, id="linkup"),
)


def _register(monkeypatch: pytest.MonkeyPatch, entry: str, price: float) -> None:
    monkeypatch.setitem(
        litellm.model_cost,
        entry,
        {"input_cost_per_query": price, "litellm_provider": entry.split("/", 1)[0], "mode": "search"},
    )


def _mode_entry(provider: str, optional_params: Mapping[str, object]) -> str:
    return f"{provider}/search-{next(iter(optional_params.values()))}"


@pytest.mark.usefixtures("_loaded_catalog_prices")
@pytest.mark.parametrize(("provider", "model", "optional_params"), MODE_CASES)
def test_the_catalog_mode_price_applies_when_nothing_is_registered_at_runtime(
    provider: str, model: str, optional_params: Mapping[str, object]
) -> None:
    catalog: Final = GetModelCostMap.loaded_model_cost_map()
    mode_price: Final = catalog[_mode_entry(provider, optional_params)]["input_cost_per_query"]
    assert mode_price != catalog[f"{provider}/search"]["input_cost_per_query"]
    cost: Final = search_provider_cost_per_query(
        model=model, custom_llm_provider=provider, optional_params=optional_params
    )
    assert cost == pytest.approx((mode_price, 0.0))


@pytest.mark.usefixtures("_loaded_catalog_prices")
@pytest.mark.parametrize(
    ("provider", "optional_params"),
    [
        ("tavily", {"search_depth": "ultra-fast"}),
        ("tavily", {"search_depth": 7}),
        ("tavily", {"search_depth": ["advanced"]}),
        ("linkup", {"depth": "standard"}),
        ("linkup", {"depth": "DEEP"}),
    ],
    ids=["tavily-ultra-fast", "tavily-int", "tavily-list", "linkup-standard", "linkup-uppercase"],
)
def test_a_mode_without_its_own_catalog_entry_gets_the_catalog_base_price(
    provider: str, optional_params: Mapping[str, object]
) -> None:
    cost: Final = search_provider_cost_per_query(
        model=f"{provider}/search", custom_llm_provider=provider, optional_params=optional_params
    )
    base_price: Final = GetModelCostMap.loaded_model_cost_map()[f"{provider}/search"]["input_cost_per_query"]
    assert cost == pytest.approx((base_price, 0.0))


@pytest.mark.usefixtures("_loaded_catalog_prices")
@pytest.mark.parametrize(("provider", "model", "optional_params"), MODE_CASES)
def test_a_runtime_price_on_the_generic_entry_keeps_applying_to_a_mode_without_a_runtime_price(
    monkeypatch: pytest.MonkeyPatch, provider: str, model: str, optional_params: Mapping[str, object]
) -> None:
    _register(monkeypatch, f"{provider}/search", 0.004)
    cost: Final = search_provider_cost_per_query(
        model=model, custom_llm_provider=provider, optional_params=optional_params
    )
    assert cost == pytest.approx((0.004, 0.0))


@pytest.mark.usefixtures("_loaded_catalog_prices")
@pytest.mark.parametrize(("provider", "model", "optional_params"), MODE_CASES)
def test_a_runtime_price_on_the_mode_entry_wins_over_a_runtime_price_on_the_generic_entry(
    monkeypatch: pytest.MonkeyPatch, provider: str, model: str, optional_params: Mapping[str, object]
) -> None:
    _register(monkeypatch, f"{provider}/search", 0.004)
    _register(monkeypatch, _mode_entry(provider, optional_params), 0.011)
    cost: Final = search_provider_cost_per_query(
        model=model, custom_llm_provider=provider, optional_params=optional_params
    )
    assert cost == pytest.approx((0.011, 0.0))


@pytest.mark.usefixtures("_search_mode_prices")
def test_sync_tavily_advanced_search_reports_the_advanced_price(respx_mock: respx.MockRouter) -> None:
    respx_mock.post(TAVILY_SEARCH_URL).respond(json={"results": [{"title": "t", "url": "https://example.com"}]})
    response: Final = litellm.search(
        query="litellm release notes", search_provider="tavily", api_key="test-tavily-key", search_depth="advanced"
    )
    assert json.loads(respx_mock.calls.last.request.content)["search_depth"] == "advanced"
    assert response.hidden_params["response_cost"] == pytest.approx(0.007)


@pytest.mark.asyncio
@pytest.mark.usefixtures("_search_mode_prices")
async def test_async_linkup_deep_search_reports_the_deep_price(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post(LINKUP_SEARCH_URL).respond(
        json={"results": [{"type": "text", "name": "t", "url": "https://example.com", "content": "c"}]}
    )
    response: Final = await litellm.asearch(
        query="litellm release notes", search_provider="linkup", api_key="test-linkup-key", depth="deep"
    )
    assert json.loads(respx_mock.calls.last.request.content)["depth"] == "deep"
    assert response.hidden_params["response_cost"] == pytest.approx(0.011)
