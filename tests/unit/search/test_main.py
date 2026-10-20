import datetime
from typing import Final

import pytest
import respx

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.search.cost_calculator import search_provider_cost_per_query

EXA_SEARCH_URL: Final = "https://api.exa.ai/search"


@pytest.fixture
def _local_model_cost_map(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap

    monkeypatch.setattr(litellm, "model_cost", GetModelCostMap.load_local_model_cost_map())


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
