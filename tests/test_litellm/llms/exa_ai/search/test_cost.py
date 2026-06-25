from unittest.mock import Mock

from litellm.cost_calculator import cost_per_token
from litellm.llms.base_llm.search.transformation import SearchResponse
from litellm.llms.exa_ai.search.transformation import ExaAISearchConfig


def _resp(payload):
    r = Mock()
    r.json.return_value = payload
    return r


def _transform(payload):
    return ExaAISearchConfig().transform_search_response(
        _resp(payload), logging_obj=Mock()
    )


def test_transform_stashes_provider_reported_cost():
    """Exa's costDollars.total is surfaced for cost tracking via _hidden_params."""
    resp = _transform(
        {
            "results": [{"title": "T", "url": "https://e.com", "text": "x"}],
            "costDollars": {"total": 0.012},
        }
    )
    assert resp._hidden_params.get("provider_reported_cost") == 0.012


def test_transform_no_cost_when_costdollars_absent():
    resp = _transform({"results": [{"title": "T", "url": "https://e.com", "text": "x"}]})
    assert "provider_reported_cost" not in resp._hidden_params


def test_transform_no_cost_when_costdollars_malformed():
    resp = _transform(
        {
            "results": [{"title": "T", "url": "https://e.com", "text": "x"}],
            "costDollars": "oops",
        }
    )
    assert "provider_reported_cost" not in resp._hidden_params


def _search_response_with_cost(cost=None):
    resp = SearchResponse(results=[], object="search")
    if cost is not None:
        resp._hidden_params["provider_reported_cost"] = cost
    return resp


def test_cost_per_token_prefers_provider_reported_cost():
    input_cost, output_cost = cost_per_token(
        model="exa_ai/search",
        custom_llm_provider="exa_ai",
        call_type="search",
        number_of_queries=1,
        response=_search_response_with_cost(0.012),
    )
    assert input_cost == 0.012
    assert output_cost == 0.0


def test_cost_per_token_falls_back_to_price_list_when_no_provider_cost():
    # No provider-reported cost -> fall back to the existing price-list calc
    # (must not crash, must not invent the provider value).
    input_cost, output_cost = cost_per_token(
        model="exa_ai/search",
        custom_llm_provider="exa_ai",
        call_type="search",
        number_of_queries=1,
        response=_search_response_with_cost(),
    )
    assert isinstance(input_cost, float)
    assert output_cost == 0.0
