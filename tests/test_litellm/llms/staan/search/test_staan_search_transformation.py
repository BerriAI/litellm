"""Tests for the Staan search provider transformation."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest

import litellm
from litellm.llms.staan.search.transformation import StaanSearchConfig
from litellm.router_utils.search_api_router import SearchAPIRouter
from litellm.types.utils import SearchProviders
from litellm.utils import ProviderConfigManager


def test_staan_is_registered_as_a_search_provider() -> None:
    config = ProviderConfigManager.get_provider_search_config(SearchProviders.STAAN)

    assert isinstance(config, StaanSearchConfig)


def test_staan_has_ui_friendly_provider_name() -> None:
    assert StaanSearchConfig.ui_friendly_name() == "Staan"


def test_staan_environment_validation_returns_mutable_request_headers() -> None:
    headers = StaanSearchConfig().validate_environment(
        headers={"X-Request-ID": "request-id"},
        api_key="test-key",
    )

    assert type(headers) is dict
    assert headers == {
        "X-Request-ID": "request-id",
        "Accept": "application/json",
        "Authorization": "Bearer test-key",
        "Content-Type": "application/json",
    }


def test_staan_api_defaults_resolve_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAAN_API_KEY", "environment-key")
    monkeypatch.setenv("STAAN_API_BASE", "https://search.example/api")
    config = StaanSearchConfig()

    assert config.validate_environment(headers={})["Authorization"] == "Bearer environment-key"
    assert config.get_complete_url(None, {}) == "https://search.example/api"
    assert config.transform_search_request("query", {}) == {"q": "query"}


def test_staan_market_is_not_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAAN_MARKET", "de-de")

    request = StaanSearchConfig().transform_search_request("query", {})

    assert request == {"q": "query"}


def test_staan_explicit_api_base_takes_precedence_over_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAAN_API_BASE", "https://search.example/api")

    url = StaanSearchConfig().get_complete_url("https://caller.example/search", {})

    assert url == "https://caller.example/search"


def test_staan_environment_validation_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STAAN_API_KEY", raising=False)

    with pytest.raises(ValueError, match="STAAN_API_KEY is not set"):
        StaanSearchConfig().validate_environment(headers={})


def test_staan_request_maps_enrichment_and_domain_parameters() -> None:
    config = StaanSearchConfig()

    request = config.transform_search_request(
        query="vector search",
        optional_params={
            "market": "en-us",
            "offset": 20,
            "search_domain_filter": ["example.com"],
            "extra_snippets": True,
            "max_snippets": 5,
            "min_score": 0.2,
            "full_content": "markdown",
            "max_results": 3,
        },
    )

    assert config.get_http_method() == "POST"
    assert config.get_complete_url(None, {}, request) == config.STAAN_API_BASE
    assert request == {
        "q": "vector search",
        "market": "en-us",
        "offset": 20,
        "include_domains": ["example.com"],
        "extra_snippets": True,
        "max_snippets": 5,
        "min_score": 0.2,
        "full_content": "markdown",
    }


@pytest.mark.parametrize(
    "optional_params",
    [
        {"include_domains": [], "exclude_domains": ["blocked.example"]},
        {"search_domain_filter": [], "exclude_domains": ["blocked.example"]},
        {
            "include_domains": [],
            "search_domain_filter": [],
            "exclude_domains": ["blocked.example"],
        },
    ],
)
def test_staan_empty_include_filters_allow_exclude_domains(optional_params: dict[str, object]) -> None:
    request = StaanSearchConfig().transform_search_request("query", optional_params)

    assert request == {"q": "query", "exclude_domains": ["blocked.example"]}


def test_staan_max_snippets_enables_enrichment_and_explicit_false_disables_it() -> None:
    config = StaanSearchConfig()

    enabled = config.transform_search_request("query", {"max_snippets": 5})
    disabled = config.transform_search_request("query", {"extra_snippets": False, "max_snippets": 5})

    assert enabled["extra_snippets"] is True
    assert enabled["max_snippets"] == 5
    assert "extra_snippets" not in disabled
    assert "max_snippets" not in disabled


def test_staan_maps_litellm_country_to_staan_market() -> None:
    request = StaanSearchConfig().transform_search_request("query", {"country": "DE"})

    assert request["market"] == "de-de"


def test_staan_empty_country_uses_explicit_market() -> None:
    request = StaanSearchConfig().transform_search_request("query", {"country": "", "market": "en-us"})

    assert request["market"] == "en-us"


def test_staan_country_and_explicit_market_use_explicit_market() -> None:
    request = StaanSearchConfig().transform_search_request("query", {"country": "DE", "market": "en-us"})

    assert request["market"] == "en-us"


def test_staan_joins_multi_part_query_and_keeps_count_fixed() -> None:
    request = StaanSearchConfig().transform_search_request(
        query=("vector", "search"),
        optional_params={"count": 10},
    )

    assert request == {"q": "vector search"}


def test_staan_rejects_queries_over_provider_limit() -> None:
    with pytest.raises(ValueError, match="query must be at most 400 characters"):
        StaanSearchConfig().transform_search_request("q" * 401, {})


@pytest.mark.parametrize("value", ["true", "1", " YES "])
def test_staan_accepts_true_string_values_for_extra_snippets(value: str) -> None:
    request = StaanSearchConfig().transform_search_request("query", {"extra_snippets": value})

    assert request["extra_snippets"] is True


@pytest.mark.parametrize("value", ["false", "0", " NO "])
def test_staan_accepts_false_string_values_for_extra_snippets(value: str) -> None:
    request = StaanSearchConfig().transform_search_request("query", {"extra_snippets": value, "max_snippets": 5})

    assert "extra_snippets" not in request
    assert "max_snippets" not in request


@pytest.mark.parametrize(
    ("optional_params", "message"),
    [
        ({"offset": 5}, "offset"),
        ({"market": "es-es"}, "market"),
        ({"country": "GB"}, "country"),
        ({"count": 9}, "count"),
        ({"max_results": 0}, "max_results"),
        ({"max_results": -1}, "max_results"),
        ({"max_results": 1.5}, "max_results"),
        ({"max_results": True}, "max_results"),
        ({"extra_snippets": True, "max_snippets": 11}, "max_snippets"),
        ({"extra_snippets": True, "min_score": 1.1}, "min_score"),
        ({"extra_snippets": "sometimes"}, "extra_snippets"),
        ({"full_content": "text"}, "full_content"),
        (
            {"include_domains": [f"{index}.example" for index in range(11)]},
            "include_domains",
        ),
        (
            {"include_domains": ["a.example"], "exclude_domains": ["b.example"]},
            "mutually exclusive",
        ),
        (
            {"search_domain_filter": ["a.example"], "exclude_domains": ["b.example"]},
            "mutually exclusive",
        ),
        (
            {
                "include_domains": [f"include-{index}.example" for index in range(6)],
                "search_domain_filter": [f"filter-{index}.example" for index in range(5)],
            },
            "Invalid Staan domain filters",
        ),
        ({"include_domains": "example.com"}, "include_domains"),
        ({"search_domain_filter": "example.com"}, "search_domain_filter"),
        ({"exclude_domains": "example.com"}, "exclude_domains"),
    ],
)
def test_staan_rejects_invalid_parameters(optional_params: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        StaanSearchConfig().transform_search_request("query", optional_params)


def test_staan_rejects_non_boolean_extra_snippets_type() -> None:
    with pytest.raises(TypeError, match="extra_snippets must be a boolean"):
        StaanSearchConfig().transform_search_request("query", {"extra_snippets": 1})


def test_staan_response_preserves_all_enriched_result_fields() -> None:
    response = httpx.Response(
        200,
        json={
            "search_id": "search-id",
            "query": {
                "q": "vector database",
                "market": "en-us",
                "count": 10,
                "offset": 0,
            },
            "web": {
                "results": [
                    {
                        "title": "Search result",
                        "url": "https://example.com/page",
                        "snippet": "Provider preview",
                        "display_url": "example.com > page",
                        "hostname": "www.example.com",
                        "favicon_url": "https://example.com/favicon.ico",
                        "extra_snippets": [
                            {"chunk": "First relevant passage", "score": 0.91},
                            {"chunk": "Second relevant passage", "score": 0.82},
                            {"chunk": "Third relevant passage", "score": 0.67},
                        ],
                        "full_content": {
                            "text": "Complete page body",
                            "format": "markdown",
                            "length": 19,
                        },
                        "published_date": "2026-09-24T00:00:00Z",
                    }
                ],
                "displayed_results": 1,
            },
        },
    )

    result_response = StaanSearchConfig().transform_search_response(
        response,
        logging_obj=None,
        optional_params={},
    )

    assert result_response.results[0].model_dump(exclude_none=True) == {
        "title": "Search result",
        "url": "https://example.com/page",
        "snippet": ("Provider preview\n\nFirst relevant passage\n\nSecond relevant passage\n\nThird relevant passage"),
        "date": "2026-09-24T00:00:00Z",
        "display_url": "example.com > page",
        "hostname": "www.example.com",
        "favicon_url": "https://example.com/favicon.ico",
        "extra_snippets": (
            {"chunk": "First relevant passage", "score": 0.91},
            {"chunk": "Second relevant passage", "score": 0.82},
            {"chunk": "Third relevant passage", "score": 0.67},
        ),
        "full_content": {
            "text": "Complete page body",
            "format": "markdown",
            "length": 19,
        },
    }
    assert result_response.search_id == "search-id"
    assert result_response.displayed_results == 1
    assert result_response.query == {
        "q": "vector database",
        "market": "en-us",
        "count": 10,
        "offset": 0,
    }


def test_staan_provider_metadata_cannot_replace_filtered_results() -> None:
    response = httpx.Response(
        200,
        json={
            "results": [{"title": "Provider override", "url": "https://blocked.example/"}],
            "object": "provider-object",
            "web": {
                "results": [
                    {"title": "Allowed", "url": "https://allowed.example/"},
                    {"title": "Blocked", "url": "https://blocked.example/"},
                ]
            },
        },
    )

    result = StaanSearchConfig().transform_search_response(
        response,
        logging_obj=None,
        optional_params={"search_domain_filter": ["allowed.example"], "max_results": 1},
    )

    assert [item.title for item in result.results] == ["Allowed"]
    assert result.object == "search"


@pytest.mark.parametrize("max_results", [0, -1])
def test_staan_response_rejects_non_positive_max_results(max_results: int) -> None:
    response = httpx.Response(
        200,
        json={"web": {"results": [{"title": "Result", "url": "https://example.com/"}]}},
    )

    with pytest.raises(ValueError, match="max_results must be a positive integer"):
        StaanSearchConfig().transform_search_response(
            response,
            logging_obj=None,
            optional_params={"max_results": max_results},
        )


def test_staan_response_filters_results_by_domain_boundaries() -> None:
    response = httpx.Response(
        200,
        json={
            "web": {
                "results": [
                    {"title": "Root", "url": "https://example.com/"},
                    {"title": "Subdomain", "url": "https://docs.example.com/"},
                    {"title": "Lookalike", "url": "https://notexample.com/"},
                    {"title": "No host", "url": "not a URL"},
                ]
            }
        },
    )

    result = StaanSearchConfig().transform_search_response(
        response,
        logging_obj=None,
        optional_params={"search_domain_filter": ["www.example.com"]},
    )

    assert [item.title for item in result.results] == ["Root", "Subdomain"]


def test_staan_response_excludes_matching_domain_and_subdomains() -> None:
    response = httpx.Response(
        200,
        json={
            "web": {
                "results": [
                    {"title": "Excluded", "url": "https://www.example.com/"},
                    {"title": "Excluded subdomain", "url": "https://docs.example.com/"},
                    {"title": "Retained", "url": "https://other.com/"},
                ]
            }
        },
    )

    result = StaanSearchConfig().transform_search_response(
        response,
        logging_obj=None,
        optional_params={"exclude_domains": ["example.com"]},
    )

    assert [item.title for item in result.results] == ["Retained"]


def test_staan_response_preserves_empty_enrichment_values() -> None:
    response = httpx.Response(
        200,
        json={
            "web": {
                "results": [
                    {
                        "title": "Search result",
                        "url": "https://example.com/page",
                        "snippet": "Provider preview",
                        "extra_snippets": [],
                        "full_content": {"text": "", "format": "markdown", "length": 0},
                    }
                ]
            }
        },
    )

    result = (
        StaanSearchConfig()
        .transform_search_response(
            response,
            logging_obj=None,
            optional_params={},
        )
        .results[0]
    )

    assert result.snippet == "Provider preview"
    assert result.extra_snippets == ()
    assert result.full_content == {"text": "", "format": "markdown", "length": 0}


def test_staan_max_results_does_not_truncate_extra_snippets() -> None:
    response = httpx.Response(
        200,
        json={
            "web": {
                "results": [
                    {
                        "title": "Search result",
                        "url": "https://example.com/page",
                        "snippet": "Base",
                        "extra_snippets": [
                            {"chunk": "One", "score": 0.9},
                            {"chunk": "Two", "score": 0.8},
                        ],
                    },
                    {
                        "title": "Second result",
                        "url": "https://example.com/second",
                        "snippet": "Second base",
                    },
                ]
            }
        },
    )

    result = StaanSearchConfig().transform_search_response(
        response,
        logging_obj=None,
        optional_params={"max_results": 1},
    )

    assert len(result.results) == 1
    assert result.results[0].snippet == "Base\n\nOne\n\nTwo"
    assert result.results[0].extra_snippets == (
        {"chunk": "One", "score": 0.9},
        {"chunk": "Two", "score": 0.8},
    )


@pytest.mark.asyncio
async def test_search_tool_enrichment_defaults_reach_provider_call() -> None:
    router = SimpleNamespace(
        search_tools=[
            {
                "search_tool_name": "search",
                "litellm_params": {
                    "search_provider": "staan",
                    "api_key": "test-key",
                    "extra_snippets": True,
                    "max_snippets": 5,
                },
            }
        ]
    )

    response = httpx.Response(
        200,
        json={
            "web": {
                "results": [
                    {
                        "title": "Search result",
                        "url": "https://example.com/page",
                        "snippet": "Preview",
                        "extra_snippets": [
                            {"chunk": "Passage one", "score": 0.9},
                            {"chunk": "Passage two", "score": 0.8},
                        ],
                    }
                ]
            }
        },
    )
    with (
        patch.dict("os.environ", {"STAAN_API_KEY": "test-key"}),
        patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
            return_value=response,
        ) as post,
    ):
        result = await SearchAPIRouter.async_search_with_fallbacks_helper(
            router_instance=router,
            model="search",
            original_generic_function=litellm.asearch,
            query="vector search",
        )

        assert post.call_args.kwargs["json"] == {
            "q": "vector search",
            "extra_snippets": True,
            "max_snippets": 5,
        }
        assert result.results[0].snippet == "Preview\n\nPassage one\n\nPassage two"

        await SearchAPIRouter.async_search_with_fallbacks_helper(
            router_instance=router,
            model="search",
            original_generic_function=litellm.asearch,
            query="vector search",
            max_snippets=3,
            extra_snippets=False,
        )

        assert post.call_args.kwargs["json"] == {"q": "vector search"}
