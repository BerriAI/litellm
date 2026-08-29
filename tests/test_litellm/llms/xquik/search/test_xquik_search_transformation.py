import json
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from respx import MockRouter

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.llms.xquik.search.transformation import XquikSearchConfig
from litellm.search.cost_calculator import search_provider_cost_per_query
from litellm.types.utils import SearchProviders
from litellm.utils import ProviderConfigManager, get_model_info

pytestmark = pytest.mark.usefixtures("local_model_cost_map")


class _SearchLoggingStub:
    def pre_call(
        self,
        input: str,
        api_key: str | None,
        additional_args: dict[str, object],
    ) -> None:
        return None


def _config() -> XquikSearchConfig:
    return XquikSearchConfig()


def _response(payload: object, status_code: int = 200, headers: dict[str, str] | None = None) -> httpx.Response:
    content = payload if isinstance(payload, str) else json.dumps(payload)
    return httpx.Response(
        status_code=status_code,
        content=content.encode(),
        headers=headers,
        request=httpx.Request("GET", "https://xquik.com/api/v1/x/tweets/search"),
    )


def _tweet(**overrides: object) -> dict[str, object]:
    return {
        "id": "1893456789012345678",
        "text": "LiteLLM can now search X posts.",
        "createdAt": "2026-08-24T10:00:00Z",
        "url": "https://x.com/example/status/1893456789012345678",
        "likeCount": 12,
        "retweetCount": 3,
        "replyCount": 2,
        "quoteCount": 1,
        "viewCount": 900,
        "bookmarkCount": 4,
        "lang": "en",
        "author": {
            "id": "42",
            "username": "example",
            "name": "Example User",
            "followers": 100,
            "verified": True,
            "profilePicture": "https://example.com/avatar.jpg",
        },
        **overrides,
    }


def _params(query: str | list[str], optional_params: dict[str, object]) -> dict[str, list[str]]:
    request_data = _config().transform_search_request(query, optional_params)
    url = _config().get_complete_url(None, optional_params, data=request_data)
    return parse_qs(urlsplit(url).query)


def test_provider_registration_and_model_metadata() -> None:
    assert SearchProviders.XQUIK.value == "xquik"
    assert isinstance(
        ProviderConfigManager.get_provider_search_config(SearchProviders.XQUIK),
        XquikSearchConfig,
    )
    model_info = get_model_info("xquik/search", custom_llm_provider="xquik")
    assert model_info["mode"] == "search"
    assert model_info["input_cost_per_result"] == 0.00015


def test_provider_identity_and_http_method() -> None:
    assert _config().ui_friendly_name() == "Xquik"
    assert _config().get_http_method() == "GET"


def test_validate_environment_adds_explicit_key_without_mutating_headers() -> None:
    original = {"X-Custom": "keep", "X-API-Key": "old"}
    headers = _config().validate_environment(original, api_key="new")

    assert original == {"X-Custom": "keep", "X-API-Key": "old"}
    assert headers == {"X-Custom": "keep", "x-api-key": "new", "Accept": "application/json"}


@pytest.mark.parametrize(
    "headers",
    [
        {"x-api-key": "key"},
        {"Authorization": "Bearer guest-key"},
    ],
)
def test_validate_environment_accepts_documented_auth_headers(headers: dict[str, str]) -> None:
    validated = _config().validate_environment(headers)
    assert validated == {**headers, "Accept": "application/json"}


def test_validate_environment_is_idempotent() -> None:
    once = _config().validate_environment({}, api_key="key")
    twice = _config().validate_environment(once, api_key="key")
    assert twice == once


def test_validate_environment_requires_authentication() -> None:
    with pytest.raises(ValueError, match="requires api_key or an authentication header"):
        _config().validate_environment({})


@pytest.mark.parametrize(
    ("api_base", "expected"),
    [
        (None, "https://xquik.com/api/v1/x/tweets/search"),
        ("https://proxy.example/v1", "https://proxy.example/v1/x/tweets/search"),
        ("https://proxy.example/v1/x/tweets/search", "https://proxy.example/v1/x/tweets/search"),
        ("https://proxy.example/v1/x/tweets/search/", "https://proxy.example/v1/x/tweets/search"),
    ],
)
def test_get_complete_url_appends_endpoint_once(api_base: str | None, expected: str) -> None:
    assert _config().get_complete_url(api_base, {}) == expected


def test_get_complete_url_rejects_non_mapping_parameters() -> None:
    with pytest.raises(ValueError, match="request parameters must be a mapping"):
        _config().get_complete_url(None, {}, data={"_xquik_params": ["invalid"]})


def test_transform_request_maps_unified_parameters() -> None:
    params = _params(
        ["latest", "launch"],
        {
            "max_results": 25,
            "country": "us",
            "max_tokens_per_page": 1024,
            "queryType": "Top",
            "verifiedOnly": True,
        },
    )

    assert params == {
        "q": ["latest launch"],
        "limit": ["25"],
        "placeCountry": ["US"],
        "queryType": ["Top"],
        "verifiedOnly": ["true"],
    }


def test_transform_request_maps_include_and_exclude_domains() -> None:
    params = _params(
        "release notes",
        {"search_domain_filter": ["github.com", "docs.example", "-spam.example"]},
    )
    assert params["q"] == ["(release notes) (url:github.com OR url:docs.example) -url:spam.example"]


@pytest.mark.parametrize("domain_filter", [None, "example.com", [1]])
def test_transform_request_ignores_invalid_domain_filters(domain_filter: object) -> None:
    assert _params("query", {"search_domain_filter": domain_filter})["q"] == ["query"]


def test_transform_request_serializes_provider_specific_collections() -> None:
    params = _params("query", {"custom": {"nested": [1, 2]}})
    assert params["custom"] == ['{"nested":[1,2]}']


def test_transform_response_maps_standard_and_xquik_fields() -> None:
    response = _config().transform_search_response(
        _response(
            {
                "tweets": [_tweet()],
                "has_next_page": True,
                "next_cursor": "cursor-1",
            },
            headers={"x-request-id": "request-1"},
        ),
        logging_obj=None,
    )

    result = response.results[0]
    assert result.title == "Example User (@example)"
    assert result.url == "https://x.com/example/status/1893456789012345678"
    assert result.snippet == "LiteLLM can now search X posts."
    assert result.date == "2026-08-24T10:00:00Z"
    assert result.xquik_tweet == _tweet()
    assert response.has_next_page is True
    assert response.next_cursor == "cursor-1"
    assert response._hidden_params["billed_results"] == 1
    assert response._hidden_params["headers"]["x-request-id"] == "request-1"


@pytest.mark.parametrize(
    ("tweet", "expected_title", "expected_url"),
    [
        (
            {"id": "123", "author": {"username": "somebody"}},
            "@somebody",
            "https://x.com/somebody/status/123",
        ),
        ({"id": "123", "author": {"name": "Somebody"}}, "Somebody", ""),
        ({"id": "123"}, "X post 123", ""),
        ({}, "X post", ""),
    ],
)
def test_transform_response_builds_url_and_title_for_degraded_tweet(
    tweet: dict[str, object], expected_title: str, expected_url: str
) -> None:
    response = _config().transform_search_response(
        _response(
            {
                "tweets": [tweet],
                "has_next_page": False,
                "next_cursor": "",
            }
        ),
        logging_obj=None,
    )
    result = response.results[0]
    assert result.title == expected_title
    assert result.url == expected_url
    assert result.snippet == ""
    assert result.date is None


def test_transform_response_preserves_zero_result_page_and_billing() -> None:
    response = _config().transform_search_response(
        _response({"tweets": [], "has_next_page": False, "next_cursor": ""}),
        logging_obj=None,
    )
    assert response.results == []
    assert response._hidden_params["billed_results"] == 0


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        {},
        {"tweets": None, "has_next_page": False, "next_cursor": ""},
        {"tweets": ["bad"], "has_next_page": False, "next_cursor": ""},
    ],
)
def test_transform_response_rejects_malformed_success_body(payload: object) -> None:
    with pytest.raises(Exception, match="Xquik Search: response does not match"):
        _config().transform_search_response(_response(payload), logging_obj=None)


def test_transform_response_unwraps_error_and_preserves_retry_header() -> None:
    with pytest.raises(Exception, match="Xquik Search: Too many requests") as exc_info:
        _config().transform_search_response(
            _response(
                {"error": "rate_limit_exceeded", "message": "Too many requests. Try again later."},
                status_code=429,
                headers={"Retry-After": "60"},
            ),
            logging_obj=None,
        )
    assert exc_info.value.status_code == 429
    assert exc_info.value.headers["retry-after"] == "60"
    assert "docs.xquik.com/api-reference/x/search-tweets" in str(exc_info.value)


def test_transform_response_uses_error_code_when_message_is_absent() -> None:
    with pytest.raises(Exception, match="Xquik Search: unauthenticated"):
        _config().transform_search_response(
            _response({"error": "unauthenticated"}, status_code=401),
            logging_obj=None,
        )


@pytest.mark.respx()
def test_search_routes_through_get_and_records_exact_cost(respx_mock: MockRouter) -> None:
    route = respx_mock.get(url__regex=r"https://xquik\.com/api/v1/x/tweets/search.*").respond(
        json={
            "tweets": [_tweet()],
            "has_next_page": False,
            "next_cursor": "",
        },
        status_code=200,
    )
    response = litellm.search(
        query="xquik launch",
        search_provider="xquik",
        api_key="test-key",
        max_results=1,
        queryType="Latest",
    )

    request = route.calls[0].request
    assert parse_qs(urlsplit(str(request.url)).query) == {
        "q": ["xquik launch"],
        "limit": ["1"],
        "queryType": ["Latest"],
    }
    assert request.headers["x-api-key"] == "test-key"
    assert response.results[0].snippet == "LiteLLM can now search X posts."
    assert response._hidden_params["response_cost"] == pytest.approx(0.00015)


@pytest.mark.asyncio
async def test_async_handler_uses_injected_http_transport() -> None:
    async def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={
                "tweets": [_tweet()],
                "has_next_page": False,
                "next_cursor": "",
            },
            request=request,
        )

    transport_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    http_handler = AsyncHTTPHandler()
    await http_handler.close()
    http_handler.client = transport_client
    response = await BaseLLMHTTPHandler().async_search(
        query="xquik launch",
        optional_params={"max_results": 1},
        timeout=30,
        logging_obj=_SearchLoggingStub(),
        api_key="test-key",
        api_base=None,
        custom_llm_provider="xquik",
        client=http_handler,
        provider_config=_config(),
    )
    await http_handler.close()

    assert response.results[0].url == "https://x.com/example/status/1893456789012345678"


def test_cost_uses_exact_returned_result_count() -> None:
    assert search_provider_cost_per_query(
        model="xquik/search",
        custom_llm_provider="xquik",
        optional_params={"billed_results": 3, "max_results": 100},
    ) == pytest.approx((0.00045, 0.0))


def test_cost_estimates_from_requested_results_without_response() -> None:
    assert search_provider_cost_per_query(
        model="xquik/search",
        custom_llm_provider="xquik",
        number_of_queries=2,
        optional_params={"max_results": 5},
    ) == pytest.approx((0.0015, 0.0))


def test_cost_defaults_to_ten_results_for_estimates() -> None:
    assert search_provider_cost_per_query(
        model="xquik/search",
        custom_llm_provider="xquik",
    ) == pytest.approx((0.0015, 0.0))
