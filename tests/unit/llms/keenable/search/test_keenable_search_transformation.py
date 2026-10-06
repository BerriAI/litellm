import json
from unittest.mock import Mock, patch

import httpx
import pytest

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.keenable.search.transformation import KeenableSearchConfig

DEFAULT_ROOT = "https://api.keenable.ai/v1"


@pytest.fixture(autouse=True)
def _no_server_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("KEENABLE_API_KEY", raising=False)


def _config() -> KeenableSearchConfig:
    return KeenableSearchConfig()


def _resp(payload, status_code: int = 200):
    r = Mock()
    r.status_code = status_code
    r.headers = {}
    r.content = (payload if isinstance(payload, str) else json.dumps(payload)).encode()
    return r


def _result(**overrides):
    base = {
        "title": "Test Title",
        "url": "https://example.com",
        "description": "",
        "snippet": "Test page text",
        "published_at": "2026-01-15T10:30:00Z",
        "acquired_at": "2026-01-16T08:12:34Z",
    }
    return {**base, **overrides}


def test_ui_friendly_name():
    assert _config().ui_friendly_name() == "Keenable"


def test_keyless_request_names_the_app_and_targets_the_public_endpoint():
    headers = _config().validate_environment({})

    assert "Authorization" not in headers
    assert headers["X-Keenable-Title"] == "litellm"
    assert headers["Content-Type"] == "application/json"
    assert _config().get_complete_url(None, {}) == f"{DEFAULT_ROOT}/search/public"


def test_caller_key_authenticates_and_targets_the_keyed_endpoint():
    headers = _config().validate_environment({}, api_key="caller-key")

    assert headers["Authorization"] == "Bearer caller-key"
    assert headers["X-Keenable-Title"] == "litellm"
    assert _config().get_complete_url(None, {}, api_key="caller-key") == f"{DEFAULT_ROOT}/search"


def test_server_key_authenticates_and_targets_the_keyed_endpoint(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("KEENABLE_API_KEY", "env-key")

    assert _config().validate_environment({})["Authorization"] == "Bearer env-key"
    assert _config().get_complete_url(None, {}) == f"{DEFAULT_ROOT}/search"


def test_validate_environment_does_not_mutate_and_is_idempotent():
    """The http handler re-runs validate_environment after search/main.py already did."""
    config = _config()
    caller_headers = {"X-Custom": "keep-me"}

    once = config.validate_environment(caller_headers, api_key="k")
    twice = config.validate_environment(once, api_key="k")

    assert caller_headers == {"X-Custom": "keep-me"}
    assert once == twice
    assert once["X-Custom"] == "keep-me"


@pytest.mark.parametrize(
    "api_base",
    [
        "https://self-hosted.local/v1",
        "https://self-hosted.local/v1/",
        "https://self-hosted.local/v1/search",
        "https://self-hosted.local/v1/search/public",
        "https://self-hosted.local/v1/search/public/",
    ],
)
@pytest.mark.parametrize(
    "api_key, path",
    [(None, "/search/public"), ("caller-key", "/search")],
)
def test_get_complete_url_is_idempotent_and_the_key_picks_the_path(api_base: str, api_key, path: str):
    # search/main.py builds the URL, then the http handler calls get_complete_url again with
    # that URL as api_base, so feeding the result back in must not change it.
    url = _config().get_complete_url(api_base, {}, api_key=api_key)

    assert url == f"https://self-hosted.local/v1{path}"
    assert _config().get_complete_url(url, {}, api_key=api_key) == url


def test_request_maps_query_and_max_results_and_forwards_provider_params():
    body = _config().transform_search_request(
        ["latest", "rust release"],
        {
            "max_results": 7,
            "country": "US",
            "max_tokens_per_page": 512,
            "published_after": "7d",
            "snippet_max_length": 500,
        },
    )

    assert body == {
        "query": "latest rust release",
        "max_results": 7,
        "published_after": "7d",
        "snippet_max_length": 500,
    }


def test_single_domain_filter_uses_the_site_field():
    body = _config().transform_search_request("asyncio", {"search_domain_filter": ["docs.python.org"]})

    assert body == {"query": "asyncio", "site": "docs.python.org"}


def test_several_domains_and_exclusions_become_query_clauses():
    body = _config().transform_search_request(
        "asyncio",
        {"search_domain_filter": ["realpython.com", "docs.python.org", "-medium.com", "-"]},
    )

    assert body == {"query": "asyncio (site:realpython.com OR site:docs.python.org) -site:medium.com"}


def test_single_domain_with_exclusion_combines_site_and_clause():
    body = _config().transform_search_request("asyncio", {"search_domain_filter": ["docs.python.org", "-medium.com"]})

    assert body == {"query": "asyncio -site:medium.com", "site": "docs.python.org"}


def test_explicit_site_wins_over_the_derived_one():
    body = _config().transform_search_request(
        "asyncio", {"search_domain_filter": ["docs.python.org"], "site": "peps.python.org"}
    )

    assert body["site"] == "peps.python.org"


def test_malformed_domain_filter_is_ignored():
    body = _config().transform_search_request("asyncio", {"search_domain_filter": "docs.python.org"})

    assert body == {"query": "asyncio"}


def test_response_maps_snippet_and_dates():
    response = _config().transform_search_response(
        _resp({"query": "q", "results": [_result(), _result(snippet="", description="Only a description")]}),
        logging_obj=Mock(),
    )

    first, second = response.results
    assert (first.title, first.url, first.snippet) == ("Test Title", "https://example.com", "Test page text")
    assert first.date == "2026-01-15T10:30:00Z"
    assert first.last_updated == "2026-01-16T08:12:34Z"
    assert second.snippet == "Only a description"


def test_degraded_result_maps_to_empty_strings():
    response = _config().transform_search_response(_resp({"results": [{}]}), logging_obj=Mock())

    only = response.results[0]
    assert (only.title, only.url, only.snippet, only.date, only.last_updated) == ("", "", "", None, None)


def test_empty_result_list_is_a_successful_search():
    assert _config().transform_search_response(_resp({"results": []}), logging_obj=Mock()).results == []


@pytest.mark.parametrize("payload", [{}, {"results": None}, "<html>bad gateway</html>"])
def test_body_that_is_not_a_search_response_raises(payload):
    with pytest.raises(BaseLLMException, match="Keenable Search: response does not match"):
        _config().transform_search_response(_resp(payload, status_code=200), logging_obj=Mock())


def test_error_class_surfaces_the_api_message():
    error = _config().get_error_class(
        '{"error": "Invalid parameter", "message": "\\"max_results\\": must be an integer between 1 and 50"}',
        400,
        {},
    )

    assert error.status_code == 400
    assert error.message.startswith('Keenable Search: "max_results": must be an integer between 1 and 50.')
    assert "KEENABLE_API_KEY" not in error.message


def test_rate_limit_error_points_at_the_api_key():
    error = _config().get_error_class(
        '{"error": "Rate limit exceeded", "message": "Public API hourly limit reached"}', 429, {}
    )

    assert error.status_code == 429
    assert "Public API hourly limit reached" in error.message
    assert "set KEENABLE_API_KEY" in error.message


def test_non_json_error_body_is_kept():
    assert "upstream timeout" in _config().get_error_class("upstream timeout", 502, {}).message


@pytest.mark.parametrize(
    "api_key, path, auth",
    [(None, "/search/public", None), ("caller-key", "/search", "Bearer caller-key")],
)
def test_litellm_search_end_to_end(api_key, path: str, auth):
    with patch(
        "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
        return_value=_resp({"query": "q", "results": [_result()]}),
    ) as mock_post:
        response = litellm.search(
            query="rust release",
            search_provider="keenable",
            api_key=api_key,
            max_results=3,
        )

    call = mock_post.call_args.kwargs
    assert call["url"] == f"{DEFAULT_ROOT}{path}"
    assert call["headers"]["X-Keenable-Title"] == "litellm"
    assert call["headers"].get("Authorization") == auth
    assert call["json"] == {"query": "rust release", "max_results": 3}
    assert response.results[0].snippet == "Test page text"


def test_litellm_search_surfaces_the_api_error_message():
    # A non-2xx answer must reach the caller as Keenable's own message, never as a schema mismatch.
    request = httpx.Request("POST", f"{DEFAULT_ROOT}/search/public")
    rate_limited = httpx.Response(
        429,
        json={"error": "Rate limit exceeded", "message": "Public API hourly limit reached"},
        request=request,
    )
    with (
        patch(
            "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
            side_effect=httpx.HTTPStatusError("429", request=request, response=rate_limited),
        ),
        pytest.raises(litellm.RateLimitError) as excinfo,
    ):
        litellm.search(query="rust release", search_provider="keenable")

    assert "Public API hourly limit reached" in str(excinfo.value)
    assert "set KEENABLE_API_KEY" in str(excinfo.value)
    assert "does not match" not in str(excinfo.value)
