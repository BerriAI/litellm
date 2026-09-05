import json
from unittest.mock import Mock

import pytest

from litellm.llms.search1api.search.transformation import Search1APISearchConfig


def _config() -> Search1APISearchConfig:
    return Search1APISearchConfig()


def _resp(payload, status_code: int = 200):
    r = Mock()
    r.status_code = status_code
    r.headers = {}
    r.text = payload if isinstance(payload, str) else json.dumps(payload)
    r.content = r.text.encode()
    return r


def _result(**overrides):
    base = {
        "title": "Test Title",
        "link": "https://example.com",
        "snippet": "Test snippet",
    }
    return {**base, **overrides}


def _payload(*results, **extra):
    return {"searchParameters": {"query": "q"}, "results": list(results), **extra}


def test_ui_friendly_name():
    assert _config().ui_friendly_name() == "Search1API"


def test_validate_environment_with_explicit_key():
    headers = _config().validate_environment({}, api_key="explicit-key")
    assert headers["Authorization"] == "Bearer explicit-key"
    assert headers["Content-Type"] == "application/json"


def test_validate_environment_reads_env_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SEARCH1API_KEY", raising=False)
    monkeypatch.setenv("SEARCH1API_API_KEY", "env-key")
    assert _config().validate_environment({})["Authorization"] == "Bearer env-key"


def test_validate_environment_falls_back_to_search1api_key(monkeypatch: pytest.MonkeyPatch):
    """`SEARCH1API_KEY` is what Search1API's own CLI/SDKs read, so a user who already has it set is not asked twice."""
    monkeypatch.delenv("SEARCH1API_API_KEY", raising=False)
    monkeypatch.setenv("SEARCH1API_KEY", "cli-key")
    assert _config().validate_environment({})["Authorization"] == "Bearer cli-key"


def test_validate_environment_prefers_litellm_style_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SEARCH1API_API_KEY", "litellm-key")
    monkeypatch.setenv("SEARCH1API_KEY", "cli-key")
    assert _config().validate_environment({})["Authorization"] == "Bearer litellm-key"


def test_validate_environment_missing_key_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SEARCH1API_API_KEY", raising=False)
    monkeypatch.delenv("SEARCH1API_KEY", raising=False)
    with pytest.raises(ValueError, match="SEARCH1API_API_KEY"):
        _config().validate_environment({})


def test_validate_environment_does_not_mutate_and_is_idempotent():
    """The http handler re-runs validate_environment after search/main.py already did."""
    config = _config()
    caller_headers = {"X-Custom": "keep-me"}

    once = config.validate_environment(caller_headers, api_key="k")
    twice = config.validate_environment(once, api_key="k")

    assert caller_headers == {"X-Custom": "keep-me"}
    assert once == twice
    assert once["X-Custom"] == "keep-me"


def test_get_complete_url_default_base(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SEARCH1API_API_BASE", raising=False)
    assert _config().get_complete_url(None, {}) == "https://api.search1api.com/search"


def test_get_complete_url_reads_env_base(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SEARCH1API_API_BASE", "https://env-base.local")
    assert _config().get_complete_url(None, {}) == "https://env-base.local/search"


@pytest.mark.parametrize(
    "api_base",
    [
        "https://self-hosted.local",
        "https://self-hosted.local/",
        "https://self-hosted.local/search",
        "https://self-hosted.local/search/",
    ],
)
def test_get_complete_url_appends_search_exactly_once(api_base: str):
    assert _config().get_complete_url(api_base, {}) == "https://self-hosted.local/search"


def test_transform_search_request_joins_list_query():
    assert _config().transform_search_request(["foo", "bar"], {})["query"] == "foo bar"


def test_transform_search_request_defaults_max_results_to_unified_spec():
    """Search1API defaults to 5 results; the unified spec documents 10, so 10 is sent explicitly."""
    assert _config().transform_search_request("q", {})["max_results"] == 10


def test_transform_search_request_max_results_is_not_clamped():
    """Search1API validates 1-50 itself; a clearer error beats silently rewriting the request."""
    assert _config().transform_search_request("q", {"max_results": 500})["max_results"] == 500


@pytest.mark.parametrize("dropped", ["country", "max_tokens_per_page"])
def test_transform_search_request_drops_params_without_equivalent(dropped: str):
    assert dropped not in _config().transform_search_request("q", {dropped: "US"})


def test_transform_search_request_splits_domain_filter():
    data = _config().transform_search_request("q", {"search_domain_filter": ["arxiv.org", "-spam.com", "nature.com"]})
    assert data["include_sites"] == ("arxiv.org", "nature.com")
    assert data["exclude_sites"] == ("spam.com",)
    assert "search_domain_filter" not in data


def test_transform_search_request_omits_empty_site_lists():
    data = _config().transform_search_request("q", {"search_domain_filter": ["arxiv.org"]})
    assert data["include_sites"] == ("arxiv.org",)
    assert "exclude_sites" not in data


def test_transform_search_request_ignores_non_list_domain_filter():
    assert "include_sites" not in _config().transform_search_request("q", {"search_domain_filter": "arxiv.org"})


@pytest.mark.parametrize("native_key", ["include_sites", "exclude_sites"])
def test_transform_search_request_native_sites_win(native_key: str):
    """An explicit provider-native value must not be silently clobbered by the unified param."""
    data = _config().transform_search_request(
        "q",
        {"search_domain_filter": ["derived.com", "-derived-ex.com"], native_key: ["native.com"]},
    )
    assert data[native_key] == ["native.com"]


def test_transform_search_request_forwards_provider_params():
    data = _config().transform_search_request(
        "q",
        {"search_service": "bing", "time_range": "month", "language": "de"},
    )
    assert data["search_service"] == "bing"
    assert data["time_range"] == "month"
    assert data["language"] == "de"


@pytest.mark.parametrize("param", ["crawl_results", "image"])
def test_transform_search_request_rejects_params_the_unified_response_cannot_carry(param: str):
    """Fetched page text and image URLs have no field in the unified response, and every crawled page
    bills a Search1API credit LiteLLM would not track, so these must fail loudly instead of silently
    costing money for output that is thrown away."""
    with pytest.raises(ValueError, match=param):
        _config().transform_search_request("q", {param: 1})


@pytest.mark.parametrize("param, value", [("crawl_results", 0), ("image", False)])
def test_transform_search_request_drops_disabled_unsupported_params(param: str, value):
    """Explicitly disabling crawling or images is a valid search and must not be rejected."""
    data = _config().transform_search_request("q", {param: value, "search_service": "bing"})
    assert param not in data
    assert data["search_service"] == "bing"


def test_transform_search_response_maps_fields():
    resp = _config().transform_search_response(_resp(_payload(_result())), logging_obj=Mock())
    assert resp.object == "search"
    assert resp.results[0].title == "Test Title"
    assert resp.results[0].url == "https://example.com"
    assert resp.results[0].snippet == "Test snippet"
    assert resp.results[0].date is None


def test_transform_search_response_ignores_fields_outside_the_unified_shape():
    """`content` and `images` only appear for params this adapter rejects; a caller-supplied `api_base`
    proxy may still return them and they must not leak into the unified response."""
    resp = _config().transform_search_response(
        _resp(_payload(_result(content="Full page text"), images=["https://img.example/1.png"])),
        logging_obj=Mock(),
    )
    assert not hasattr(resp.results[0], "content")
    assert not hasattr(resp, "images")


def test_transform_search_response_preserves_order():
    resp = _config().transform_search_response(
        _resp(_payload(*(_result(title=t) for t in ("first", "second", "third")))),
        logging_obj=Mock(),
    )
    assert [r.title for r in resp.results] == ["first", "second", "third"]


def test_transform_search_response_degraded_result_does_not_fail_the_call():
    resp = _config().transform_search_response(
        _resp(_payload({"link": "https://example.com"}, _result())), logging_obj=Mock()
    )
    assert len(resp.results) == 2
    assert resp.results[0].title == ""
    assert resp.results[0].snippet == ""
    assert resp.results[1].title == "Test Title"


def test_transform_search_response_zero_hits():
    """A search with no hits really does come back as `"results": []` with HTTP 200."""
    assert _config().transform_search_response(_resp(_payload()), logging_obj=Mock()).results == []


@pytest.mark.parametrize(
    "body",
    [
        "<html>502 Bad Gateway</html>",  # non-JSON body
        '{"results": ["garbage"]}',  # right key, wrong element shape
        '{"results": {"unexpected": "shape"}}',
        '{"results": null}',  # must not degrade to a successful empty search
        "{}",  # ditto for an absent key
    ],
)
def test_transform_search_response_malformed_body_raises_instead_of_reporting_empty(body: str):
    """A 2xx body LiteLLM cannot parse must not be reported as a successful zero-result search."""
    with pytest.raises(Exception, match="Search1API"):
        _config().transform_search_response(_resp(body), logging_obj=Mock())


def test_transform_search_response_non_2xx_surfaces_search1api_message():
    """When a non-2xx body reaches the transform, the user sees Search1API's own message, not schema output."""
    body = '{"ok":false,"error":"Payment Required","message":"Insufficient credits"}'
    with pytest.raises(Exception, match=r"^Search1API: Insufficient credits\. See ") as excinfo:
        _config().transform_search_response(_resp(body, status_code=402), logging_obj=Mock())
    assert excinfo.value.status_code == 402


def test_get_error_class_attributes_the_provider():
    error = _config().get_error_class(error_message="quota exceeded", status_code=429, headers={})
    assert error.status_code == 429
    assert "Search1API: quota exceeded" in str(error)
    assert "s1.dev/docs" in str(error)


def test_get_error_class_unwraps_search1api_message_envelope():
    """Verbatim shape of a Search1API 401; the raw JSON envelope should not reach the user."""
    error = _config().get_error_class(
        error_message='{"ok":false,"error":"Unauthorized","message":"Unauthorized: Invalid bearer credential"}',
        status_code=401,
        headers={},
    )
    assert (
        str(error) == "Search1API: Unauthorized: Invalid bearer credential. "
        "See https://s1.dev/docs/basic/search for details."
    )


def test_get_error_class_falls_back_to_error_field():
    error = _config().get_error_class(error_message='{"ok":false,"error":"Payment Required"}', status_code=402, headers={})
    assert str(error).startswith("Search1API: Payment Required.")


@pytest.mark.parametrize("body", ["<html>502 Bad Gateway</html>", '{"message": null}'])
def test_get_error_class_falls_back_to_the_raw_body(body: str):
    assert f"Search1API: {body}." in str(_config().get_error_class(body, status_code=500, headers={}))
