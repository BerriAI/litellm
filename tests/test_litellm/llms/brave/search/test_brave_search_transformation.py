from unittest.mock import Mock

import pytest

from litellm.llms.brave.search.transformation import BraveSearchConfig, to_yyyy_mm_dd


def _config() -> BraveSearchConfig:
    return BraveSearchConfig()


def _resp(payload: dict, params: dict | None = None) -> Mock:
    """Mock httpx.Response with the .json() and .request.url.params shape
    Brave's transform_search_response actually reads."""
    r = Mock()
    r.json.return_value = payload
    req = Mock()
    req.url = Mock()
    req.url.params = params or {}
    r.request = req
    return r


def _result(**overrides):
    base = {
        "title": "Test Title",
        "url": "https://example.com",
        "description": "Test description",
        "page_age": None,
        "age": None,
        "fetched_content_timestamp": None,
    }
    return {**base, **overrides}


# --- ui_friendly_name / get_http_method ---


def test_ui_friendly_name():
    assert _config().ui_friendly_name() == "Brave Search"


def test_get_http_method_is_get():
    """Brave's /web/search endpoint takes query params, not a JSON body."""
    assert _config().get_http_method() == "GET"


# --- validate_environment ---


def test_validate_environment_with_explicit_key():
    headers = _config().validate_environment({}, api_key="explicit-key")
    assert headers["X-Subscription-Token"] == "explicit-key"
    assert headers["Accept"] == "application/json"
    assert headers["Accept-Encoding"] == "gzip"
    assert headers["Content-Type"] == "application/json"


def test_validate_environment_reads_env_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BRAVE_API_KEY", "env-key")
    assert _config().validate_environment({})["X-Subscription-Token"] == "env-key"


def test_validate_environment_missing_key_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="BRAVE_API_KEY"):
        _config().validate_environment({})


# --- get_complete_url ---


def test_get_complete_url_default_base(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("BRAVE_API_BASE", raising=False)
    url = _config().get_complete_url(None, {}, data={"_brave_params": {"q": "test"}})
    assert url.startswith("https://api.search.brave.com/res/v1/web/search?")
    assert "q=test" in url


def test_get_complete_url_reads_env_base(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BRAVE_API_BASE", "https://env-base.local/search")
    url = _config().get_complete_url(None, {}, data={"_brave_params": {"q": "test"}})
    assert url.startswith("https://env-base.local/search?")


def test_get_complete_url_without_brave_params_returns_bare_base(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("BRAVE_API_BASE", raising=False)
    assert _config().get_complete_url(None, {}, data=None) == "https://api.search.brave.com/res/v1/web/search"


# --- transform_search_request ---


def test_transform_search_request_joins_list_query():
    data = _config().transform_search_request(["foo", "bar"], {})
    assert data["_brave_params"]["q"] == "foo bar"


def test_transform_search_request_max_results_clamped_to_20():
    """Unlike Nimble, Brave's /web/search hard-caps at 20 results per request."""
    data = _config().transform_search_request("q", {"max_results": 500})
    assert data["_brave_params"]["count"] == 20


def test_transform_search_request_max_results_under_cap_passes_through():
    data = _config().transform_search_request("q", {"max_results": 5})
    assert data["_brave_params"]["count"] == 5


def test_transform_search_request_appends_domain_filters_as_site_clauses():
    data = _config().transform_search_request("q", {"search_domain_filter": ["arxiv.org", "nature.com"]})
    assert data["_brave_params"]["q"] == "(q) AND (site:arxiv.org OR site:nature.com)"


def test_transform_search_request_empty_domain_filter_leaves_query_unchanged():
    data = _config().transform_search_request("q", {"search_domain_filter": []})
    assert data["_brave_params"]["q"] == "q"


def test_transform_search_request_drops_max_tokens_per_page():
    """No Brave equivalent — must not leak through as an unrecognized param."""
    assert "max_tokens_per_page" not in _config().transform_search_request("q", {"max_tokens_per_page": 1024})[
        "_brave_params"
    ]


def test_transform_search_request_country_is_forwarded():
    """Regression test: the docstring always claimed country -> country, but
    there was previously no code path that actually implemented it — it
    relied on shared-list passthrough, which excluded it."""
    data = _config().transform_search_request("q", {"country": "US"})
    assert data["_brave_params"]["country"] == "US"


def test_transform_search_request_passes_through_unhandled_kwargs():
    data = _config().transform_search_request("q", {"safesearch": "strict"})
    assert data["_brave_params"]["safesearch"] == "strict"


@pytest.mark.parametrize(
    "optional_params",
    [
        {},  # absent entirely
        {"include_fetch_metadata": True},  # explicitly True
    ],
)
def test_transform_search_request_include_fetch_metadata_defaults_true(optional_params):
    data = _config().transform_search_request("q", optional_params)
    assert data["_brave_params"]["include_fetch_metadata"] is True


def test_transform_search_request_include_fetch_metadata_explicit_false_is_respected():
    data = _config().transform_search_request("q", {"include_fetch_metadata": False})
    assert data["_brave_params"]["include_fetch_metadata"] is False


# --- transform_search_request: date filtering ---


def test_transform_search_request_date_range_maps_to_freshness():
    data = _config().transform_search_request("q", {"start_date": "2022-04-01", "end_date": "2022-07-30"})
    assert data["_brave_params"]["freshness"] == "2022-04-01to2022-07-30"


def test_transform_search_request_only_start_date_is_dropped():
    """Brave's custom range needs both bounds; there's no correct open-ended equivalent."""
    data = _config().transform_search_request("q", {"start_date": "2022-04-01"})
    params = data["_brave_params"]
    assert "freshness" not in params
    assert "start_date" not in params  # must not leak through raw either


def test_transform_search_request_only_end_date_is_dropped():
    data = _config().transform_search_request("q", {"end_date": "2022-07-30"})
    params = data["_brave_params"]
    assert "freshness" not in params
    assert "end_date" not in params


def test_transform_search_request_no_dates_omits_freshness():
    data = _config().transform_search_request("q", {"max_results": 5})
    assert "freshness" not in data["_brave_params"]


def test_transform_search_request_explicit_freshness_not_clobbered_by_absent_dates():
    """A caller-supplied native `freshness` preset (e.g. 'pw') should survive
    when start_date/end_date aren't given at all."""
    data = _config().transform_search_request("q", {"freshness": "pw"})
    assert data["_brave_params"]["freshness"] == "pw"


# --- transform_search_response ---


def test_transform_search_response_extracts_basic_fields():
    resp = _resp({"web": {"results": [_result()]}})
    result = _config().transform_search_response(resp, logging_obj=Mock()).results[0]
    assert result.title == "Test Title"
    assert result.url == "https://example.com"
    assert result.snippet == "Test description"


def test_transform_search_response_date_from_page_age():
    resp = _resp({"web": {"results": [_result(page_age="2024-01-15")]}})
    result = _config().transform_search_response(resp, logging_obj=Mock()).results[0]
    assert result.date == "2024-01-15"


def test_transform_search_response_date_falls_back_to_age_when_page_age_absent():
    resp = _resp({"web": {"results": [_result(page_age=None, age="2024-01-15")]}})
    result = _config().transform_search_response(resp, logging_obj=Mock()).results[0]
    assert result.date == "2024-01-15"


def test_transform_search_response_last_updated_from_fetched_content_timestamp():
    resp = _resp({"web": {"results": [_result(fetched_content_timestamp="1705334400")]}})
    result = _config().transform_search_response(resp, logging_obj=Mock()).results[0]
    assert result.last_updated == "2024-01-15"


def test_transform_search_response_zero_hits():
    resp = _resp({"web": {"results": []}})
    assert _config().transform_search_response(resp, logging_obj=Mock()).results == []


def test_transform_search_response_result_filter_limits_sections():
    resp = _resp(
        {
            "web": {"results": [_result(title="web result")]},
            "news": {"results": [_result(title="news result")]},
        },
        params={"result_filter": "news"},
    )
    titles = [r.title for r in _config().transform_search_response(resp, logging_obj=Mock()).results]
    assert titles == ["news result"]


def test_transform_search_response_no_result_filter_includes_all_sections():
    resp = _resp(
        {
            "web": {"results": [_result(title="web result")]},
            "news": {"results": [_result(title="news result")]},
        },
    )
    titles = {r.title for r in _config().transform_search_response(resp, logging_obj=Mock()).results}
    assert titles == {"web result", "news result"}


def test_transform_search_response_max_results_limits_across_combined_sections():
    """count doesn't limit Brave's own per-section results server-side, so
    LiteLLM has to truncate across sections manually."""
    resp = _resp(
        {
            "web": {"results": [_result(title=f"web-{i}") for i in range(5)]},
            "news": {"results": [_result(title=f"news-{i}") for i in range(5)]},
        },
        params={"count": "3"},
    )
    results = _config().transform_search_response(resp, logging_obj=Mock()).results
    assert len(results) == 3


# --- to_yyyy_mm_dd helper ---


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        ("", None),
        ("not-a-date", None),
        ("2024-01-15", "2024-01-15"),
        ("2024/01/15", "2024-01-15"),
        (1705334400, "2024-01-15"),  # unix seconds
        (1705334400000, "2024-01-15"),  # unix milliseconds
    ],
)
def test_to_yyyy_mm_dd(value, expected):
    assert to_yyyy_mm_dd(value) == expected