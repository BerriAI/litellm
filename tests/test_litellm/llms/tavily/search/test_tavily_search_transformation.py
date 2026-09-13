from unittest.mock import Mock

import pytest

from litellm.llms.tavily.search.transformation import TavilySearchConfig


def _config() -> TavilySearchConfig:
    return TavilySearchConfig()


def _resp(payload: dict) -> Mock:
    r = Mock()
    r.json.return_value = payload
    return r


def _result(**overrides):
    base = {
        "title": "Test Title",
        "url": "https://example.com",
        "content": "Test content",
    }
    return {**base, **overrides}


# --- ui_friendly_name ---


def test_ui_friendly_name():
    assert _config().ui_friendly_name() == "Tavily"


# --- validate_environment ---


def test_validate_environment_with_explicit_key():
    headers = _config().validate_environment({}, api_key="explicit-key")
    assert headers["Authorization"] == "Bearer explicit-key"
    assert headers["Content-Type"] == "application/json"


def test_validate_environment_reads_env_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TAVILY_API_KEY", "env-key")
    assert _config().validate_environment({})["Authorization"] == "Bearer env-key"


def test_validate_environment_missing_key_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(ValueError, match="TAVILY_API_KEY"):
        _config().validate_environment({})


# --- get_complete_url ---


def test_get_complete_url_default_base(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TAVILY_API_BASE", raising=False)
    assert _config().get_complete_url(None, {}) == "https://api.tavily.com/search"


def test_get_complete_url_reads_env_base(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TAVILY_API_BASE", "https://env-base.local")
    assert _config().get_complete_url(None, {}) == "https://env-base.local/search"


def test_get_complete_url_does_not_duplicate_search_suffix():
    assert _config().get_complete_url("https://self-hosted.local/search", {}) == "https://self-hosted.local/search"


# --- transform_search_request: query ---


def test_transform_search_request_joins_list_query():
    assert _config().transform_search_request(["foo", "bar"], {})["query"] == "foo bar"


def test_transform_search_request_string_query_unchanged():
    assert _config().transform_search_request("foo bar", {})["query"] == "foo bar"


# --- transform_search_request: unified param mapping ---


def test_transform_search_request_max_results_passes_through():
    data = _config().transform_search_request("q", {"max_results": 10})
    assert data["max_results"] == 10


def test_transform_search_request_search_domain_filter_maps_to_include_domains():
    data = _config().transform_search_request("q", {"search_domain_filter": ["arxiv.org", "nature.com"]})
    assert data["include_domains"] == ["arxiv.org", "nature.com"]
    assert "search_domain_filter" not in data


def test_transform_search_request_lowercases_country():
    data = _config().transform_search_request("q", {"country": "US"})
    assert data["country"] == "us"


def test_transform_search_request_drops_max_tokens_per_page():
    """No Tavily equivalent — must not leak through as an unrecognized param."""
    assert "max_tokens_per_page" not in _config().transform_search_request("q", {"max_tokens_per_page": 1024})


# --- transform_search_request: date filtering (native passthrough) ---


def test_transform_search_request_date_range_passes_through_native_names():
    """Tavily already uses the unified spec's own field names for date filtering."""
    data = _config().transform_search_request(
        "q", {"start_date": "1999-03-20", "end_date": "1999-04-20"}
    )
    assert data["start_date"] == "1999-03-20"
    assert data["end_date"] == "1999-04-20"


def test_transform_search_request_single_date_passes_through():
    """Unlike Brave, Tavily has no combined-range requirement — a lone bound is valid."""
    data = _config().transform_search_request("q", {"start_date": "1999-03-20"})
    assert data["start_date"] == "1999-03-20"
    assert "end_date" not in data


def test_transform_search_request_without_dates_omits_both():
    data = _config().transform_search_request("q", {"max_results": 5})
    assert "start_date" not in data
    assert "end_date" not in data


# --- transform_search_request: passthrough ---


def test_transform_search_request_passes_through_unhandled_kwargs():
    """Native Tavily-specific params (topic, search_depth, time_range, etc.)
    aren't explicitly mapped, but should still reach the request unchanged."""
    data = _config().transform_search_request("q", {"topic": "news", "search_depth": "advanced"})
    assert data["topic"] == "news"
    assert data["search_depth"] == "advanced"


def test_transform_search_request_does_not_duplicate_consumed_params():
    """Regression test: a translated param (search_domain_filter -> include_domains)
    must not also leak through raw via the passthrough path."""
    data = _config().transform_search_request(
        "q", {"search_domain_filter": ["arxiv.org"], "max_results": 5}
    )
    assert "search_domain_filter" not in data
    assert data["include_domains"] == ["arxiv.org"]
    assert data["max_results"] == 5


# --- transform_search_response ---


def test_transform_search_response_extracts_basic_fields():
    resp = _resp({"results": [_result()]})
    result = _config().transform_search_response(resp, logging_obj=Mock()).results[0]
    assert result.title == "Test Title"
    assert result.url == "https://example.com"
    assert result.snippet == "Test content"


def test_transform_search_response_date_and_last_updated_are_none():
    """Tavily's response has no date/last_updated equivalent."""
    resp = _resp({"results": [_result()]})
    result = _config().transform_search_response(resp, logging_obj=Mock()).results[0]
    assert result.date is None
    assert result.last_updated is None


def test_transform_search_response_zero_hits():
    resp = _resp({"results": []})
    assert _config().transform_search_response(resp, logging_obj=Mock()).results == []


def test_transform_search_response_preserves_order():
    resp = _resp({"results": [_result(title=t) for t in ("first", "second", "third")]})
    titles = [r.title for r in _config().transform_search_response(resp, logging_obj=Mock()).results]
    assert titles == ["first", "second", "third"]