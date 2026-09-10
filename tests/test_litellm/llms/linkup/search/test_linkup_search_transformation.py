from unittest.mock import Mock

import pytest

from litellm.llms.linkup.search.transformation import LinkupSearchConfig


def _config() -> LinkupSearchConfig:
    return LinkupSearchConfig()


def _resp(payload: dict) -> Mock:
    r = Mock()
    r.json.return_value = payload
    return r


def _result(**overrides):
    base = {
        "type": "text",
        "name": "Test Title",
        "url": "https://example.com",
        "content": "Test content",
    }
    return {**base, **overrides}


# --- ui_friendly_name ---


def test_ui_friendly_name():
    assert _config().ui_friendly_name() == "Linkup"


# --- validate_environment ---


def test_validate_environment_with_explicit_key():
    headers = _config().validate_environment({}, api_key="explicit-key")
    assert headers["Authorization"] == "Bearer explicit-key"
    assert headers["Content-Type"] == "application/json"


def test_validate_environment_reads_env_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LINKUP_API_KEY", "env-key")
    assert _config().validate_environment({})["Authorization"] == "Bearer env-key"


def test_validate_environment_missing_key_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LINKUP_API_KEY", raising=False)
    with pytest.raises(ValueError, match="LINKUP_API_KEY"):
        _config().validate_environment({})


# --- get_complete_url ---


def test_get_complete_url_default_base(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LINKUP_API_BASE", raising=False)
    assert _config().get_complete_url(None, {}) == "https://api.linkup.so/v1/search"


def test_get_complete_url_reads_env_base(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LINKUP_API_BASE", "https://env-base.local/v1")
    assert _config().get_complete_url(None, {}) == "https://env-base.local/v1/search"


def test_get_complete_url_does_not_duplicate_search_suffix():
    assert _config().get_complete_url("https://self-hosted.local/v1/search", {}) == "https://self-hosted.local/v1/search"


def test_get_complete_url_appends_search_when_missing():
    assert _config().get_complete_url("https://self-hosted.local/v1", {}) == "https://self-hosted.local/v1/search"


# --- transform_search_request: query / defaults ---


def test_transform_search_request_joins_list_query():
    assert _config().transform_search_request(["foo", "bar"], {})["q"] == "foo bar"


def test_transform_search_request_string_query_unchanged():
    assert _config().transform_search_request("foo bar", {})["q"] == "foo bar"


def test_transform_search_request_defaults_depth_and_output_type():
    data = _config().transform_search_request("q", {})
    assert data["depth"] == "standard"
    assert data["outputType"] == "searchResults"


def test_transform_search_request_respects_explicit_depth_and_output_type():
    data = _config().transform_search_request("q", {"depth": "deep", "outputType": "sourcedAnswer"})
    assert data["depth"] == "deep"
    assert data["outputType"] == "sourcedAnswer"


# --- transform_search_request: unified param mapping ---


def test_transform_search_request_max_results_maps_to_maxResults():
    data = _config().transform_search_request("q", {"max_results": 10})
    assert data["maxResults"] == 10
    assert "max_results" not in data


def test_transform_search_request_search_domain_filter_maps_to_includeDomains():
    data = _config().transform_search_request("q", {"search_domain_filter": ["arxiv.org", "nature.com"]})
    assert data["includeDomains"] == ["arxiv.org", "nature.com"]
    assert "search_domain_filter" not in data


def test_transform_search_request_drops_max_tokens_per_page():
    """No Linkup equivalent — must not leak through as an unrecognized param."""
    assert "max_tokens_per_page" not in _config().transform_search_request("q", {"max_tokens_per_page": 1024})


def test_transform_search_request_country_has_no_native_equivalent():
    """Linkup has no native country filter; per the docstring this is intentionally unmapped —
    it should pass through raw rather than silently vanish, since Linkup's real API will just
    ignore an unrecognized field."""
    data = _config().transform_search_request("q", {"country": "US"})
    assert data.get("country") == "US"


# --- transform_search_request: date filtering ---


def test_transform_search_request_maps_start_date_to_from_date():
    data = _config().transform_search_request("q", {"start_date": "1999-03-20"})
    assert data["fromDate"] == "1999-03-20"
    assert "start_date" not in data


def test_transform_search_request_maps_end_date_to_to_date():
    data = _config().transform_search_request("q", {"end_date": "1999-04-20"})
    assert data["toDate"] == "1999-04-20"
    assert "end_date" not in data


def test_transform_search_request_date_range_together():
    data = _config().transform_search_request(
        "q", {"start_date": "1999-03-20", "end_date": "1999-04-20"}
    )
    assert data["fromDate"] == "1999-03-20"
    assert data["toDate"] == "1999-04-20"


def test_transform_search_request_without_dates_omits_both():
    data = _config().transform_search_request("q", {"max_results": 5})
    assert "fromDate" not in data
    assert "toDate" not in data


# --- transform_search_request: passthrough ---


def test_transform_search_request_passes_through_unhandled_kwargs():
    """A param this function doesn't explicitly handle should still reach Linkup unchanged."""
    data = _config().transform_search_request("q", {"includeImages": True})
    assert data["includeImages"] is True


def test_transform_search_request_does_not_duplicate_consumed_params():
    """Regression test: a translated param (start_date -> fromDate) must not
    also leak through raw via the passthrough path."""
    data = _config().transform_search_request(
        "q", {"start_date": "1999-03-20", "max_results": 5, "includeImages": True}
    )
    assert "start_date" not in data
    assert "max_results" not in data
    assert data["fromDate"] == "1999-03-20"
    assert data["maxResults"] == 5
    assert data["includeImages"] is True


# --- transform_search_response ---


def test_transform_search_response_text_result_fields():
    resp = _resp({"results": [_result()]})
    result = _config().transform_search_response(resp, logging_obj=Mock()).results[0]
    assert result.title == "Test Title"
    assert result.url == "https://example.com"
    assert result.snippet == "Test content"
    assert result.date is None
    assert result.last_updated is None


def test_transform_search_response_image_result_falls_back_to_url_for_title():
    resp = _resp({"results": [_result(type="image", name="", url="https://example.com/img.png")]})
    result = _config().transform_search_response(resp, logging_obj=Mock()).results[0]
    assert result.title == "https://example.com/img.png"


def test_transform_search_response_image_result_uses_name_when_present():
    resp = _resp({"results": [_result(type="image", name="Alt text")]})
    result = _config().transform_search_response(resp, logging_obj=Mock()).results[0]
    assert result.title == "Alt text"


def test_transform_search_response_preserves_order():
    resp = _resp({"results": [_result(name=n) for n in ("first", "second", "third")]})
    titles = [r.title for r in _config().transform_search_response(resp, logging_obj=Mock()).results]
    assert titles == ["first", "second", "third"]


def test_transform_search_response_zero_hits():
    resp = _resp({"results": []})
    assert _config().transform_search_response(resp, logging_obj=Mock()).results == []


def test_transform_search_response_unknown_result_type_is_skipped():
    """A result type that's neither "text" nor "image" (e.g. a future Linkup
    addition) shouldn't crash the call; it's just silently omitted."""
    resp = _resp({"results": [_result(type="video"), _result()]})
    results = _config().transform_search_response(resp, logging_obj=Mock()).results
    assert len(results) == 1
    assert results[0].title == "Test Title"