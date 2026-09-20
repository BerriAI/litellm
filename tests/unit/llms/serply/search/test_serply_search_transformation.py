import json
from unittest.mock import Mock
from urllib.parse import parse_qs, urlsplit

import pytest

from litellm.llms.serply.search.transformation import SerplySearchConfig


def _config() -> SerplySearchConfig:
    return SerplySearchConfig()


def _resp(payload: object, status_code: int = 200) -> Mock:
    r = Mock()
    r.status_code = status_code
    r.headers = {}
    r.content = (payload if isinstance(payload, str) else json.dumps(payload)).encode()
    return r


def _result(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "title": "Test Title",
        "link": "https://example.com",
        "description": "Test description",
        "position": 1,
        "realPosition": 1,
        "result_type": "organic",
        "metadata": {"display_url": "example.com"},
    }
    return {**base, **overrides}


def _params(query: str | list[str], optional_params: dict[str, object] | None = None) -> dict[str, list[str]]:
    """The query string transform_search_request + get_complete_url actually produce."""
    config = _config()
    data = config.transform_search_request(query, optional_params or {})
    return parse_qs(urlsplit(config.get_complete_url(None, {}, data=data)).query)


def test_ui_friendly_name():
    assert _config().ui_friendly_name() == "Serply"


def test_http_method_is_get():
    """Serply takes its parameters in the query string, so the handler must not post a body."""
    assert _config().get_http_method() == "GET"


def test_validate_environment_with_explicit_key():
    headers = _config().validate_environment({}, api_key="explicit-key")
    assert headers["X-Api-Key"] == "explicit-key"
    assert headers["Content-Type"] == "application/json"
    assert headers["User-Agent"] == "litellm"


def test_validate_environment_reads_env_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SERPLY_API_KEY", "env-key")
    assert _config().validate_environment({})["X-Api-Key"] == "env-key"


def test_validate_environment_missing_key_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SERPLY_API_KEY", raising=False)
    with pytest.raises(ValueError, match="SERPLY_API_KEY"):
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
    monkeypatch.delenv("SERPLY_API_BASE", raising=False)
    assert _config().get_complete_url(None, {}) == "https://api.serply.io/v1/search"


def test_get_complete_url_reads_env_base(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SERPLY_API_BASE", "https://env-base.local/v1")
    assert _config().get_complete_url(None, {}) == "https://env-base.local/v1/search"


@pytest.mark.parametrize(
    "api_base",
    [
        "https://self-hosted.local/v1",
        "https://self-hosted.local/v1/",
        "https://self-hosted.local/v1/search",
        "https://self-hosted.local/v1/search/",
    ],
)
def test_get_complete_url_appends_search_exactly_once(api_base: str):
    assert _config().get_complete_url(api_base, {}) == "https://self-hosted.local/v1/search"


def test_get_complete_url_encodes_the_transformed_params():
    url = _config().get_complete_url(None, {}, data={"_serply_params": {"q": "a b", "num": 5}})
    assert url == "https://api.serply.io/v1/search?q=a+b&num=5"


@pytest.mark.parametrize("data", [None, {}, {"q": "not-wrapped"}, [{"_serply_params": {"q": "x"}}]])
def test_get_complete_url_without_usable_params_returns_a_bare_endpoint(data: object):
    """A malformed body must not produce a URL with a stray query string."""
    assert _config().get_complete_url(None, {}, data=data) == "https://api.serply.io/v1/search"


def test_transform_search_request_wraps_params_for_the_url_builder():
    """The GET handler only sends the URL, so the params have to reach get_complete_url."""
    assert _config().transform_search_request("q", {}) == {"_serply_params": {"q": "q"}}


def test_transform_search_request_joins_list_query():
    assert _params(["foo", "bar"])["q"] == ["foo bar"]


def test_transform_search_request_maps_max_results_to_num():
    assert _params("q", {"max_results": 5})["num"] == ["5"]


def test_transform_search_request_max_results_is_not_clamped():
    """Serply serves one page of Google results, so a larger value returns about ten rather
    than failing; clamping here would only hide that from the caller."""
    assert _params("q", {"max_results": 500})["num"] == ["500"]


def test_transform_search_request_lowercases_country():
    assert _params("q", {"country": "US"})["gl"] == ["us"]


def test_transform_search_request_ignores_non_string_country():
    assert "gl" not in _params("q", {"country": 42})


def test_transform_search_request_drops_max_tokens_per_page():
    assert "max_tokens_per_page" not in _params("q", {"max_tokens_per_page": 1024})


def test_transform_search_request_passes_through_native_params():
    """`tbs` is how a caller reaches Serply's time-range filter, which has no unified param."""
    assert _params("q", {"tbs": "qdr:w", "hl": "de"}) | {} == {"q": ["q"], "tbs": ["qdr:w"], "hl": ["de"]}


def test_transform_search_request_native_param_wins_over_unified():
    """An explicit provider-native value must not be silently clobbered by the unified param."""
    assert _params("q", {"max_results": 5, "num": 3})["num"] == ["3"]


def test_transform_search_request_appends_include_domains():
    assert _params("machine learning", {"search_domain_filter": ["arxiv.org", "nature.com"]})["q"] == [
        "(machine learning) (site:arxiv.org OR site:nature.com)"
    ]


def test_transform_search_request_appends_exclude_domains():
    assert _params("q", {"search_domain_filter": ["-spam.com"]})["q"] == ["(q) -site:spam.com"]


def test_transform_search_request_mixes_include_and_exclude_domains():
    assert _params("q", {"search_domain_filter": ["arxiv.org", "-spam.com"]})["q"] == [
        "(q) (site:arxiv.org) -site:spam.com"
    ]


@pytest.mark.parametrize("domains", ["arxiv.org", [], ["-"], 7, None])
def test_transform_search_request_leaves_query_alone_for_unusable_domain_filter(domains: object):
    """The filter only ever narrows an otherwise valid search, so it must not raise."""
    assert _params("q", {"search_domain_filter": domains})["q"] == ["q"]


def test_transform_search_response_maps_the_documented_fields():
    result = _config().transform_search_response(_resp({"results": [_result()]}), logging_obj=Mock()).results[0]
    assert result.title == "Test Title"
    assert result.url == "https://example.com"
    assert result.snippet == "Test description"


@pytest.mark.parametrize("published_time", ["Aug 11, 2026", "3 days ago"])
def test_transform_search_response_reads_published_time(published_time: str):
    """Google shows an absolute date for older pages and a relative one for the last week;
    both reach the caller verbatim."""
    resp = _config().transform_search_response(
        _resp({"results": [_result(metadata={"published_time": published_time})]}), logging_obj=Mock()
    )
    assert resp.results[0].date == published_time


@pytest.mark.parametrize("metadata", [{}, "not-a-dict", None])
def test_transform_search_response_date_is_none_without_published_time(metadata: object):
    resp = _config().transform_search_response(_resp({"results": [_result(metadata=metadata)]}), logging_obj=Mock())
    assert resp.results[0].date is None


def test_transform_search_response_keeps_metadata():
    """`sitelinks` and `attributes` have no unified home; they must still reach the caller."""
    metadata = {"display_url": "example.com", "sitelinks": [{"title": "Intro", "url": "https://example.com/#i"}]}
    resp = _config().transform_search_response(_resp({"results": [_result(metadata=metadata)]}), logging_obj=Mock())
    assert resp.results[0].metadata == metadata


def test_transform_search_response_omits_metadata_when_absent():
    resp = _config().transform_search_response(_resp({"results": [_result(metadata=None)]}), logging_obj=Mock())
    assert not hasattr(resp.results[0], "metadata")


def test_transform_search_response_preserves_order():
    """Serply ranks results itself via `position`, so the order is reported as received."""
    resp = _config().transform_search_response(
        _resp({"results": [_result(title=t) for t in ("first", "second", "third")]}), logging_obj=Mock()
    )
    assert [r.title for r in resp.results] == ["first", "second", "third"]


def test_transform_search_response_degraded_result_does_not_fail_the_call():
    resp = _config().transform_search_response(
        _resp({"results": [{"link": "https://example.com"}, _result()]}), logging_obj=Mock()
    )
    assert len(resp.results) == 2
    assert resp.results[0].title == ""
    assert resp.results[0].snippet == ""
    assert resp.results[1].title == "Test Title"


def test_transform_search_response_zero_hits():
    """A search with no hits really does come back as `"results": []`."""
    payload = {"results": [], "ads": [], "answers": [], "related_searches": {"text": []}}
    assert _config().transform_search_response(_resp(payload), logging_obj=Mock()).results == []


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
    """A body LiteLLM cannot parse must not be reported as a successful zero-result search."""
    with pytest.raises(Exception, match="Serply Search"):
        _config().transform_search_response(_resp(body, status_code=502), logging_obj=Mock())


def test_get_error_class_attributes_the_provider():
    error = _config().get_error_class(error_message="quota exceeded", status_code=429, headers={})
    assert error.status_code == 429
    assert "Serply Search: quota exceeded" in str(error)
    assert "serply.io/docs" in str(error)


def test_get_error_class_unwraps_serply_detail_envelope():
    """Verbatim body from a live 401; the raw JSON envelope should not reach the user."""
    error = _config().get_error_class(error_message='{"detail":"Invalid API key"}', status_code=401, headers={})
    assert str(error) == "Serply Search: Invalid API key. See https://serply.io/docs for details."


@pytest.mark.parametrize("body", ["<html>502 Bad Gateway</html>", '{"detail": null}'])
def test_get_error_class_falls_back_to_the_raw_body(body: str):
    assert f"Serply Search: {body}." in str(_config().get_error_class(body, status_code=500, headers={}))
