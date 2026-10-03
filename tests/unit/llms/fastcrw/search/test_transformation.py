import os
from unittest.mock import Mock, patch

import pytest

import litellm
from litellm.llms.fastcrw.search.transformation import FastCRWSearchConfig


def _config() -> FastCRWSearchConfig:
    return FastCRWSearchConfig()


def test_fastcrw_search_request_body():
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "success": True,
        "data": [
            {
                "title": "Test Title",
                "url": "https://example.com",
                "description": "Test description",
                "markdown": "Test content",
            }
        ],
    }

    with (
        patch.dict(os.environ, {"CRW_API_KEY": "test-api-key"}),
        patch(
            "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
            return_value=mock_response,
        ) as mock_post,
    ):
        response = litellm.search(
            query="test query",
            search_provider="fastcrw",
            max_results=10,
        )

        assert mock_post.called
        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs.get("url", "").endswith("/search")

        request_body = call_kwargs.get("json")
        assert request_body is not None
        assert request_body["query"] == "test query"
        assert request_body["limit"] == 10

        assert len(response.results) == 1
        result = response.results[0]
        assert result.title == "Test Title"
        assert result.url == "https://example.com"
        assert result.snippet == "Test content"


def test_ui_friendly_name():
    assert _config().ui_friendly_name() == "fastCRW"


def test_validate_environment_with_explicit_key():
    headers = _config().validate_environment({}, api_key="explicit-key")
    assert headers["Authorization"] == "Bearer explicit-key"
    assert headers["Content-Type"] == "application/json"


def test_validate_environment_reads_env_key():
    with patch.dict(os.environ, {"CRW_API_KEY": "env-key"}, clear=False):
        headers = _config().validate_environment({})
    assert headers["Authorization"] == "Bearer env-key"


def test_validate_environment_missing_key_raises():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="CRW_API_KEY"):
            _config().validate_environment({})


def test_get_complete_url_default_base():
    with patch.dict(os.environ, {}, clear=True):
        assert _config().get_complete_url(None, {}) == "https://fastcrw.com/api/v1/search"


def test_get_complete_url_appends_search():
    assert (
        _config().get_complete_url("https://self-hosted.local/api/v1", {})
        == "https://self-hosted.local/api/v1/search"
    )


def test_get_complete_url_does_not_double_append():
    assert (
        _config().get_complete_url("https://self-hosted.local/api/v1/search", {})
        == "https://self-hosted.local/api/v1/search"
    )


def test_get_complete_url_reads_env_base():
    with patch.dict(
        os.environ, {"CRW_API_BASE": "https://env-base.local/v1"}, clear=True
    ):
        assert _config().get_complete_url(None, {}) == "https://env-base.local/v1/search"


def test_transform_search_request_basic():
    data = _config().transform_search_request("hello", {"max_results": 5})
    assert data["query"] == "hello"
    assert data["limit"] == 5
    assert data["scrapeOptions"]["formats"] == ["markdown"]
    assert data["scrapeOptions"]["onlyMainContent"] is True


def test_transform_search_request_joins_list_query():
    assert _config().transform_search_request(["foo", "bar"], {})["query"] == "foo bar"


def test_transform_search_request_passes_through_extra_params():
    data = _config().transform_search_request("q", {"sources": ["web", "images"]})
    assert data["sources"] == ["web", "images"]


def test_transform_search_request_preserves_explicit_scrape_options():
    custom = {"formats": ["html"]}
    data = _config().transform_search_request("q", {"scrapeOptions": custom})
    assert data["scrapeOptions"] == custom


def _resp(payload):
    r = Mock()
    r.json.return_value = payload
    return r


def test_transform_search_response_prefers_markdown():
    resp = _config().transform_search_response(
        _resp(
            {
                "success": True,
                "data": [
                    {
                        "title": "T",
                        "url": "https://e.com",
                        "description": "d",
                        "markdown": "md",
                    }
                ],
            }
        ),
        logging_obj=Mock(),
    )
    assert len(resp.results) == 1
    assert resp.results[0].snippet == "md"


def test_transform_search_response_falls_back_to_description():
    resp = _config().transform_search_response(
        _resp(
            {
                "success": True,
                "data": [
                    {"title": "T", "url": "https://e.com", "description": "only-desc"}
                ],
            }
        ),
        logging_obj=Mock(),
    )
    assert resp.results[0].snippet == "only-desc"


def test_transform_search_response_empty_data():
    resp = _config().transform_search_response(
        _resp({"success": True, "data": []}), logging_obj=Mock()
    )
    assert resp.results == []


def test_transform_search_response_non_list_data():
    resp = _config().transform_search_response(
        _resp({"success": True, "data": {"unexpected": "shape"}}), logging_obj=Mock()
    )
    assert resp.results == []
