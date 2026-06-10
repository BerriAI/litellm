"""
Tests for Parallel AI Search API integration (v1 endpoint).
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm

MOCK_V1_RESPONSE = {
    "search_id": "search_abc123",
    "session_id": "session_xyz",
    "results": [
        {
            "url": "https://example.com/1",
            "title": "Test Result 1",
            "publish_date": "2026-01-15",
            "excerpts": ["First excerpt.", "Second excerpt."],
        },
        {
            "url": "https://example.com/2",
            "title": None,
            "publish_date": None,
            "excerpts": ["Only excerpt."],
        },
    ],
    "usage": [{"name": "search_advanced", "count": 1}],
}


def _mock_response():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = MOCK_V1_RESPONSE
    return mock_response


class TestParallelAISearch:
    @pytest.fixture(autouse=True)
    def _set_api_key(self, monkeypatch):
        monkeypatch.setenv("PARALLEL_API_KEY", "test-api-key")
        monkeypatch.delenv("PARALLEL_AI_API_BASE", raising=False)

    @pytest.mark.asyncio
    async def test_v1_endpoint_and_headers(self):
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post:
            mock_post.return_value = _mock_response()

            await litellm.asearch(
                query="latest developments in AI",
                search_provider="parallel_ai",
            )

            call_args = mock_post.call_args
            assert call_args.kwargs["url"] == "https://api.parallel.ai/v1/search"

            headers = call_args.kwargs.get("headers", {})
            assert headers["x-api-key"] == "test-api-key"
            assert headers["Content-Type"] == "application/json"
            assert "parallel-beta" not in headers

    @pytest.mark.asyncio
    async def test_string_query_maps_to_search_queries_and_objective(self):
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post:
            mock_post.return_value = _mock_response()

            await litellm.asearch(
                query="latest developments in AI",
                search_provider="parallel_ai",
            )

            json_data = mock_post.call_args.kwargs.get("json")
            assert json_data["search_queries"] == ["latest developments in AI"]
            assert json_data["objective"] == "latest developments in AI"

    @pytest.mark.asyncio
    async def test_list_query_maps_to_search_queries(self):
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post:
            mock_post.return_value = _mock_response()

            await litellm.asearch(
                query=["AI developments", "machine learning trends"],
                search_provider="parallel_ai",
            )

            json_data = mock_post.call_args.kwargs.get("json")
            assert json_data["search_queries"] == [
                "AI developments",
                "machine learning trends",
            ]
            assert "objective" not in json_data

    @pytest.mark.asyncio
    async def test_mode_param_passthrough(self):
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post:
            mock_post.return_value = _mock_response()

            await litellm.asearch(
                query="AI developments",
                search_provider="parallel_ai",
                mode="basic",
            )

            json_data = mock_post.call_args.kwargs.get("json")
            assert json_data["mode"] == "basic"

    @pytest.mark.asyncio
    async def test_legacy_processor_maps_to_mode(self):
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post:
            mock_post.return_value = _mock_response()

            await litellm.asearch(
                query="AI developments",
                search_provider="parallel_ai",
                processor="pro",
            )

            json_data = mock_post.call_args.kwargs.get("json")
            assert json_data["mode"] == "advanced"
            assert "processor" not in json_data

    @pytest.mark.asyncio
    async def test_optional_params_nest_under_advanced_settings(self):
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post:
            mock_post.return_value = _mock_response()

            await litellm.asearch(
                query="AI developments",
                search_provider="parallel_ai",
                max_results=5,
                country="US",
                search_domain_filter=["arxiv.org", "nature.com"],
                exclude_domains=["reddit.com"],
                max_chars_per_result=1500,
            )

            json_data = mock_post.call_args.kwargs.get("json")
            advanced_settings = json_data["advanced_settings"]
            assert advanced_settings["max_results"] == 5
            assert advanced_settings["location"] == "US"
            assert advanced_settings["source_policy"]["include_domains"] == [
                "arxiv.org",
                "nature.com",
            ]
            assert advanced_settings["source_policy"]["exclude_domains"] == [
                "reddit.com"
            ]
            assert advanced_settings["excerpt_settings"]["max_chars_per_result"] == 1500

            assert "max_results" not in json_data
            assert "source_policy" not in json_data
            assert "search_domain_filter" not in json_data
            assert "exclude_domains" not in json_data
            assert "max_chars_per_result" not in json_data
            assert "country" not in json_data

    @pytest.mark.asyncio
    async def test_explicit_advanced_settings_take_precedence(self):
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post:
            mock_post.return_value = _mock_response()

            await litellm.asearch(
                query="AI developments",
                search_provider="parallel_ai",
                max_results=5,
                advanced_settings={"max_results": 7},
            )

            json_data = mock_post.call_args.kwargs.get("json")
            assert json_data["advanced_settings"]["max_results"] == 7

    @pytest.mark.asyncio
    async def test_response_transformation(self):
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post:
            mock_post.return_value = _mock_response()

            response = await litellm.asearch(
                query="AI developments",
                search_provider="parallel_ai",
            )

            assert response.object == "search"
            assert len(response.results) == 2

            first = response.results[0]
            assert first.title == "Test Result 1"
            assert first.url == "https://example.com/1"
            assert first.snippet == "First excerpt. ... Second excerpt."
            assert first.date == "2026-01-15"

            second = response.results[1]
            assert second.title == ""
            assert second.snippet == "Only excerpt."
            assert second.date is None

    @pytest.mark.asyncio
    async def test_custom_api_base_appends_v1_search(self):
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post:
            mock_post.return_value = _mock_response()

            await litellm.asearch(
                query="AI developments",
                search_provider="parallel_ai",
                api_base="https://proxy.internal.example.com/",
            )

            call_args = mock_post.call_args
            assert (
                call_args.kwargs["url"]
                == "https://proxy.internal.example.com/v1/search"
            )

    @pytest.mark.asyncio
    async def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
        monkeypatch.delenv("PARALLEL_AI_API_KEY", raising=False)

        with pytest.raises(Exception, match="PARALLEL_API_KEY"):
            await litellm.asearch(
                query="AI developments",
                search_provider="parallel_ai",
            )
