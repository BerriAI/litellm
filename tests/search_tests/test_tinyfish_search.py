"""
Tests for TinyFish Search API integration.
"""

import os
import pytest
from urllib.parse import urlparse, parse_qs
from unittest.mock import AsyncMock, MagicMock, patch

import litellm


MOCK_TINYFISH_RESPONSE = {
    "query": "web automation tools",
    "results": [
        {
            "position": 1,
            "site_name": "tinyfish.ai",
            "title": "TinyFish — AI Web Automation",
            "snippet": "Automate any website with natural language.",
            "url": "https://tinyfish.ai",
        },
        {
            "position": 2,
            "site_name": "github.com",
            "title": "Top Web Automation Tools",
            "snippet": "A curated list of browser automation frameworks.",
            "url": "https://github.com/example/web-automation",
        },
    ],
    "total_results": 2,
    "page": 0,
}


def _make_mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_data
    return mock


class TestTinyfishSearch:
    @pytest.mark.asyncio
    async def test_basic_search(self):
        os.environ["TINYFISH_API_KEY"] = "sk-tinyfish-test"

        mock_response = _make_mock_response(MOCK_TINYFISH_RESPONSE)

        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.get",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_response

            response = await litellm.asearch(
                query="web automation tools",
                search_provider="tinyfish",
            )

            assert mock_get.call_count == 1

            call_args = mock_get.call_args
            parsed_url = urlparse(call_args.kwargs["url"])
            assert parsed_url.scheme == "https"
            assert parsed_url.netloc == "api.search.tinyfish.ai"
            assert parsed_url.path == ""

            query_params = parse_qs(parsed_url.query)
            assert query_params["query"] == ["web automation tools"]

            headers = call_args.kwargs.get("headers", {})
            assert headers["X-API-Key"] == "sk-tinyfish-test"

            assert hasattr(response, "results")
            assert response.object == "search"
            assert len(response.results) == 2

            first = response.results[0]
            assert first.title == "TinyFish — AI Web Automation"
            assert first.url == "https://tinyfish.ai"
            assert first.snippet == "Automate any website with natural language."

    @pytest.mark.asyncio
    async def test_country_maps_to_location(self):
        os.environ["TINYFISH_API_KEY"] = "sk-tinyfish-test"

        mock_response = _make_mock_response(MOCK_TINYFISH_RESPONSE)

        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.get",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_response

            await litellm.asearch(
                query="test",
                search_provider="tinyfish",
                country="US",
            )

            call_args = mock_get.call_args
            parsed_url = urlparse(call_args.kwargs["url"])
            query_params = parse_qs(parsed_url.query)
            assert query_params["location"] == ["US"]

    @pytest.mark.asyncio
    async def test_domain_filter_injection(self):
        os.environ["TINYFISH_API_KEY"] = "sk-tinyfish-test"

        mock_response = _make_mock_response(MOCK_TINYFISH_RESPONSE)

        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.get",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_response

            await litellm.asearch(
                query="python tutorials",
                search_provider="tinyfish",
                search_domain_filter=["arxiv.org", "github.com"],
            )

            call_args = mock_get.call_args
            parsed_url = urlparse(call_args.kwargs["url"])
            query_params = parse_qs(parsed_url.query)
            query_value = query_params["query"][0]
            assert "site:arxiv.org" in query_value
            assert "site:github.com" in query_value
            assert "python tutorials" in query_value

    @pytest.mark.asyncio
    async def test_language_passthrough(self):
        os.environ["TINYFISH_API_KEY"] = "sk-tinyfish-test"

        mock_response = _make_mock_response(MOCK_TINYFISH_RESPONSE)

        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.get",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_response

            await litellm.asearch(
                query="test",
                search_provider="tinyfish",
                language="en",
            )

            call_args = mock_get.call_args
            parsed_url = urlparse(call_args.kwargs["url"])
            query_params = parse_qs(parsed_url.query)
            assert query_params["language"] == ["en"]

    @pytest.mark.asyncio
    async def test_empty_results(self):
        os.environ["TINYFISH_API_KEY"] = "sk-tinyfish-test"

        empty_response = {
            "query": "xyznonexistent",
            "results": [],
            "total_results": 0,
            "page": 0,
        }
        mock_response = _make_mock_response(empty_response)

        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.get",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_response

            response = await litellm.asearch(
                query="xyznonexistent",
                search_provider="tinyfish",
            )

            assert response.object == "search"
            assert len(response.results) == 0

    def test_missing_api_key(self):
        os.environ.pop("TINYFISH_API_KEY", None)

        from litellm.llms.tinyfish.search.transformation import TinyfishSearchConfig

        config = TinyfishSearchConfig()
        with pytest.raises(ValueError, match="TINYFISH_API_KEY"):
            config.validate_environment(headers={})
