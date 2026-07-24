"""
Tests for You.com Search API integration.
"""

import os
import sys
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

sys.path.insert(0, os.path.abspath("../.."))

import litellm


class TestYouComSearch:
    """
    Tests for You.com Search functionality with mocked network responses.
    """

    @pytest.fixture(autouse=True)
    def _set_api_key(self, monkeypatch):
        """
        Default fixture: YOUCOM_API_KEY is set, scoped to this test.
        Tests that need the key absent should call `monkeypatch.delenv` themselves.
        """
        monkeypatch.setenv("YOUCOM_API_KEY", "test-api-key")

    @pytest.mark.asyncio
    async def test_you_com_search_request_payload(self):
        """
        Validate the You.com search request payload structure without real API calls.
        """
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": {
                "web": [
                    {
                        "title": "Test Result 1",
                        "url": "https://example.com/1",
                        "description": "Brief description 1",
                        "snippets": ["This is a test snippet for result 1"],
                        "page_age": "2025-01-15T00:00:00Z",
                    },
                    {
                        "title": "Test Result 2",
                        "url": "https://example.com/2",
                        "description": "Brief description 2",
                        "snippets": ["This is a test snippet for result 2"],
                        "page_age": "2025-01-10T00:00:00Z",
                    },
                ],
                "news": [],
            },
            "metadata": {
                "search_uuid": "abc-123",
                "query": "latest developments in AI",
                "latency": 0.42,
            },
        }

        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post:
            mock_post.return_value = mock_response

            response = await litellm.asearch(
                query="latest developments in AI",
                search_provider="you_com",
                max_results=5,
            )

            assert mock_post.call_count == 1
            call_args = mock_post.call_args

            assert call_args.kwargs["url"] == "https://ydc-index.io/v1/search"

            headers = call_args.kwargs.get("headers", {})
            assert "X-API-Key" in headers
            assert headers["X-API-Key"] == "test-api-key"
            assert headers["Content-Type"] == "application/json"

            json_data = call_args.kwargs.get("json")
            assert json_data is not None
            assert json_data["query"] == "latest developments in AI"
            # max_results is mapped to You.com's `count` parameter
            assert json_data["count"] == 5

            assert hasattr(response, "results")
            assert hasattr(response, "object")
            assert response.object == "search"
            assert len(response.results) == 2

            first_result = response.results[0]
            assert first_result.title == "Test Result 1"
            assert first_result.url == "https://example.com/1"
            assert first_result.snippet == "This is a test snippet for result 1"
            assert first_result.date == "2025-01-15T00:00:00Z"

    @pytest.mark.asyncio
    async def test_you_com_search_domain_filter_and_country(self):
        """
        Validate that Perplexity-spec optional params map to You.com's parameters:
        - search_domain_filter -> include_domains
        - country              -> country (lowercased to match Tavily's convention)
        """
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": {"web": [], "news": []},
            "metadata": {},
        }

        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post:
            mock_post.return_value = mock_response

            await litellm.asearch(
                query="machine learning",
                search_provider="you_com",
                search_domain_filter=["arxiv.org", "nature.com"],
                country="US",
            )

            call_args = mock_post.call_args
            json_data = call_args.kwargs.get("json")

            assert json_data["query"] == "machine learning"
            assert json_data["include_domains"] == ["arxiv.org", "nature.com"]
            # Country is normalized to lowercase, matching Tavily's behavior.
            assert json_data["country"] == "us"
            # search_domain_filter and max_tokens_per_page (perplexity-spec names)
            # should NOT leak through to the upstream payload.
            assert "search_domain_filter" not in json_data
            assert "max_tokens_per_page" not in json_data

    @pytest.mark.asyncio
    async def test_you_com_search_snippet_fallback_to_description(self):
        """
        When `snippets` is missing/empty, snippet falls back to `description`.
        """
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": {
                "web": [
                    {
                        "title": "No snippets here",
                        "url": "https://example.com/3",
                        "description": "Fallback description text",
                        "snippets": [],
                        "page_age": None,
                    }
                ],
                "news": [],
            },
            "metadata": {},
        }

        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post:
            mock_post.return_value = mock_response

            response = await litellm.asearch(
                query="anything",
                search_provider="you_com",
            )

            assert len(response.results) == 1
            assert response.results[0].snippet == "Fallback description text"
            assert response.results[0].date is None

    @pytest.mark.asyncio
    async def test_you_com_search_news_results_appended(self):
        """
        News results are flattened in after web results.
        """
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": {
                "web": [
                    {
                        "title": "Web Result",
                        "url": "https://example.com/web",
                        "snippets": ["web snippet"],
                        "description": "web desc",
                        "page_age": "2025-01-01T00:00:00Z",
                    }
                ],
                "news": [
                    {
                        "title": "News Result",
                        "url": "https://news.example.com/article",
                        "description": "news desc",
                        "page_age": "2025-02-01T00:00:00Z",
                    }
                ],
            },
            "metadata": {},
        }

        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post:
            mock_post.return_value = mock_response

            response = await litellm.asearch(
                query="anything",
                search_provider="you_com",
            )

            assert len(response.results) == 2
            assert response.results[0].title == "Web Result"
            assert response.results[1].title == "News Result"
            # News result has no `snippets` -> falls back to description
            assert response.results[1].snippet == "news desc"

    def test_you_com_search_complete_url_handles_trailing_slash(self):
        """
        get_complete_url must normalize trailing slashes on api_base, so a custom
        base like `https://x.example/v1/search/` does not become
        `https://x.example/v1/search/v1/search`.
        """
        from litellm.llms.you_com.search.transformation import YouComSearchConfig

        config = YouComSearchConfig()
        assert (
            config.get_complete_url(
                api_base="https://x.example/v1/search/", optional_params={}
            )
            == "https://x.example/v1/search"
        )
        assert (
            config.get_complete_url(api_base="https://x.example/", optional_params={})
            == "https://x.example/v1/search"
        )
        # With an API key configured, default base is the keyed endpoint.
        assert (
            config.get_complete_url(api_base=None, optional_params={})
            == "https://ydc-index.io/v1/search"
        )

    @pytest.mark.asyncio
    async def test_you_com_search_keyless_free_tier(self, monkeypatch):
        """
        Without YOUCOM_API_KEY, the adapter targets the keyless free-tier
        endpoint and sends no X-API-Key header.
        """
        monkeypatch.delenv("YOUCOM_API_KEY", raising=False)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": {
                "web": [
                    {
                        "title": "Keyless Result",
                        "url": "https://example.com/keyless",
                        "snippets": ["snippet from keyless tier"],
                        "description": "desc",
                        "page_age": "2025-03-01T00:00:00Z",
                    }
                ],
                "news": [],
            },
            "metadata": {},
        }

        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post:
            mock_post.return_value = mock_response

            response = await litellm.asearch(
                query="hello world",
                search_provider="you_com",
            )

            call_args = mock_post.call_args
            assert call_args.kwargs["url"] == "https://api.you.com/v1/agents/search"
            headers = call_args.kwargs.get("headers", {})
            assert "X-API-Key" not in headers
            assert headers["Content-Type"] == "application/json"

            assert len(response.results) == 1
            assert response.results[0].title == "Keyless Result"

    @pytest.mark.asyncio
    async def test_you_com_search_programmatic_api_key_selects_keyed_endpoint(
        self, monkeypatch
    ):
        """
        When the key is passed programmatically (no YOUCOM_API_KEY in the env),
        the keyed endpoint must be selected and the X-API-Key header sent, instead
        of silently falling back to the keyless free tier.
        """
        monkeypatch.delenv("YOUCOM_API_KEY", raising=False)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": {"web": [], "news": []},
            "metadata": {},
        }

        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post:
            mock_post.return_value = mock_response

            await litellm.asearch(
                query="anything",
                search_provider="you_com",
                api_key="my-programmatic-key",
            )

            call_args = mock_post.call_args
            assert call_args.kwargs["url"] == "https://ydc-index.io/v1/search"
            headers = call_args.kwargs.get("headers", {})
            assert headers["X-API-Key"] == "my-programmatic-key"

    def test_you_com_search_complete_url_uses_programmatic_api_key(self, monkeypatch):
        """
        get_complete_url selects the keyed endpoint from a forwarded api_key even
        when YOUCOM_API_KEY is absent from the environment.
        """
        monkeypatch.delenv("YOUCOM_API_KEY", raising=False)

        from litellm.llms.you_com.search.transformation import YouComSearchConfig

        config = YouComSearchConfig()
        assert (
            config.get_complete_url(
                api_base=None, optional_params={}, api_key="my-programmatic-key"
            )
            == "https://ydc-index.io/v1/search"
        )
        assert (
            config.get_complete_url(api_base=None, optional_params={}, api_key=None)
            == "https://api.you.com/v1/agents/search"
        )

    def test_you_com_search_validate_environment_keyless(self, monkeypatch):
        """
        validate_environment must NOT raise when no key is configured —
        the keyless free tier is the default behavior.
        """
        monkeypatch.delenv("YOUCOM_API_KEY", raising=False)

        from litellm.llms.you_com.search.transformation import YouComSearchConfig

        config = YouComSearchConfig()
        headers = config.validate_environment(headers={}, api_key=None)
        assert "X-API-Key" not in headers
        assert headers["Content-Type"] == "application/json"

    def test_you_com_search_pins_identity_accept_encoding(self, monkeypatch):
        """
        The adapter pins Accept-Encoding: identity to work around the keyless
        endpoint advertising gzip content-encoding while returning bytes httpx
        can't decode. Without this, every keyless request raises DecodingError.
        """
        monkeypatch.delenv("YOUCOM_API_KEY", raising=False)

        from litellm.llms.you_com.search.transformation import YouComSearchConfig

        config = YouComSearchConfig()
        headers = config.validate_environment(headers={}, api_key=None)
        assert headers["Accept-Encoding"] == "identity"

        # setdefault: a caller-supplied Accept-Encoding should win
        headers = config.validate_environment(
            headers={"Accept-Encoding": "gzip"}, api_key=None
        )
        assert headers["Accept-Encoding"] == "gzip"
