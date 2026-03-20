"""
Tests for Brave Search API integration.
"""

import os
import pytest
from urllib.parse import urlparse, parse_qs
from unittest.mock import AsyncMock, patch, MagicMock

import litellm
from tests.search_tests.base_search_unit_tests import BaseSearchTest

@pytest.mark.skip(reason="Not yet implemented")
class TestBraveSearch(BaseSearchTest):
    """
    Tests for Brave Search functionality with mocked network responses.
    """

    def get_search_provider(self) -> str:
        """Return the search provider name"""
        return "brave"

    @pytest.mark.asyncio
    async def test_basic_search(self):
        """
        Test basic search functionality with a simple query.
        """
        os.environ["BRAVE_API_KEY"] = "test-api-key"

        # Create a mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {
                        "title": "Test Result 1",
                        "url": "https://example.com/1",
                        "description": "This is a test snippet for result 1",
                    }
                ]
            }
        }

        # Mock the httpx AsyncClient get method
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.get",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = mock_response

            # Make the search call
            response = await litellm.asearch(
                query="Brave browser features",
                search_provider="brave",
                max_results=5,
                result_filter="web",
            )

            # Verify the get method was called once
            assert mock_get.call_count == 1

            # Get the actual call arguments
            call_args = mock_get.call_args

            # Verify URL (include_fetch_metadata=True is added by default)
            parsed_url = urlparse(call_args.kwargs["url"])
            assert parsed_url.scheme == "https"
            assert parsed_url.netloc == "api.search.brave.com"
            assert parsed_url.path == "/res/v1/web/search"

            query_params = parse_qs(parsed_url.query)
            assert query_params == {
                "q": ["Brave browser features"],
                "include_fetch_metadata": ["True"],
                "count": ["5"],
                "result_filter": ["web"],
            }

            # Verify headers contains X-Subscription-Token
            headers = call_args.kwargs.get("headers", {})
            assert "X-Subscription-Token" in headers
            assert headers["X-Subscription-Token"] == "test-api-key"

            # Note: Brave uses GET requests, so parameters are in the URL, not in JSON body
            # The URL already contains all the parameters we need to verify

            # Verify response structure
            assert hasattr(response, "results")
            assert hasattr(response, "object")
            assert response.object == "search"
            assert len(response.results) == 1

            # Verify first result
            first_result = response.results[0]
            assert first_result.title == "Test Result 1"
            assert first_result.url == "https://example.com/1"
            assert first_result.snippet == "This is a test snippet for result 1"
