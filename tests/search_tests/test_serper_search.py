"""
Tests for Serper Search API integration.
"""
import os
import sys
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

sys.path.insert(
    0, os.path.abspath("../..")
)

import litellm
from tests.search_tests.base_search_unit_tests import BaseSearchTest


@pytest.mark.skipif(
    not os.environ.get("SERPER_API_KEY"),
    reason="SERPER_API_KEY not set",
)
class TestSerperSearchBase(BaseSearchTest):
    """
    Tests for Serper Search functionality.
    """
    
    def get_search_provider(self) -> str:
        """
        Return search_provider for Serper Search.
        """
        return "serper"


class TestSerperSearch:
    """
    Tests for Serper Search functionality with mocked network responses.
    """
    
    @pytest.mark.asyncio
    async def test_serper_search_request_payload(self):
        """
        Test that validates the Serper search request payload structure without making real API calls.
        """
        # Set environment variable for API key
        os.environ["SERPER_API_KEY"] = "test-api-key"
        
        # Create a mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "organic": [
                {
                    "title": "Test Result 1",
                    "link": "https://example.com/1",
                    "snippet": "This is a test snippet for result 1",
                    "position": 1,
                },
                {
                    "title": "Test Result 2",
                    "link": "https://example.com/2",
                    "snippet": "This is a test snippet for result 2",
                    "position": 2,
                    "date": "Jan 15, 2025",
                },
            ],
        }
        
        # Mock the httpx AsyncClient post method
        with patch("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            
            # Make the search call
            response = await litellm.asearch(
                query="latest developments in AI",
                search_provider="serper",
                max_results=5
            )
            
            # Verify the post method was called once
            assert mock_post.call_count == 1
            
            # Get the actual call arguments
            call_args = mock_post.call_args
            
            # Verify URL
            assert call_args.kwargs["url"] == "https://google.serper.dev/search"
            
            # Verify headers contain X-API-KEY
            headers = call_args.kwargs.get("headers", {})
            assert "X-API-KEY" in headers
            assert headers["X-API-KEY"] == "test-api-key"
            assert headers["Content-Type"] == "application/json"
            
            # Verify request payload
            json_data = call_args.kwargs.get("json")
            assert json_data is not None
            assert json_data["q"] == "latest developments in AI"
            assert json_data["num"] == 5
            
            # Verify response structure
            assert hasattr(response, "results")
            assert hasattr(response, "object")
            assert response.object == "search"
            assert len(response.results) == 2
            
            # Verify first result
            first_result = response.results[0]
            assert first_result.title == "Test Result 1"
            assert first_result.url == "https://example.com/1"
            assert first_result.snippet == "This is a test snippet for result 1"
            
            # Verify date on second result
            second_result = response.results[1]
            assert second_result.date == "Jan 15, 2025"

    @pytest.mark.asyncio
    async def test_serper_search_with_country(self):
        """
        Test that country parameter is mapped to 'gl' in Serper request.
        """
        os.environ["SERPER_API_KEY"] = "test-api-key"
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "organic": [
                {
                    "title": "Result",
                    "link": "https://example.com",
                    "snippet": "Snippet",
                }
            ]
        }
        
        with patch("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            
            await litellm.asearch(
                query="test query",
                search_provider="serper",
                country="US",
            )
            
            json_data = mock_post.call_args.kwargs.get("json")
            assert json_data["gl"] == "us"

    @pytest.mark.asyncio
    async def test_serper_search_with_domain_filter(self):
        """
        Test that search_domain_filter is appended as site: clauses to the query.
        """
        os.environ["SERPER_API_KEY"] = "test-api-key"
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "organic": [
                {
                    "title": "Result",
                    "link": "https://arxiv.org/paper/1",
                    "snippet": "Snippet",
                }
            ]
        }
        
        with patch("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            
            await litellm.asearch(
                query="machine learning",
                search_provider="serper",
                search_domain_filter=["arxiv.org", "nature.com"],
            )
            
            json_data = mock_post.call_args.kwargs.get("json")
            assert "site:arxiv.org" in json_data["q"]
            assert "site:nature.com" in json_data["q"]
            assert "machine learning" in json_data["q"]

    @pytest.mark.asyncio
    async def test_serper_search_empty_organic(self):
        """
        Test handling of response with no organic results.
        """
        os.environ["SERPER_API_KEY"] = "test-api-key"
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "searchParameters": {"q": "xyznonexistent"},
        }
        
        with patch("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            
            response = await litellm.asearch(
                query="xyznonexistent",
                search_provider="serper",
            )
            
            assert response.object == "search"
            assert len(response.results) == 0


@pytest.mark.skipif(
    not os.environ.get("SERPER_API_KEY"),
    reason="SERPER_API_KEY not set",
)
class TestSerperSearchIntegration:
    """
    Live integration tests for Serper Search API.
    Requires SERPER_API_KEY environment variable to be set.
    """

    def test_serper_live_search(self):
        """
        Test a real search call against the Serper API.
        """
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        response = litellm.search(
            query="latest developments in AI",
            search_provider="serper",
            max_results=5,
        )

        # Validate response structure
        assert response.object == "search"
        assert isinstance(response.results, list)
        assert len(response.results) > 0
        assert len(response.results) <= 5

        # Validate first result has required fields
        first_result = response.results[0]
        assert len(first_result.title) > 0
        assert len(first_result.url) > 0
        assert len(first_result.snippet) > 0

        # Validate cost tracking
        assert hasattr(response, "_hidden_params")
        response_cost = response._hidden_params.get("response_cost")
        assert response_cost is not None
        assert response_cost >= 0

    def test_serper_live_search_with_max_results(self):
        """
        Test that max_results parameter limits the number of returned results.
        """
        response = litellm.search(
            query="python programming",
            search_provider="serper",
            max_results=3,
        )

        assert response.object == "search"
        assert len(response.results) > 0
        assert len(response.results) <= 3

        # Validate all results have valid URLs
        for result in response.results:
            assert result.url.startswith("http")

    def test_serper_live_search_with_country(self):
        """
        Test search with country filter returns results.
        """
        response = litellm.search(
            query="top news today",
            search_provider="serper",
            max_results=5,
            country="DE",
        )

        assert response.object == "search"
        assert isinstance(response.results, list)
        assert len(response.results) > 0

        # Validate results have required fields
        for result in response.results:
            assert len(result.title) > 0
            assert len(result.url) > 0
            assert len(result.snippet) > 0

    def test_serper_live_search_with_passthrough_params(self):
        """
        Test that Serper-specific parameters (tbs, hl) are passed through
        to the API and don't break the request.
        """
        response = litellm.search(
            query="OpenAI",
            search_provider="serper",
            max_results=3,
            tbs="qdr:w",  # Serper-specific: results from past week
            hl="en",      # Serper-specific: language hint
        )

        assert response.object == "search"
        assert len(response.results) > 0
        assert len(response.results) <= 3

        for result in response.results:
            assert result.url.startswith("http")
