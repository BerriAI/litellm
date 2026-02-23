"""
Tests for Tavily Search API integration.
"""
import os
import sys
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

sys.path.insert(
    0, os.path.abspath("../..")
)

import litellm


class TestTavilySearch:
    """
    Tests for Tavily Search functionality with mocked network responses.
    """
    
    @pytest.mark.asyncio
    async def test_tavily_search_request_payload(self):
        """
        Test that validates the Tavily search request payload structure without making real API calls.
        """
        # Set environment variable for API key
        os.environ["TAVILY_API_KEY"] = "test-api-key"
        
        # Create a mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "title": "Test Result 1",
                    "url": "https://example.com/1",
                    "content": "This is a test snippet for result 1"
                },
                {
                    "title": "Test Result 2",
                    "url": "https://example.com/2",
                    "content": "This is a test snippet for result 2"
                }
            ]
        }
        
        # Mock the httpx AsyncClient post method
        with patch("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            
            # Make the search call
            response = await litellm.asearch(
                query="latest developments in AI",
                search_provider="tavily",
                max_results=5
            )
            
            # Verify the post method was called once
            assert mock_post.call_count == 1
            
            # Get the actual call arguments
            call_args = mock_post.call_args
            
            # Verify URL
            assert call_args.kwargs["url"] == "https://api.tavily.com/search"
            
            # Verify headers contain Authorization
            headers = call_args.kwargs.get("headers", {})
            assert "Authorization" in headers
            assert headers["Authorization"] == "Bearer test-api-key"
            assert headers["Content-Type"] == "application/json"
            
            # Verify request payload
            json_data = call_args.kwargs.get("json")
            assert json_data is not None
            assert json_data["query"] == "latest developments in AI"
            assert json_data["max_results"] == 5
            
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

