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
