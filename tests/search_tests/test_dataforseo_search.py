"""
Unit tests for DataForSEO Search functionality.
"""

import sys
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import litellm
from litellm.llms.base_llm.search.transformation import SearchResponse, SearchResult


@pytest.mark.asyncio
async def test_dataforseo_search_basic():
    """
    Test DataForSEO search with mocked network response.
    """
    os.environ["DATAFORSEO_LOGIN"] = "test_login"
    os.environ["DATAFORSEO_PASSWORD"] = "test_password"
    
    mock_response = SearchResponse(
        object="search",
        results=[
            SearchResult(
                title="Latest AI Developments in 2025",
                url="https://example.com/ai-news",
                snippet="Recent advances in artificial intelligence have shown remarkable progress in machine learning and neural networks."
            ),
            SearchResult(
                title="AI Research Breakthroughs",
                url="https://example.com/ai-research",
                snippet="Scientists announce breakthrough in AI technology with new models achieving unprecedented accuracy."
            )
        ]
    )
    
    with patch("litellm.llms.custom_httpx.llm_http_handler.BaseLLMHTTPHandler.async_search", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = mock_response
        
        response = await litellm.asearch(
            query="latest developments in AI",
            search_provider="dataforseo",
        )
        
        assert response.object == "search"
        assert len(response.results) == 2
        assert response.results[0].title == "Latest AI Developments in 2025"
        assert response.results[0].url == "https://example.com/ai-news"
        assert len(response.results[0].snippet) > 0
