"""
Tests for Perplexity Search API integration.
"""
import os
import sys
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)

from tests.search_tests.base_search_unit_tests import BaseSearchTest


class TestPerplexitySearch(BaseSearchTest):
    """
    Tests for Perplexity Search functionality.
    """
    
    def get_search_provider(self) -> str:
        """
        Return search_provider for Perplexity Search.
        """
        return "perplexity"


class TestRouterSearch:
    """
    Tests for Router Search functionality.
    """
    
    @pytest.mark.asyncio
    async def test_router_search_with_search_tools(self):
        """
        Test router's asearch method with search_tools configuration.
        """
        from litellm import Router
        import litellm
        
        litellm._turn_on_debug()
        
        # Create router with search_tools config
        router = Router(
            search_tools=[
                {
                    "search_tool_name": "litellm-search",
                    "litellm_params": {
                        "search_provider": "perplexity",
                        "api_key": os.environ.get("PERPLEXITYAI_API_KEY"),
                    }
                }
            ]
        )
        
        # Test the search
        response = await router.asearch(
            query="latest AI developments",
            search_tool_name="litellm-search",
            max_results=3
        )
        
        print(f"\n{'='*80}")
        print(f"Router Search Test Results:")
        print(f"Response type: {type(response)}")
        print(f"Response object: {response.object}")
        print(f"Number of results: {len(response.results)}")
        
        # Validate response structure
        assert hasattr(response, "results"), "Response should have 'results' attribute"
        assert hasattr(response, "object"), "Response should have 'object' attribute"
        assert response.object == "search", f"Expected object='search', got '{response.object}'"
        assert isinstance(response.results, list), "results should be a list"
        assert len(response.results) > 0, "Should have at least one result"
        assert len(response.results) <= 3, "Should return at most 3 results"
        
        # Validate first result
        first_result = response.results[0]
        assert hasattr(first_result, "title"), "Result should have 'title' attribute"
        assert hasattr(first_result, "url"), "Result should have 'url' attribute"
        assert hasattr(first_result, "snippet"), "Result should have 'snippet' attribute"
        
        print(f"First result title: {first_result.title}")
        print(f"First result URL: {first_result.url}")
        print(f"{'='*80}\n")
        
        print("✅ Router search test passed!")

