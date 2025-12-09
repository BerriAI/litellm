import pytest
import litellm
import os
from typing import List, Union

from tests.search_tests.base_search_unit_tests import BaseSearchTest


class TestSearXNGSearch(BaseSearchTest):
    """
    Tests for SearXNG Search functionality.
    """
    
    def get_search_provider(self) -> str:
        """
        Return search_provider for SearXNG Search.
        """
        return "searxng"
    
    @pytest.mark.asyncio
    async def test_basic_search(self):
        """
        Test basic search functionality with a simple query.
        Override to handle free (0.0 cost) provider.
        """
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")
        litellm._turn_on_debug()
        search_provider = self.get_search_provider()
        print("Search Provider=", search_provider)

        try:
            response = await litellm.asearch(
                query="latest developments in AI",
                search_provider=search_provider,
            )
            print("Search response=", response.model_dump_json(indent=4))

            print(f"\n{'='*80}")
            print(f"Response type: {type(response)}")
            print(f"Response object: {response.object if hasattr(response, 'object') else 'N/A'}")
            
            # Check if response has expected Search format
            assert hasattr(response, "results"), "Response should have 'results' attribute"
            assert hasattr(response, "object"), "Response should have 'object' attribute"
            assert response.object == "search", f"Expected object='search', got '{response.object}'"
            
            # Validate results structure
            assert isinstance(response.results, list), "results should be a list"
            assert len(response.results) > 0, "Should have at least one result"
            
            # Check first result structure
            first_result = response.results[0]
            assert hasattr(first_result, "title"), "Result should have 'title' attribute"
            assert hasattr(first_result, "url"), "Result should have 'url' attribute"
            assert hasattr(first_result, "snippet"), "Result should have 'snippet' attribute"
            
            print(f"Total results: {len(response.results)}")
            print(f"First result title: {first_result.title}")
            print(f"First result URL: {first_result.url}")
            print(f"First result snippet: {first_result.snippet[:100]}...")
            print(f"{'='*80}\n")
            
            assert len(first_result.title) > 0, "Title should not be empty"
            assert len(first_result.url) > 0, "URL should not be empty"
            assert len(first_result.snippet) > 0, "Snippet should not be empty"
            
            # Validate cost tracking in _hidden_params
            # For SearXNG (free provider), cost can be None or 0.0
            assert hasattr(response, "_hidden_params"), "Response should have '_hidden_params' attribute"
            hidden_params = response._hidden_params
            assert "response_cost" in hidden_params, "_hidden_params should contain 'response_cost'"
            
            response_cost = hidden_params["response_cost"]
            # SearXNG is free, so cost can be None or 0.0
            if response_cost is not None:
                assert isinstance(response_cost, (int, float)), "response_cost should be a number"
                assert response_cost >= 0, "response_cost should be non-negative"
                print(f"Cost tracking: ${response_cost:.6f}")
            else:
                print(f"Cost tracking: Free (None)")
            
        except Exception as e:
            pytest.fail(f"Search call failed: {str(e)}")
    
    def test_search_with_optional_params(self):
        """
        Test search with optional parameters.
        Override for SearXNG since it doesn't natively limit results.
        """
        litellm.set_verbose = True
        search_provider = self.get_search_provider()

        response = litellm.search(
            query="machine learning",
            search_provider=search_provider,
            max_results=5,
        )

        # Validate response
        assert hasattr(response, "results"), "Response should have 'results' attribute"
        assert isinstance(response.results, list), "results should be a list"
        assert len(response.results) > 0, "Should have at least one result"
        # Note: SearXNG doesn't natively limit results, so we don't check <= 5
        
        print(f"\nSearch with optional params validated:")
        print(f"  - Requested max_results: 5")
        print(f"  - Received results: {len(response.results)}")

