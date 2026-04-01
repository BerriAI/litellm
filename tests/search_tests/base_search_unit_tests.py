"""
Base test class for Search functionality across different providers.

This follows the same pattern as BaseOCRTest in tests/ocr_tests/base_ocr_unit_tests.py
"""
import pytest
import litellm
from abc import ABC, abstractmethod
import os
import json


class BaseSearchTest(ABC):
    """
    Abstract base test class that enforces common Search tests across all providers.
    
    Each provider-specific test class should inherit from this and implement
    get_search_provider() to return provider name.
    """

    @abstractmethod
    def get_search_provider(self) -> str:
        """Must return the search_provider for the specific provider"""
        pass

    @pytest.fixture(autouse=True)
    def _handle_rate_limits(self):
        """Fixture to handle rate limit errors for all test methods"""
        try:
            yield
        except litellm.RateLimitError:
            pytest.skip("Rate limit exceeded")
        except litellm.InternalServerError:
            pytest.skip("Model is overloaded")

    @pytest.mark.asyncio
    async def test_basic_search(self):
        """
        Test basic search functionality with a simple query.
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
            assert hasattr(response, "_hidden_params"), "Response should have '_hidden_params' attribute"
            hidden_params = response._hidden_params
            assert "response_cost" in hidden_params, "_hidden_params should contain 'response_cost'"
            
            response_cost = hidden_params["response_cost"]
            assert response_cost is not None, "response_cost should not be None"
            assert isinstance(response_cost, (int, float)), "response_cost should be a number"
            assert response_cost >= 0, "response_cost should be non-negative"
            
            print(f"Cost tracking: ${response_cost:.6f}")
            
        except Exception as e:
            pytest.fail(f"Search call failed: {str(e)}")

    def test_search_response_structure(self):
        """
        Test that the Search response has the correct structure.
        """
        litellm.set_verbose = True
        search_provider = self.get_search_provider()

        response = litellm.search(
            query="artificial intelligence recent news",
            search_provider=search_provider,
        )

        # Validate response structure
        assert hasattr(response, "results"), "Response should have 'results' attribute"
        assert hasattr(response, "object"), "Response should have 'object' attribute"
        
        assert isinstance(response.results, list), "results should be a list"
        assert len(response.results) > 0, "Should have at least one result"
        assert response.object == "search", "object should be 'search'"
        
        # Validate first result structure
        first_result = response.results[0]
        assert hasattr(first_result, "title"), "Result should have 'title' attribute"
        assert hasattr(first_result, "url"), "Result should have 'url' attribute"
        assert hasattr(first_result, "snippet"), "Result should have 'snippet' attribute"
        assert isinstance(first_result.title, str), "title should be a string"
        assert isinstance(first_result.url, str), "url should be a string"
        assert isinstance(first_result.snippet, str), "snippet should be a string"
        
        print(f"\nResponse structure validated:")
        print(f"  - object: {response.object}")
        print(f"  - results: {len(response.results)}")
        print(f"  - first result has all required fields")

    def test_search_with_optional_params(self):
        """
        Test search with optional parameters.
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
        assert len(response.results) <= 5, "Should have at most 5 results as requested"
        
        print(f"\nSearch with optional params validated:")
        print(f"  - Requested max_results: 5")
        print(f"  - Received results: {len(response.results)}")

