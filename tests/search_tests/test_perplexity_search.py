"""
Tests for Perplexity Search API integration.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath("../.."))

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


# Substrings that indicate the failure is caused by the CI provider account
# (billing, quota, missing scope) rather than a LiteLLM regression. Perplexity
# in particular returns quota exhaustion as a 401 AuthenticationError, which
# would otherwise fail this test.
_ENV_PROVIDER_ERROR_SUBSTRINGS = (
    "missing scopes",
    "insufficient permissions",
    "insufficient_quota",
    "credit balance is too low",
    "exceeded your current quota",
    "billing",
)


def _is_env_provider_error(exc: Exception) -> bool:
    """Return True when `exc` was raised by a provider for an env/account reason."""
    message = str(exc) if exc else ""
    if not message:
        return False
    lowered = message.lower()
    return any(needle in lowered for needle in _ENV_PROVIDER_ERROR_SUBSTRINGS)


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
                    },
                }
            ]
        )

        # Test the search. Perplexity can return transient upstream conditions
        # (rate-limit, overloaded, quota-as-401) that are CI account-state
        # issues rather than LiteLLM regressions. Mirror the behavior of
        # BaseSearchTest._handle_rate_limits (which this class does not inherit)
        # and `pytest.skip` in those narrow cases. Any other exception still
        # propagates normally.
        try:
            response = await router.asearch(
                query="latest AI developments",
                search_tool_name="litellm-search",
                max_results=3,
            )
        except litellm.RateLimitError:
            pytest.skip("Rate limit exceeded")
        except litellm.InternalServerError:
            pytest.skip("Model is overloaded")
        except litellm.AuthenticationError as e:
            if _is_env_provider_error(e):
                pytest.skip(f"Skipping due to provider env/account condition: {e}")
            raise

        print(f"\n{'='*80}")
        print(f"Router Search Test Results:")
        print(f"Response type: {type(response)}")
        print(f"Response object: {response.object}")
        print(f"Number of results: {len(response.results)}")

        # Validate response structure
        assert hasattr(response, "results"), "Response should have 'results' attribute"
        assert hasattr(response, "object"), "Response should have 'object' attribute"
        assert (
            response.object == "search"
        ), f"Expected object='search', got '{response.object}'"
        assert isinstance(response.results, list), "results should be a list"
        assert len(response.results) > 0, "Should have at least one result"
        assert len(response.results) <= 3, "Should return at most 3 results"

        # Validate first result
        first_result = response.results[0]
        assert hasattr(first_result, "title"), "Result should have 'title' attribute"
        assert hasattr(first_result, "url"), "Result should have 'url' attribute"
        assert hasattr(
            first_result, "snippet"
        ), "Result should have 'snippet' attribute"

        print(f"First result title: {first_result.title}")
        print(f"First result URL: {first_result.url}")
        print(f"{'='*80}\n")

        print("✅ Router search test passed!")
