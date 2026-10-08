"""
Tests for SearchAPI.io (Google Search) integration.
"""

import os

import pytest


@pytest.mark.skipif(
    os.environ.get("SEARCHAPI_API_KEY") is None,
    reason="SEARCHAPI_API_KEY not set in environment",
)
class TestSearchAPIIntegration:
    """Integration tests for SearchAPI.io (requires API key)."""

    def test_real_search_request(self):
        """
        Test a real search request to SearchAPI.io.
        This test is skipped if SEARCHAPI_API_KEY is not set.
        """
        import litellm

        response = litellm.search(query="Python programming", search_provider="searchapi", max_results=5)

        assert response is not None
        assert hasattr(response, "results")
        assert len(response.results) > 0
        assert all(hasattr(r, "title") for r in response.results)
        assert all(hasattr(r, "url") for r in response.results)
        assert all(hasattr(r, "snippet") for r in response.results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
