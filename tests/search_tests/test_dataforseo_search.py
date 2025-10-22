"""
Unit tests for DataForSEO Search functionality.

These tests verify that the DataForSEO search provider integration works correctly
with LiteLLM's unified search interface.
"""

import sys
import os

# Add the parent directory to the path so we can import litellm
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tests.search_tests.base_search_unit_tests import BaseSearchTest


class TestDataForSEOSearch(BaseSearchTest):
    """
    Test suite for DataForSEO search provider.
    Inherits all test cases from BaseSearchTest.
    """

    def get_search_provider(self) -> str:
        """Return the search provider name for DataForSEO."""
        return "dataforseo"

