"""
Tests for Tavily Search API integration.
"""
import os
import sys
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)

from tests.search_tests.base_search_unit_tests import BaseSearchTest


class TestTavilySearch(BaseSearchTest):
    """
    Tests for Tavily Search functionality.
    """
    
    def get_search_provider(self) -> str:
        """
        Return search_provider for Tavily Search.
        """
        return "tavily"

