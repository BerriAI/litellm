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
    
    def get_custom_llm_provider(self) -> str:
        """
        Return custom_llm_provider for Tavily Search.
        """
        return "tavily"

