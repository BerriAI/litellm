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
    
    def get_custom_llm_provider(self) -> str:
        """
        Return custom_llm_provider for Perplexity Search.
        """
        return "perplexity"

