"""
Tests for Google Programmable Search Engine (PSE) API integration.
"""
import os
import sys
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)

from tests.search_tests.base_search_unit_tests import BaseSearchTest


class TestGooglePSESearch(BaseSearchTest):
    """
    Tests for Google PSE Search functionality.
    """
    
    def get_search_provider(self) -> str:
        """
        Return search_provider for Google PSE Search.
        """
        return "google_pse"


