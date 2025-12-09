import pytest
import litellm
from typing import List, Union

from tests.search_tests.base_search_unit_tests import BaseSearchTest


class TestParallelAISearch(BaseSearchTest):
    """
    Tests for Parallel AI Search functionality.
    """
    
    def get_search_provider(self) -> str:
        """
        Return search_provider for Parallel AI Search.
        """
        return "parallel_ai"


