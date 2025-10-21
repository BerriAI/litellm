import pytest
import litellm
from typing import List, Union

from tests.search_tests.base_search_unit_tests import BaseSearchTest


class TestExaAISearch(BaseSearchTest):
    """
    Tests for Exa AI Search functionality.
    """
    
    def get_custom_llm_provider(self) -> str:
        """
        Return custom_llm_provider for Exa AI Search.
        """
        return "exa_ai"

