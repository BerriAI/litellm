"""
Test that Azure AI Anthropic models have cache pricing configured.
Verifies the fix for issue #19532.
"""


import litellm
from litellm import get_model_info
from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map
import pytest


@pytest.fixture(autouse=True)
def reload_model_costs():
    """Reload model costs from JSON before each test."""
    litellm.model_cost = get_model_cost_map(url=None)
    yield


