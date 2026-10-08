"""
Test TogetherAI LLM
"""

from base_llm_unit_tests import BaseLLMChatTest
from tests._live_test_helpers import cheapest_together_chat_model
import json
from datetime import datetime
from unittest.mock import AsyncMock

import litellm
import pytest

class TestTogetherAI(BaseLLMChatTest):
    test_basic_tool_calling = None
    test_function_calling_with_tool_response = None

    def get_base_completion_call_args(self) -> dict:
        litellm.set_verbose = True
        return {"model": cheapest_together_chat_model(function_calling=True, response_schema=True)}
