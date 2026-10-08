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
    test_empty_tools = None
    test_function_calling_with_tool_response = None
    test_json_response_format = None
    test_json_response_nested_json_schema = None
    test_json_response_nested_pydantic_obj = None
    test_json_response_pydantic_obj = None
    test_tool_call_with_empty_enum_property = None
    test_tool_call_with_property_type_array = None

    def get_base_completion_call_args(self) -> dict:
        litellm.set_verbose = True
        return {"model": cheapest_together_chat_model(function_calling=True, response_schema=True)}
