import os
from unittest.mock import patch


import pytest

import litellm
from litellm import ModelResponse
from base_llm_unit_tests import BaseLLMChatTest, BaseOSeriesModelsTest


class TestOpenAIO1(BaseOSeriesModelsTest, BaseLLMChatTest):
    test_empty_tools = None
    test_tool_call_with_empty_enum_property = None
    test_tool_call_with_property_type_array = None

    def get_base_completion_call_args(self):
        return {
            "model": "o1",
        }

    def get_client(self):
        from openai import OpenAI

        return OpenAI(api_key="fake-api-key")

    def test_tool_call_no_arguments(self, tool_call_no_arguments):
        """Test that tool calls with no arguments is translated correctly. Relevant issue: https://github.com/BerriAI/litellm/issues/6833"""
        pass

    def test_prompt_caching(self):
        """Temporary override. o1 prompt caching is not working."""
        pass


class TestOpenAIO3(BaseOSeriesModelsTest, BaseLLMChatTest):
    test_basic_tool_calling = None
    test_function_calling_with_tool_response = None

    def get_base_completion_call_args(self):
        return {
            "model": "o3-mini",
        }

    def get_client(self):
        from openai import OpenAI

        return OpenAI(api_key="fake-api-key")

    def test_tool_call_no_arguments(self, tool_call_no_arguments):
        """Test that tool calls with no arguments is translated correctly. Relevant issue: https://github.com/BerriAI/litellm/issues/6833"""
        pass

    def test_prompt_caching(self):
        """Override, as o3 prompt caching is flaky"""
        pass


def test_o3_reasoning_effort():
    resp = litellm.completion(
        model="o3-mini",
        messages=[{"role": "user", "content": "Hello!"}],
        reasoning_effort="high",
    )
    assert resp.choices[0].message.content is not None
