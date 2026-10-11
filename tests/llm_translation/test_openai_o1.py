import os
from unittest.mock import patch

import pytest

import litellm
from litellm import ModelResponse
from base_llm_unit_tests import BaseLLMChatTest, BaseOSeriesModelsTest

class TestOpenAIO1(BaseOSeriesModelsTest, BaseLLMChatTest):

    def get_base_completion_call_args(self):
        return {
            "model": "o1",
        }

    def get_client(self):
        from openai import OpenAI

        return OpenAI(api_key="fake-api-key")

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

def test_o3_reasoning_effort():
    resp = litellm.completion(
        model="o3-mini",
        messages=[{"role": "user", "content": "Hello!"}],
        reasoning_effort="high",
    )
    assert resp.choices[0].message.content is not None
