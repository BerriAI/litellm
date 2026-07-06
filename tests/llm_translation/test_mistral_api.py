import asyncio
import os
import sys
import traceback

from dotenv import load_dotenv

import litellm.types
import litellm.types.utils
from litellm.llms.anthropic.chat import ModelResponseIterator

load_dotenv()
import io
import os

sys.path.insert(0, os.path.abspath("../.."))  # Adds the parent directory to the system path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

import litellm

from litellm.llms.anthropic.common_utils import process_anthropic_headers
from httpx import Headers
from base_llm_unit_tests import BaseLLMChatTest


@pytest.mark.flaky(retries=3, delay=2)
class TestMistralCompletion(BaseLLMChatTest):
    def get_base_completion_call_args(self) -> dict:
        litellm.set_verbose = True
        return {"model": "mistral/mistral-medium-latest"}

    def test_tool_call_no_arguments(self, tool_call_no_arguments):
        """Test that tool calls with no arguments is translated correctly. Relevant issue: https://github.com/BerriAI/litellm/issues/6833"""
        pass

    def test_web_search(self):
        """Web search is routed to the Conversations API for models that support it"""
        from litellm.utils import supports_web_search

        if not os.getenv("MISTRAL_API_KEY"):
            pytest.skip("MISTRAL_API_KEY not set")

        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        litellm._turn_on_debug()

        model = "mistral/mistral-medium-latest"

        if not supports_web_search(model, None):
            pytest.skip("Model does not support web search")

        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "What's the weather like in Boston today?"}],
            web_search_options={},
            max_tokens=100,
        )

        assert response is not None

    def test_web_search_with_tool_call_history(self):
        """The Conversations API accepts prior function-call history mapped to
        function.call / function.result input entries alongside a web search
        request (a 400 here means the mapped wire schema is wrong)."""
        if not os.getenv("MISTRAL_API_KEY"):
            pytest.skip("MISTRAL_API_KEY not set")

        response = litellm.completion(
            model="mistral/mistral-medium-latest",
            messages=[
                {"role": "user", "content": "What's the weather in Paris?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "18C and sunny"},
                {"role": "user", "content": "Now search the web for who won Euro 2024 and cite sources."},
            ],
            web_search_options={},
            max_tokens=200,
        )

        assert response is not None
        assert response.choices[0].message.content
