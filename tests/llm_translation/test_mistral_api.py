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

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

import litellm

from litellm.llms.anthropic.common_utils import process_anthropic_headers
from httpx import Headers
from base_llm_unit_tests import BaseLLMChatTest


class TestMistralCompletion(BaseLLMChatTest):
    def get_base_completion_call_args(self) -> dict:
        litellm.set_verbose = True
        return {"model": "mistral/mistral-small-latest"}

    def test_tool_call_no_arguments(self, tool_call_no_arguments):
        """Test that tool calls with no arguments is translated correctly. Relevant issue: https://github.com/BerriAI/litellm/issues/6833"""
        pass

    def test_multilingual_requests(self):
        """
        Mistral API raises a 400 BadRequest error when the request contains invalid utf-8 sequences.
        """
        pass

    def test_mistral_max_completion_tokens(self):
        """
        Test that max_completion_tokens is properly converted to max_tokens for Mistral API
        """
        # Test the actual conversion directly
        # We'll use a real instance of MistralConfig to test the conversion
        config = litellm.llms.mistral.mistral_chat_transformation.MistralConfig()
        result = config.map_openai_params(
            non_default_params={"max_completion_tokens": 100},
            optional_params={},
            model="mistral/mistral-small-latest",
            drop_params=False
        )
        
        # Verify that max_tokens is set correctly in the result
        assert "max_tokens" in result
        assert result["max_tokens"] == 100
        
        # Test with both max_tokens and max_completion_tokens
        # max_completion_tokens should take priority
        result = config.map_openai_params(
            non_default_params={"max_tokens": 50, "max_completion_tokens": 100},
            optional_params={},
            model="mistral/mistral-small-latest",
            drop_params=False
        )
        
        # Verify that max_tokens is set correctly in the result
        assert "max_tokens" in result
        assert result["max_tokens"] == 100
