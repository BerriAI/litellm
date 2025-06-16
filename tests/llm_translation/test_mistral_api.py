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
        return {"model": "mistral/mistral-medium-latest"}

    def test_tool_call_no_arguments(self, tool_call_no_arguments):
        """Test that tool calls with no arguments is translated correctly. Relevant issue: https://github.com/BerriAI/litellm/issues/6833"""
        pass