import asyncio
import traceback

from dotenv import load_dotenv

import litellm.types
import litellm.types.utils
from litellm.llms.anthropic.chat import ModelResponseIterator

load_dotenv()
import io

from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

import litellm

from litellm.llms.anthropic.common_utils import process_anthropic_headers
from httpx import Headers
from base_llm_unit_tests import BaseLLMChatTest


@pytest.mark.flaky(retries=3, delay=2)
class TestMistralCompletion(BaseLLMChatTest):
    test_basic_tool_calling = None

    def get_base_completion_call_args(self) -> dict:
        litellm.set_verbose = True
        return {"model": "mistral/mistral-medium-latest"}
