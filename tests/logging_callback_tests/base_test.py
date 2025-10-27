import asyncio
import httpx
import json
import pytest
import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm.exceptions import BadRequestError
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.utils import CustomStreamWrapper
from litellm.types.utils import ModelResponse

# test_example.py
from abc import ABC, abstractmethod


class BaseLoggingCallbackTest(ABC):
    """
    Abstract base test class that enforces a common test across all test classes.
    """

    @pytest.fixture
    def mock_response_obj(self):
        from litellm.types.utils import (
            ModelResponse,
            Choices,
            Message,
            ChatCompletionMessageToolCall,
            Function,
            Usage,
            CompletionTokensDetailsWrapper,
            PromptTokensDetailsWrapper,
        )

        # Create a mock response object with the structure you need
        return ModelResponse(
            id="chatcmpl-ASId3YJWagBpBskWfoNEMPFSkmrEw",
            created=1731308157,
            model="gpt-4o-mini-2024-07-18",
            object="chat.completion",
            system_fingerprint="fp_0ba0d124f1",
            choices=[
                Choices(
                    finish_reason="tool_calls",
                    index=0,
                    message=Message(
                        content=None,
                        role="assistant",
                        tool_calls=[
                            ChatCompletionMessageToolCall(
                                function=Function(
                                    arguments='{"city": "New York"}', name="get_weather"
                                ),
                                id="call_PngsQS5YGmIZKnswhnUOnOVb",
                                type="function",
                            ),
                            ChatCompletionMessageToolCall(
                                function=Function(
                                    arguments='{"city": "New York"}', name="get_news"
                                ),
                                id="call_1zsDThBu0VSK7KuY7eCcJBnq",
                                type="function",
                            ),
                        ],
                        function_call=None,
                    ),
                )
            ],
            usage=Usage(
                completion_tokens=46,
                prompt_tokens=86,
                total_tokens=132,
                completion_tokens_details=CompletionTokensDetailsWrapper(
                    accepted_prediction_tokens=0,
                    audio_tokens=0,
                    reasoning_tokens=0,
                    rejected_prediction_tokens=0,
                    text_tokens=None,
                ),
                prompt_tokens_details=PromptTokensDetailsWrapper(
                    audio_tokens=0, cached_tokens=0, text_tokens=None, image_tokens=None
                ),
            ),
            service_tier=None,
        )

    @abstractmethod
    def test_parallel_tool_calls(self, mock_response_obj: ModelResponse):
        """
        Check if parallel tool calls are correctly logged by Logging callback

        Relevant issue - https://github.com/BerriAI/litellm/issues/6677
        """
        pass
