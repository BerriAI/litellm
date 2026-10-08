import json
from datetime import datetime
from typing import AsyncIterator, Dict, Any
import asyncio
import unittest.mock
from unittest.mock import AsyncMock, MagicMock

import litellm
import pytest
from dotenv import load_dotenv
from litellm.llms.anthropic.pass_through.messages.handler import (
    anthropic_messages,
)

from typing import Optional
from litellm.types.utils import StandardLoggingPayload
from litellm.integrations.custom_logger import CustomLogger
from litellm.router import Router
import importlib
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM


class TestCustomLogger(CustomLogger):
    def __init__(self):
        super().__init__()
        self.logged_standard_logging_payload: Optional[StandardLoggingPayload] = None

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        print("inside async_log_success_event")
        self.logged_standard_logging_payload = kwargs.get("standard_logging_object")

        pass


class BaseAnthropicMessagesTest:
    """Base class for anthropic messages tests to reduce code duplication"""

    @property
    def model_config(self) -> Dict[str, Any]:
        """Override in subclasses to provide model-specific configuration"""
        raise NotImplementedError("Subclasses must implement model_config")

    @property
    def expected_model_name_in_logging(self) -> str:
        """
        This is the model name that is expected to be in the logging payload
        """
        raise NotImplementedError(
            "Subclasses must implement expected_model_name_in_logging"
        )

    def _validate_response(self, response: Any):
        """Validate non-streaming response structure"""
        # Handle type checking - response should be a dict for non-streaming
        if isinstance(response, AsyncIterator):
            pytest.fail("Expected non-streaming response but got AsyncIterator")

        assert isinstance(
            response, dict
        ), f"Expected dict response, got {type(response)}"
        assert "id" in response
        assert "content" in response
        assert "model" in response
        assert response.get("role") == "assistant"
