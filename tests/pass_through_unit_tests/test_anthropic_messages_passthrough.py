import os
from datetime import datetime
from typing import Dict, Any
import unittest.mock
from unittest.mock import MagicMock

from dotenv import load_dotenv
from litellm.llms.anthropic.pass_through.messages.handler import (
    anthropic_messages,
)

from typing import Optional
from litellm.types.utils import StandardLoggingPayload
from litellm.integrations.custom_logger import CustomLogger
from base_anthropic_unified_messages_test import BaseAnthropicMessagesTest

# Load environment variables
load_dotenv()


def _validate_anthropic_response(response: Dict[str, Any]):
    assert "id" in response
    assert "content" in response
    assert "model" in response
    assert response["role"] == "assistant"


class TestAnthropicDirectAPI(BaseAnthropicMessagesTest):
    """Tests for direct Anthropic API calls"""

    test_non_streaming_base = None

    @property
    def model_config(self) -> Dict[str, Any]:
        return {
            "model": "claude-haiku-4-5-20251001",
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
        }

    @property
    def expected_model_name_in_logging(self) -> str:
        """
        This is the model name that is expected to be in the logging payload
        """
        return "claude-haiku-4-5-20251001"


class TestAnthropicBedrockAPI(BaseAnthropicMessagesTest):
    """Tests for Anthropic via Bedrock"""

    @property
    def model_config(self) -> Dict[str, Any]:
        return {
            "model": "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        }

    @property
    def expected_model_name_in_logging(self) -> str:
        """
        This is the model name that is expected to be in the logging payload
        """
        return "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"


class TestAnthropicOpenAIAPI(BaseAnthropicMessagesTest):
    """Tests for OpenAI via Anthropic messages interface"""

    @property
    def model_config(self) -> Dict[str, Any]:
        return {
            "model": "openai/gpt-4.1-mini",
            "client": None,
        }

    @property
    def expected_model_name_in_logging(self) -> str:
        """
        This is the model name that is expected to be in the logging payload
        """
        return "gpt-4.1-mini"
