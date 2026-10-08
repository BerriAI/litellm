import json
import os
from datetime import datetime
from typing import Dict, Any
import asyncio
import unittest.mock
from unittest.mock import MagicMock

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
from base_anthropic_unified_messages_test import BaseAnthropicMessagesTest

# Load environment variables
load_dotenv()


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown(event_loop):  # Add event_loop as a dependency
    curr_dir = os.getcwd()

    import litellm
    from litellm import Router

    importlib.reload(litellm)

    # Set the event loop from the fixture
    asyncio.set_event_loop(event_loop)

    print(litellm)
    yield

    # Clean up any pending tasks
    pending = asyncio.all_tasks(event_loop)
    for task in pending:
        task.cancel()

    # Run the event loop until all tasks are cancelled
    if pending:
        event_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))


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


@pytest.mark.asyncio
async def test_anthropic_messages_litellm_router_non_streaming():
    """
    Test the anthropic_messages with non-streaming request
    """
    litellm.turn_on_debug()
    router = Router(
        model_list=[
            {
                "model_name": "claude-special-alias",
                "litellm_params": {
                    "model": "claude-haiku-4-5-20251001",
                    "api_key": os.getenv("ANTHROPIC_API_KEY"),
                },
            }
        ]
    )

    # Set up test parameters
    messages = [{"role": "user", "content": "Hello, can you tell me a short joke?"}]

    # Call the handler
    response = await router.aanthropic_messages(
        messages=messages,
        model="claude-special-alias",
        max_tokens=100,
    )

    # Verify response
    assert "id" in response
    assert "content" in response
    assert "model" in response
    assert response["role"] == "assistant"

    print(f"Non-streaming response: {json.dumps(response, indent=2)}")
    return response

