
import json
import os
import sys
from datetime import datetime
from typing import AsyncIterator, Dict, Any
import asyncio
import unittest.mock
from unittest.mock import AsyncMock, MagicMock
import pytest
from litellm.router import Router

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path
import litellm
from base_anthropic_unified_messages_test import BaseAnthropicMessagesTest

INSTANCE_BASE_ANTHROPIC_MESSAGES_TEST = BaseAnthropicMessagesTest()

@pytest.mark.asyncio
async def test_anthropic_messages_litellm_router_bedrock():
    """
    Test the anthropic_messages with non-streaming request
    """

    litellm._turn_on_debug()
    router = Router(
        model_list=[
            {
                "model_name": "bedrock/converse/us.anthropic.claude-sonnet-4-20250514-v1:0",
                "litellm_params": {
                    "model": "bedrock/converse/us.anthropic.claude-sonnet-4-20250514-v1:0",
                },
            },
            {
                "model_name": "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0",
                "litellm_params": {
                    "model": "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0",
                },
            }
        ]
    )
    
    # Set up test parameters
    messages = [{"role": "user", "content": "Hello, can you tell me a short joke?"}]

    # Call 1 using bedrock/converse/us.anthropic.claude-sonnet-4-20250514-v1:0
    response = await router.aanthropic_messages(
        messages=messages,
        model="bedrock/converse/us.anthropic.claude-sonnet-4-20250514-v1:0",
        max_tokens=100,
    )

    # Verify response
    INSTANCE_BASE_ANTHROPIC_MESSAGES_TEST._validate_response(response)

    # Call 2 using bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0
    response = await router.aanthropic_messages(
        messages=messages,
        model="bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0",
        max_tokens=100,
    )

    # Verify response
    INSTANCE_BASE_ANTHROPIC_MESSAGES_TEST._validate_response(response)


