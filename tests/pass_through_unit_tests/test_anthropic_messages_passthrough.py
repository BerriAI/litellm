import json
import os
import sys
from datetime import datetime
from typing import AsyncIterator
import asyncio

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import pytest
from dotenv import load_dotenv
from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
    anthropic_messages_handler,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

# Load environment variables
load_dotenv()


@pytest.mark.asyncio
async def test_anthropic_messages_handler_non_streaming():
    """
    Test the anthropic_messages_handler with non-streaming request
    """
    # Get API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not found in environment")

    # Set up test parameters
    messages = [{"role": "user", "content": "Hello, can you tell me a short joke?"}]

    # Call the handler
    async_httpx_client = AsyncHTTPHandler()
    response = await anthropic_messages_handler(
        messages=messages,
        api_key=api_key,
        model="claude-3-haiku-20240307",
        max_tokens=100,
        client=async_httpx_client,
    )

    # Verify response
    assert "id" in response
    assert "content" in response
    assert "model" in response
    assert response["role"] == "assistant"

    print(f"Non-streaming response: {json.dumps(response, indent=2)}")
    return response


@pytest.mark.asyncio
async def test_anthropic_messages_handler_streaming():
    """
    Test the anthropic_messages_handler with streaming request
    """
    # Get API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not found in environment")

    # Set up test parameters
    messages = [{"role": "user", "content": "Hello, can you tell me a short joke?"}]

    # Call the handler
    async_httpx_client = AsyncHTTPHandler()
    response = await anthropic_messages_handler(
        messages=messages,
        api_key=api_key,
        model="claude-3-haiku-20240307",
        max_tokens=100,
        stream=True,
        client=async_httpx_client,
    )

    if isinstance(response, AsyncIterator):
        async for chunk in response:
            print("chunk=", chunk)
