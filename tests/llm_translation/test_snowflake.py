import asyncio
import json
import os
import httpx
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm import completion, acompletion, responses
from litellm.exceptions import APIConnectionError
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler


@pytest.mark.skip(reason="Requires Snowflake credentials - run manually when needed")
def test_snowflake_tool_calling_responses_api():
    """
    Test Snowflake tool calling with Responses API.
    Requires SNOWFLAKE_JWT and SNOWFLAKE_ACCOUNT_ID environment variables.
    """
    import litellm

    # Skip if credentials not available
    if not os.getenv("SNOWFLAKE_JWT") or not os.getenv("SNOWFLAKE_ACCOUNT_ID"):
        pytest.skip("Snowflake credentials not available")

    litellm.drop_params = False  # We now support tools!

    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"],
            },
        }
    ]

    try:
        # Test with tool_choice to force tool use
        response = responses(
            model="snowflake/claude-3-5-sonnet",
            input="What's the weather in Paris?",
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
            max_output_tokens=200,
        )

        assert response is not None
        assert hasattr(response, "output")
        assert len(response.output) > 0

        # Verify tool call was made
        tool_call_found = False
        for item in response.output:
            if hasattr(item, "type") and item.type == "function_call":
                tool_call_found = True
                assert item.name == "get_weather"
                assert hasattr(item, "arguments")
                print(f"✅ Tool call detected: {item.name}({item.arguments})")
                break

        assert tool_call_found, "Expected tool call but none was found"

    except APIConnectionError as e:
        if "JWT token is invalid" in str(e):
            pytest.skip("Invalid Snowflake JWT token")
        elif "Application failed to respond" in str(e) or "502" in str(e):
            pytest.skip(f"Snowflake API unavailable: {e}")
        else:
            raise
