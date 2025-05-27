import datetime
import json
import os
import sys
import unittest
from typing import List, Optional, Tuple
from unittest.mock import ANY, MagicMock, Mock, patch

import httpx
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path
import litellm
from litellm.integrations.anthropic_cache_control_hook import AnthropicCacheControlHook
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import StandardCallbackDynamicParams


@pytest.mark.asyncio
async def test_anthropic_cache_control_hook_system_message():
    # Use patch.dict to mock environment variables instead of setting them directly
    with patch.dict(
        os.environ,
        {
            "AWS_ACCESS_KEY_ID": "fake_access_key_id",
            "AWS_SECRET_ACCESS_KEY": "fake_secret_access_key",
            "AWS_REGION_NAME": "us-west-2",
        },
    ):
        anthropic_cache_control_hook = AnthropicCacheControlHook()
        litellm.callbacks = [anthropic_cache_control_hook]

        # Mock response data
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": "Here is my analysis of the key terms and conditions...",
                }
            },
            "stopReason": "stop_sequence",
            "usage": {
                "inputTokens": 100,
                "outputTokens": 200,
                "totalTokens": 300,
                "cacheReadInputTokens": 100,
                "cacheWriteInputTokens": 200,
            },
        }
        mock_response.status_code = 200

        # Mock AsyncHTTPHandler.post method
        client = AsyncHTTPHandler()
        with patch.object(client, "post", return_value=mock_response) as mock_post:
            response = await litellm.acompletion(
                model="bedrock/anthropic.claude-3-5-haiku-20241022-v1:0",
                messages=[
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": "You are an AI assistant tasked with analyzing legal documents.",
                            },
                            {
                                "type": "text",
                                "text": "Here is the full text of a complex legal agreement",
                            },
                        ],
                    },
                    {
                        "role": "user",
                        "content": "what are the key terms and conditions in this agreement?",
                    },
                ],
                cache_control_injection_points=[
                    {
                        "location": "message",
                        "role": "system",
                    },
                ],
                client=client,
            )

            mock_post.assert_called_once()
            request_body = json.loads(mock_post.call_args.kwargs["data"])

            print("request_body: ", json.dumps(request_body, indent=4))

            # Verify the request body
            assert request_body["system"][1]["cachePoint"] == {"type": "default"}


@pytest.mark.asyncio
async def test_anthropic_cache_control_hook_user_message():
    # Use patch.dict to mock environment variables instead of setting them directly
    with patch.dict(
        os.environ,
        {
            "AWS_ACCESS_KEY_ID": "fake_access_key_id",
            "AWS_SECRET_ACCESS_KEY": "fake_secret_access_key",
            "AWS_REGION_NAME": "us-west-2",
        },
    ):
        anthropic_cache_control_hook = AnthropicCacheControlHook()
        litellm.callbacks = [anthropic_cache_control_hook]

        # Mock response data
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": "Here is my analysis of the key terms and conditions...",
                }
            },
            "stopReason": "stop_sequence",
            "usage": {
                "inputTokens": 100,
                "outputTokens": 200,
                "totalTokens": 300,
                "cacheReadInputTokens": 100,
                "cacheWriteInputTokens": 200,
            },
        }
        mock_response.status_code = 200

        # Mock AsyncHTTPHandler.post method
        client = AsyncHTTPHandler()
        with patch.object(client, "post", return_value=mock_response) as mock_post:
            response = await litellm.acompletion(
                model="bedrock/anthropic.claude-3-5-haiku-20241022-v1:0",
                messages=[
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": "You are an AI assistant tasked with analyzing legal documents.",
                            },
                        ],
                    },
                    {
                        "role": "user",
                        "content": "what are the key terms and conditions in this agreement? <very_long_text>",
                    },
                ],
                cache_control_injection_points=[
                    {
                        "location": "message",
                        "role": "user",
                    },
                ],
                client=client,
            )

            mock_post.assert_called_once()
            request_body = json.loads(mock_post.call_args.kwargs["data"])

            print("request_body: ", json.dumps(request_body, indent=4))

            # Verify the request body
            assert request_body["messages"][1]["content"][1]["cachePoint"] == {
                "type": "default"
            }
