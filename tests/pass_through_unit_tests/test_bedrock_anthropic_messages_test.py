
import json
import os
import sys
from datetime import datetime
from typing import AsyncIterator, Dict, Any
import asyncio
import unittest.mock
from unittest.mock import MagicMock
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


@pytest.mark.asyncio
async def test_anthropic_messages_bedrock_converse_with_thinking():
    """
    Test that bedrock/converse model works with thinking parameter.
    Validates the request body from issue where budget_tokens was being lost.
    """
    router = Router(
        model_list=[
            {
                "model_name": "bedrock/converse/us.anthropic.claude-sonnet-4-20250514-v1:0",
                "litellm_params": {
                    "model": "bedrock/converse/us.anthropic.claude-sonnet-4-20250514-v1:0",
                },
            },
        ]
    )

    messages = [{"role": "user", "content": "What is 2+2?"}]

    response = await router.aanthropic_messages(
        messages=messages,
        model="bedrock/converse/us.anthropic.claude-sonnet-4-20250514-v1:0",
        max_tokens=1026,
        thinking={
            "type": "enabled",
            "budget_tokens": 1025
        },
    )
    print("bedrock response: ", response)

    # Verify response
    INSTANCE_BASE_ANTHROPIC_MESSAGES_TEST._validate_response(response)


@pytest.mark.asyncio
async def test_should_not_fail_with_forwarded_headers_bedrock_invoke_messages():
    """
    E2E test for Bedrock invoke messages with header forwarding enabled.
    This calls the real Bedrock endpoint (no mocks) and should not raise
    SigV4 signature mismatch errors when forwarded headers are present.
    """
    router = Router(
        model_list=[
            {
                "model_name": "claude-sonnet-4-5-20250929",
                "litellm_params": {
                    "model": "bedrock/invoke/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                    "aws_region_name": os.getenv("AWS_REGION_NAME"),
                },
            }
        ]
    )

    forwarded_headers = {
        "x-forwarded-for": "10.11.232.194",
        "x-forwarded-port": "443",
        "x-forwarded-proto": "https",
        "x-app": "cli",
    }

    response = await router.aanthropic_messages(
        messages=[{"role": "user", "content": "hi"}],
        model="claude-sonnet-4-5-20250929",
        max_tokens=5,
        stream=False,
        headers=forwarded_headers,  # simulates forward_client_headers_to_llm_api
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_region_name=os.getenv("AWS_REGION_NAME"),
    )
    print("INVOKE API RESPONSE: ", response)

    INSTANCE_BASE_ANTHROPIC_MESSAGES_TEST._validate_response(response)
