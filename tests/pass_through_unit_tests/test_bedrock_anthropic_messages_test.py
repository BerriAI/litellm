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
                "model_name": "bedrock/converse/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                "litellm_params": {
                    "model": "bedrock/converse/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                },
            },
            {
                "model_name": "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                "litellm_params": {
                    "model": "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                },
            },
        ]
    )

    # Set up test parameters
    messages = [{"role": "user", "content": "Hello, can you tell me a short joke?"}]

    # Call 1 using bedrock/converse/us.anthropic.claude-sonnet-4-5-20250929-v1:0
    response = await router.aanthropic_messages(
        messages=messages,
        model="bedrock/converse/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        max_tokens=100,
    )

    # Verify response
    INSTANCE_BASE_ANTHROPIC_MESSAGES_TEST._validate_response(response)

    # Call 2 using bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0
    response = await router.aanthropic_messages(
        messages=messages,
        model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
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
                "model_name": "bedrock/converse/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                "litellm_params": {
                    "model": "bedrock/converse/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                },
            },
        ]
    )

    messages = [{"role": "user", "content": "What is 2+2?"}]

    response = await router.aanthropic_messages(
        messages=messages,
        model="bedrock/converse/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        max_tokens=1026,
        thinking={"type": "enabled", "budget_tokens": 1025},
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


def test_bedrock_anthropic_messages_system_role_transformation():
    """
    Test that system messages in the messages array are correctly extracted and moved
    to the top-level system parameter for Bedrock.
    """
    config = litellm.AmazonAnthropicClaudeMessagesConfig()

    # 1. Test case: System message in messages array, no existing system param
    req = config.transform_anthropic_messages_request(
        model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
        ],
        anthropic_messages_optional_request_params={"max_tokens": 100},
        litellm_params={},
        headers={},
    )

    assert req["messages"] == [{"role": "user", "content": "Hello"}]
    assert req["system"] == [{"type": "text", "text": "You are a helpful assistant"}]

    # 2. Test case: System message in messages array, and existing system param
    req = config.transform_anthropic_messages_request(
        model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        messages=[
            {"role": "system", "content": "System message in messages array"},
            {"role": "user", "content": "Hello"},
        ],
        anthropic_messages_optional_request_params={
            "max_tokens": 100,
            "system": "Existing top-level system message",
        },
        litellm_params={},
        headers={},
    )

    assert req["messages"] == [{"role": "user", "content": "Hello"}]
    assert req["system"] == [
        {"type": "text", "text": "Existing top-level system message"},
        {"type": "text", "text": "System message in messages array"},
    ]

    # 3. Test case: Empty system message content (should remove system message from messages, system param unaffected)
    req = config.transform_anthropic_messages_request(
        model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": "Hello"},
        ],
        anthropic_messages_optional_request_params={"max_tokens": 100},
        litellm_params={},
        headers={},
    )

    assert req["messages"] == [{"role": "user", "content": "Hello"}]
    assert "system" not in req

    # 4. Test case: Non-text items in list content (should only extract text items, ignore non-text)
    req = config.transform_anthropic_messages_request(
        model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        messages=[
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "This is text"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": "abc",
                        },
                    },
                ],
            },
            {"role": "user", "content": "Hello"},
        ],
        anthropic_messages_optional_request_params={"max_tokens": 100},
        litellm_params={},
        headers={},
    )

    assert req["messages"] == [{"role": "user", "content": "Hello"}]
    assert req["system"] == [{"type": "text", "text": "This is text"}]

    # 5. Test case: cache_control in system message (should preserve cache_control)
    req = config.transform_anthropic_messages_request(
        model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        messages=[
            {
                "role": "system",
                "content": "Cached system message",
                "cache_control": {"type": "ephemeral"},
            },
            {"role": "user", "content": "Hello"},
        ],
        anthropic_messages_optional_request_params={"max_tokens": 100},
        litellm_params={},
        headers={},
    )
    assert req["messages"] == [{"role": "user", "content": "Hello"}]
    assert req["system"] == [
        {
            "type": "text",
            "text": "Cached system message",
            "cache_control": {"type": "ephemeral"},
        }
    ]

    # 6. Test case: content as list, type text with empty text, cache_control, and list containing non-dict items
    req = config.transform_anthropic_messages_request(
        model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "This is text",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"type": "text", "text": ""},  # empty text, should be ignored
                    "invalid_non_dict_item",  # non-dict item in list, should be ignored
                ],
            },
            {"role": "user", "content": "Hello"},
        ],
        anthropic_messages_optional_request_params={"max_tokens": 100},
        litellm_params={},
        headers={},
    )
    assert req["messages"] == [{"role": "user", "content": "Hello"}]
    assert req["system"] == [
        {
            "type": "text",
            "text": "This is text",
            "cache_control": {"type": "ephemeral"},
        }
    ]

    # 7. Test case: system_messages_found is True but system_messages_to_add is empty (e.g., content list only has non-text item)
    req = config.transform_anthropic_messages_request(
        model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": "abc",
                        },
                    }
                ],
            },
            {"role": "user", "content": "Hello"},
        ],
        anthropic_messages_optional_request_params={"max_tokens": 100},
        litellm_params={},
        headers={},
    )
    assert req["messages"] == [{"role": "user", "content": "Hello"}]
    assert "system" not in req

    # 8. Test case: existing_system is empty string
    req = config.transform_anthropic_messages_request(
        model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        messages=[
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "Hello"},
        ],
        anthropic_messages_optional_request_params={
            "max_tokens": 100,
            "system": "",
        },
        litellm_params={},
        headers={},
    )
    assert req["messages"] == [{"role": "user", "content": "Hello"}]
    assert req["system"] == [{"type": "text", "text": "System message"}]

    # 9. Test case: existing_system is list
    req = config.transform_anthropic_messages_request(
        model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        messages=[
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "Hello"},
        ],
        anthropic_messages_optional_request_params={
            "max_tokens": 100,
            "system": [{"type": "text", "text": "Existing list item"}],
        },
        litellm_params={},
        headers={},
    )
    assert req["messages"] == [{"role": "user", "content": "Hello"}]
    assert req["system"] == [
        {"type": "text", "text": "Existing list item"},
        {"type": "text", "text": "System message"},
    ]

    # 10. Test case: messages contains elements that are not dicts or do not have role="system"
    req_data = {
        "messages": [
            "not_a_dict_message",
            {"content": "no_role_key"},
            {"role": "user", "content": "Hello"},
        ]
    }
    config._extract_system_messages_from_messages(req_data)
    assert req_data["messages"] == [
        "not_a_dict_message",
        {"content": "no_role_key"},
        {"role": "user", "content": "Hello"},
    ]
