import copy
import datetime
import json
import os
import subprocess
import sys
import textwrap
import unittest
from typing import List, Optional, Tuple
from unittest.mock import ANY, MagicMock, Mock, patch

import httpx
import pytest

import litellm
from litellm.integrations.anthropic_cache_control_hook import (
    AnthropicCacheControlHook,
    supports_openai_prompt_cache_breakpoint,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import StandardCallbackDynamicParams


@pytest.fixture(autouse=True)
def _no_openai_api_base_override(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setattr(litellm, "api_base", None)


def _rendered_log_message(call):
    message = str(call.args[0])
    values = call.args[1:]
    return message % values if values else message


@pytest.mark.asyncio
async def test_anthropic_cache_control_hook_system_message(monkeypatch: pytest.MonkeyPatch):
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
        monkeypatch.setattr(litellm, "callbacks", [anthropic_cache_control_hook])

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

            # Verify that cache control was applied (Bedrock transforms it to a separate item)
            cache_control_count = sum(
                1 for item in request_body["system"] if isinstance(item, dict) and "cachePoint" in item
            )
            assert cache_control_count == 1, f"Expected exactly 1 cache control point, found {cache_control_count}"


@pytest.mark.asyncio
async def test_anthropic_cache_control_hook_user_message(monkeypatch: pytest.MonkeyPatch):
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
        monkeypatch.setattr(litellm, "callbacks", [anthropic_cache_control_hook])

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
            assert request_body["messages"][1]["content"][1]["cachePoint"] == {"type": "default"}


@pytest.mark.asyncio
async def test_anthropic_cache_control_hook_negative_indices(monkeypatch: pytest.MonkeyPatch):
    """
    Test the bug fix for handling negative indices in cache control injection points.
    This test verifies that negative indices (-1, -2) are properly converted to positive indices
    and cache control is applied to the correct messages.
    """
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
        monkeypatch.setattr(litellm, "callbacks", [anthropic_cache_control_hook])

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
            # Test with multiple messages and negative indices
            response = await litellm.acompletion(
                model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI assistant tasked with analyzing legal documents.",
                    },
                    {
                        "role": "user",
                        "content": "Here is the first part of the document.",
                    },
                    {
                        "role": "assistant",
                        "content": "I understand. Please provide the document.",
                    },
                    {
                        "role": "user",
                        "content": "Here is the full legal document text that should be cached.",
                    },
                ],
                cache_control_injection_points=[
                    {
                        "location": "message",
                        "index": -1,  # Should target the last message (index 3)
                    },
                    {
                        "location": "message",
                        "index": -2,  # Should target the second-to-last message (index 2)
                    },
                ],
                client=client,
            )

            mock_post.assert_called_once()
            request_body = json.loads(mock_post.call_args.kwargs["data"])

            print("request_body: ", json.dumps(request_body, indent=4))

            # The input `messages` has 4 elements. After removing the system message,
            # the `request_body["messages"]` will have 3 elements (indices 0, 1, 2).

            # Verify the last message (input index -1 -> request index 2) has cache control
            last_message_content = request_body["messages"][2]["content"]
            assert isinstance(last_message_content, list), "Last message content should be a list"
            assert any("cachePoint" in item for item in last_message_content if isinstance(item, dict)), (
                "CachePoint missing in last message"
            )

            # Note: Based on debug output, the hook correctly applies cache control to both messages,
            # but the Bedrock API transformation appears to only preserve cache control for user messages,
            # not assistant messages. This is a limitation of the API transformation layer.
            #
            # The second-to-last message (assistant) gets cache_control from the hook but loses it
            # during API transformation. This test documents this behavior.
            second_last_message_content = request_body["messages"][1]["content"]
            assert isinstance(second_last_message_content, list), "Second-to-last message content should be a list"

            # Check if assistant message cache control is preserved (currently it's not)
            assistant_has_cache_control = any(
                "cachePoint" in item for item in second_last_message_content if isinstance(item, dict)
            )
            print(f"Assistant message has cache control in final request: {assistant_has_cache_control}")

            # Verify the first user message (request index 0) was NOT modified
            first_user_message_content = request_body["messages"][0]["content"]
            assert isinstance(first_user_message_content, list), "First user message content should be a list"
            assert not any("cachePoint" in item for item in first_user_message_content if isinstance(item, dict)), (
                "CachePoint unexpectedly found in first user message"
            )


@pytest.mark.asyncio
async def test_anthropic_cache_control_hook_out_of_bounds_logging(monkeypatch: pytest.MonkeyPatch):
    """
    Test that warning logs are generated when out-of-bounds indices are used.
    This verifies that the verbose_logger.warning is called with the correct message.
    """
    with patch.dict(
        os.environ,
        {
            "AWS_ACCESS_KEY_ID": "fake_access_key_id",
            "AWS_SECRET_ACCESS_KEY": "fake_secret_access_key",
            "AWS_REGION_NAME": "us-west-2",
        },
    ):
        anthropic_cache_control_hook = AnthropicCacheControlHook()
        monkeypatch.setattr(litellm, "callbacks", [anthropic_cache_control_hook])

        # Mock response data
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": "Response",
                }
            },
            "stopReason": "stop_sequence",
            "usage": {
                "inputTokens": 50,
                "outputTokens": 100,
                "totalTokens": 150,
            },
        }
        mock_response.status_code = 200

        client = AsyncHTTPHandler()

        # Mock the verbose_logger to capture warning calls
        with patch("litellm.integrations.anthropic_cache_control_hook.verbose_logger") as mock_logger:
            with patch.object(client, "post", return_value=mock_response) as mock_post:
                messages = [
                    {"role": "user", "content": "Message 1"},
                    {"role": "user", "content": "Message 2"},
                ]

                await litellm.acompletion(
                    model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                    messages=messages,
                    cache_control_injection_points=[{"location": "message", "index": 10}],  # Out of bounds index
                    client=client,
                )

                # Verify that warning was called with the expected message
                mock_logger.warning.assert_called_once()
                warning_call = _rendered_log_message(mock_logger.warning.call_args)

                # Check that the warning message contains the expected information
                assert "AnthropicCacheControlHook: Provided index 10 is out of bounds" in warning_call
                assert "message list of length 2" in warning_call
                assert "Targeted index was 10" in warning_call
                assert "Skipping cache control injection for this point" in warning_call


@pytest.mark.asyncio
async def test_anthropic_cache_control_hook_negative_out_of_bounds_logging(monkeypatch: pytest.MonkeyPatch):
    """
    Test that warning logs are generated for negative indices that are out of bounds.
    """
    with patch.dict(
        os.environ,
        {
            "AWS_ACCESS_KEY_ID": "fake_access_key_id",
            "AWS_SECRET_ACCESS_KEY": "fake_secret_access_key",
            "AWS_REGION_NAME": "us-west-2",
        },
    ):
        anthropic_cache_control_hook = AnthropicCacheControlHook()
        monkeypatch.setattr(litellm, "callbacks", [anthropic_cache_control_hook])

        # Mock response data
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": "Response",
                }
            },
            "stopReason": "stop_sequence",
            "usage": {
                "inputTokens": 50,
                "outputTokens": 100,
                "totalTokens": 150,
            },
        }
        mock_response.status_code = 200

        client = AsyncHTTPHandler()

        # Mock the verbose_logger to capture warning calls
        with patch("litellm.integrations.anthropic_cache_control_hook.verbose_logger") as mock_logger:
            with patch.object(client, "post", return_value=mock_response) as mock_post:
                messages = [
                    {"role": "user", "content": "Single message"},
                ]

                await litellm.acompletion(
                    model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                    messages=messages,
                    cache_control_injection_points=[
                        {
                            "location": "message",
                            "index": -5,
                        }  # Negative out of bounds index
                    ],
                    client=client,
                )

                # Verify that warning was called with the expected message
                mock_logger.warning.assert_called_once()
                warning_call = _rendered_log_message(mock_logger.warning.call_args)

                # Check that the warning message contains the original negative index
                assert "AnthropicCacheControlHook: Provided index -5 is out of bounds" in warning_call
                assert "message list of length 1" in warning_call
                assert "Targeted index was -4" in warning_call  # -5 + 1 = -4 (converted index)
                assert "Skipping cache control injection for this point" in warning_call


@pytest.mark.asyncio
async def test_anthropic_cache_control_hook_multiple_user_messages(monkeypatch: pytest.MonkeyPatch):
    """
    Test cache control injection on multiple user messages specifically.
    Note: Bedrock API combines consecutive user messages into a single message with multiple content blocks.
    """
    with patch.dict(
        os.environ,
        {
            "AWS_ACCESS_KEY_ID": "fake_access_key_id",
            "AWS_SECRET_ACCESS_KEY": "fake_secret_access_key",
            "AWS_REGION_NAME": "us-west-2",
        },
    ):
        anthropic_cache_control_hook = AnthropicCacheControlHook()
        monkeypatch.setattr(litellm, "callbacks", [anthropic_cache_control_hook])

        # Mock response data
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": "Response",
                }
            },
            "stopReason": "stop_sequence",
            "usage": {
                "inputTokens": 100,
                "outputTokens": 200,
                "totalTokens": 300,
            },
        }
        mock_response.status_code = 200

        client = AsyncHTTPHandler()
        with patch.object(client, "post", return_value=mock_response) as mock_post:
            # Test with multiple user messages and negative indices
            response = await litellm.acompletion(
                model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                messages=[
                    {
                        "role": "user",
                        "content": "First user message.",
                    },
                    {
                        "role": "user",
                        "content": "Second user message.",
                    },
                    {
                        "role": "user",
                        "content": "Third user message that should be cached.",
                    },
                ],
                cache_control_injection_points=[
                    {
                        "location": "message",
                        "index": -1,  # Should target the last message (index 2)
                    },
                    {
                        "location": "message",
                        "index": -2,  # Should target the second-to-last message (index 1)
                    },
                ],
                client=client,
            )

            mock_post.assert_called_once()
            request_body = json.loads(mock_post.call_args.kwargs["data"])

            print(
                "Multiple user messages request_body: ",
                json.dumps(request_body, indent=4),
            )

            # Bedrock API combines consecutive user messages into a single message
            assert len(request_body["messages"]) == 1

            # The combined message should have multiple content blocks with cache control
            combined_message_content = request_body["messages"][0]["content"]
            assert isinstance(combined_message_content, list)

            # Count cache control points - should have 2 since both injection points were applied
            cache_control_count = sum(
                1 for item in combined_message_content if isinstance(item, dict) and "cachePoint" in item
            )
            assert cache_control_count == 2

            print(f"Found {cache_control_count} cache control points in the combined message")


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_index", [10, -10])
async def test_anthropic_cache_control_hook_out_of_bounds(bad_index, monkeypatch: pytest.MonkeyPatch):
    """
    Verify the hook does not raise an error and makes no changes
    when an out-of-bounds index is provided.
    """
    with patch.dict(
        os.environ,
        {
            "AWS_ACCESS_KEY_ID": "fake_access_key_id",
            "AWS_SECRET_ACCESS_KEY": "fake_secret_access_key",
            "AWS_REGION_NAME": "us-west-2",
        },
    ):
        anthropic_cache_control_hook = AnthropicCacheControlHook()
        monkeypatch.setattr(litellm, "callbacks", [anthropic_cache_control_hook])

        # Mock response data
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": "Response",
                }
            },
            "stopReason": "stop_sequence",
            "usage": {
                "inputTokens": 50,
                "outputTokens": 100,
                "totalTokens": 150,
            },
        }
        mock_response.status_code = 200

        client = AsyncHTTPHandler()
        with patch.object(client, "post", return_value=mock_response) as mock_post:
            messages = [
                {"role": "user", "content": "Message 1"},
                {"role": "user", "content": "Message 2"},
            ]

            await litellm.acompletion(
                model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                messages=messages,
                cache_control_injection_points=[{"location": "message", "index": bad_index}],
                client=client,
            )

            mock_post.assert_called_once()
            request_body = json.loads(mock_post.call_args.kwargs["data"])

            # Assert that NO cache control was applied to any message
            for msg in request_body["messages"]:
                content = msg.get("content", [])
                if isinstance(content, list):
                    assert not any("cachePoint" in item for item in content if isinstance(item, dict))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message_list",
    [[{"role": "user", "content": "Single message"}]],  # Single message only - empty list will fail at API level
)
async def test_anthropic_cache_control_hook_single_message(message_list, monkeypatch: pytest.MonkeyPatch):
    """
    Verify the hook runs without error on very short message lists.
    """
    with patch.dict(
        os.environ,
        {
            "AWS_ACCESS_KEY_ID": "fake_access_key_id",
            "AWS_SECRET_ACCESS_KEY": "fake_secret_access_key",
            "AWS_REGION_NAME": "us-west-2",
        },
    ):
        anthropic_cache_control_hook = AnthropicCacheControlHook()
        monkeypatch.setattr(litellm, "callbacks", [anthropic_cache_control_hook])

        # Mock response data
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": "Response",
                }
            },
            "stopReason": "stop_sequence",
            "usage": {
                "inputTokens": 50,
                "outputTokens": 100,
                "totalTokens": 150,
            },
        }
        mock_response.status_code = 200

        client = AsyncHTTPHandler()
        with patch.object(client, "post", return_value=mock_response) as mock_post:
            await litellm.acompletion(
                model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                messages=message_list,
                cache_control_injection_points=[{"location": "message", "index": -1}],
                client=client,
            )

            mock_post.assert_called_once()
            request_body = json.loads(mock_post.call_args.kwargs["data"])
            # For the single message, verify cache control was applied
            content = request_body["messages"][0]["content"]
            assert isinstance(content, list)
            assert any("cachePoint" in item for item in content if isinstance(item, dict))


@pytest.mark.asyncio
async def test_anthropic_cache_control_hook_empty_message_list(monkeypatch: pytest.MonkeyPatch):
    """
    Verify that empty message lists are handled appropriately (should fail at API level, not hook level).
    """
    with patch.dict(
        os.environ,
        {
            "AWS_ACCESS_KEY_ID": "fake_access_key_id",
            "AWS_SECRET_ACCESS_KEY": "fake_secret_access_key",
            "AWS_REGION_NAME": "us-west-2",
        },
    ):
        anthropic_cache_control_hook = AnthropicCacheControlHook()
        monkeypatch.setattr(litellm, "callbacks", [anthropic_cache_control_hook])

        client = AsyncHTTPHandler()
        with patch.object(client, "post", return_value=MagicMock()) as mock_post:
            # This should fail at the API level, not the hook level
            with pytest.raises(
                litellm.BadRequestError,
                match="bedrock requires at least one non-system message",
            ):
                await litellm.acompletion(
                    model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                    messages=[],
                    cache_control_injection_points=[{"location": "message", "index": -1}],
                    client=client,
                )


@pytest.mark.asyncio
async def test_anthropic_cache_control_hook_no_op(monkeypatch: pytest.MonkeyPatch):
    """
    Verify that if no injection points are specified, messages remain unmodified.
    """
    with patch.dict(
        os.environ,
        {
            "AWS_ACCESS_KEY_ID": "fake_access_key_id",
            "AWS_SECRET_ACCESS_KEY": "fake_secret_access_key",
            "AWS_REGION_NAME": "us-west-2",
        },
    ):
        anthropic_cache_control_hook = AnthropicCacheControlHook()
        monkeypatch.setattr(litellm, "callbacks", [anthropic_cache_control_hook])

        # Mock response data
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": "Response",
                }
            },
            "stopReason": "stop_sequence",
            "usage": {
                "inputTokens": 50,
                "outputTokens": 100,
                "totalTokens": 150,
            },
        }
        mock_response.status_code = 200

        client = AsyncHTTPHandler()
        with patch.object(client, "post", return_value=mock_response) as mock_post:
            messages = [
                {"role": "user", "content": "Message 1"},
                {"role": "user", "content": "Message 2"},
            ]

            await litellm.acompletion(
                model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                messages=messages,
                # No cache_control_injection_points parameter
                client=client,
            )

            mock_post.assert_called_once()
            request_body = json.loads(mock_post.call_args.kwargs["data"])

            # Assert that NO cache control was applied
            for msg in request_body["messages"]:
                content = msg.get("content", [])
                if isinstance(content, list):
                    assert not any("cachePoint" in item for item in content if isinstance(item, dict))


@pytest.mark.asyncio
async def test_anthropic_cache_control_hook_multiple_content_items_last_only(monkeypatch: pytest.MonkeyPatch):
    """
    Test that cache_control is only applied to the last content item in a list, not all items.
    This verifies the fix for https://github.com/BerriAI/litellm/issues/15696
    """
    with patch.dict(
        os.environ,
        {
            "AWS_ACCESS_KEY_ID": "fake_access_key_id",
            "AWS_SECRET_ACCESS_KEY": "fake_secret_access_key",
            "AWS_REGION_NAME": "us-west-2",
        },
    ):
        anthropic_cache_control_hook = AnthropicCacheControlHook()
        monkeypatch.setattr(litellm, "callbacks", [anthropic_cache_control_hook])

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": "Response",
                }
            },
            "stopReason": "stop_sequence",
            "usage": {
                "inputTokens": 100,
                "outputTokens": 200,
                "totalTokens": 300,
            },
        }
        mock_response.status_code = 200

        client = AsyncHTTPHandler()
        with patch.object(client, "post", return_value=mock_response) as mock_post:
            response = await litellm.acompletion(
                model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "First piece of context"},
                            {"type": "text", "text": "Second piece of context"},
                            {"type": "text", "text": "Third piece of context"},
                            {"type": "text", "text": "Fourth piece of context"},
                            {
                                "type": "text",
                                "text": "Fifth piece of context - should be cached",
                            },
                        ],
                    }
                ],
                cache_control_injection_points=[{"location": "message", "index": -1}],
                client=client,
            )

            mock_post.assert_called_once()
            request_body = json.loads(mock_post.call_args.kwargs["data"])

            print("Multi-content request_body: ", json.dumps(request_body, indent=4))

            message_content = request_body["messages"][0]["content"]
            assert isinstance(message_content, list)

            cache_control_count = sum(1 for item in message_content if isinstance(item, dict) and "cachePoint" in item)
            assert cache_control_count == 1, (
                f"Expected exactly 1 cache control point, found {cache_control_count}. This test verifies the fix for issue 15696 where cache_control was incorrectly applied to ALL content items."
            )


@pytest.mark.asyncio
async def test_anthropic_cache_control_hook_document_analysis_multiple_pages(monkeypatch: pytest.MonkeyPatch):
    """
    Test cache_control with multiple document pages to ensure only the last page gets cached.
    This simulates document analysis with 6 content blocks, verifying the fix for issue 15696.
    """
    with patch.dict(
        os.environ,
        {
            "AWS_ACCESS_KEY_ID": "fake_access_key_id",
            "AWS_SECRET_ACCESS_KEY": "fake_secret_access_key",
            "AWS_REGION_NAME": "us-west-2",
        },
    ):
        anthropic_cache_control_hook = AnthropicCacheControlHook()
        monkeypatch.setattr(litellm, "callbacks", [anthropic_cache_control_hook])

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": "Summary",
                }
            },
            "stopReason": "stop_sequence",
            "usage": {
                "inputTokens": 100,
                "outputTokens": 200,
                "totalTokens": 300,
            },
        }
        mock_response.status_code = 200

        client = AsyncHTTPHandler()
        with patch.object(client, "post", return_value=mock_response) as mock_post:
            response = await litellm.acompletion(
                model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Summarize this document"},
                            {"type": "text", "text": "Page 1 content"},
                            {"type": "text", "text": "Page 2 content"},
                            {"type": "text", "text": "Page 3 content"},
                            {"type": "text", "text": "Page 4 content"},
                            {
                                "type": "text",
                                "text": "Page 5 content - final page to cache",
                            },
                        ],
                    }
                ],
                cache_control_injection_points=[{"location": "message", "role": "user"}],
                client=client,
            )

            mock_post.assert_called_once()
            request_body = json.loads(mock_post.call_args.kwargs["data"])

            print("Document analysis request_body: ", json.dumps(request_body, indent=4))

            message_content = request_body["messages"][0]["content"]
            assert isinstance(message_content, list)

            cache_control_count = sum(1 for item in message_content if isinstance(item, dict) and "cachePoint" in item)
            assert cache_control_count == 1, (
                f"Expected exactly 1 cache control point (last item only), found {cache_control_count}. Before fix, this would be 6 (one for each content item)."
            )


def test_gemini_cache_control_injection_points_detected():
    """
    Test that cache_control_injection_points work for Gemini models.

    Verifies the full flow:
    1. The hook injects cache_control markers on string-content messages
    2. is_cached_message() detects the injected markers (message-level cache_control)
    3. separate_cached_messages() correctly separates the messages

    Fixes GitHub issue #18519.
    """
    from litellm.llms.vertex_ai.context_caching.transformation import (
        separate_cached_messages,
    )
    from litellm.utils import is_cached_message

    hook = AnthropicCacheControlHook()

    # Simulate messages as they would appear for a Gemini call with string content
    messages: List[AllMessageValues] = [
        {
            "role": "system",
            "content": "You are a helpful assistant that analyzes legal documents.",
        },
        {
            "role": "user",
            "content": "What are the key terms?",
        },
    ]

    # Simulate what the hook does: inject cache_control on the system message
    injection_points = [{"location": "message", "role": "system"}]

    # Manually apply the hook's logic for the system message (string content case)
    # The hook sets message["cache_control"] = {"type": "ephemeral"} for string content
    hook._safe_insert_cache_control_in_message(
        message=messages[0],
        control={"type": "ephemeral"},
    )

    # Verify the hook injected message-level cache_control (string content path)
    assert messages[0].get("cache_control") == {"type": "ephemeral"}

    # Verify is_cached_message detects message-level cache_control
    assert is_cached_message(messages[0]) is True
    assert is_cached_message(messages[1]) is False

    # Verify separate_cached_messages correctly separates them
    cached, non_cached = separate_cached_messages(messages)
    assert len(cached) == 1
    assert cached[0]["role"] == "system"
    assert len(non_cached) == 1
    assert non_cached[0]["role"] == "user"


def test_gemini_cache_control_injection_list_content_detected():
    """
    Test that cache_control_injection_points work for Gemini models
    when the message content is a list (not string).
    """
    from litellm.llms.vertex_ai.context_caching.transformation import (
        separate_cached_messages,
    )
    from litellm.utils import is_cached_message

    hook = AnthropicCacheControlHook()

    messages: List[AllMessageValues] = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a helpful assistant."},
                {"type": "text", "text": "Analyze legal documents carefully."},
            ],
        },
        {
            "role": "user",
            "content": "What are the key terms?",
        },
    ]

    # Apply the hook's logic for list content - sets cache_control on last item
    hook._safe_insert_cache_control_in_message(
        message=messages[0],
        control={"type": "ephemeral"},
    )

    # Verify cache_control was set on the last content item
    assert messages[0]["content"][-1]["cache_control"] == {"type": "ephemeral"}

    # Verify is_cached_message detects content-item-level cache_control
    assert is_cached_message(messages[0]) is True
    assert is_cached_message(messages[1]) is False

    # Verify separate_cached_messages correctly separates them
    cached, non_cached = separate_cached_messages(messages)
    assert len(cached) == 1
    assert len(non_cached) == 1


@pytest.mark.asyncio
async def test_anthropic_cache_control_hook_string_negative_index(monkeypatch: pytest.MonkeyPatch):
    """
    Test that string negative indices like "-1" are handled correctly.

    When cache_control_injection_points are stored in DB/config as JSON, indices
    like -1 become the string "-1". Previously, str.isdigit() returned False for
    "-1" so the cache control was silently skipped. This tests the fix.
    """
    with patch.dict(
        os.environ,
        {
            "AWS_ACCESS_KEY_ID": "fake_access_key_id",
            "AWS_SECRET_ACCESS_KEY": "fake_secret_access_key",
            "AWS_REGION_NAME": "us-west-2",
        },
    ):
        anthropic_cache_control_hook = AnthropicCacheControlHook()
        monkeypatch.setattr(litellm, "callbacks", [anthropic_cache_control_hook])

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": "Response",
                }
            },
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 100,
                "outputTokens": 50,
                "totalTokens": 150,
            },
        }
        mock_response.status_code = 200

        client = AsyncHTTPHandler()
        with patch.object(client, "post", return_value=mock_response) as mock_post:
            await litellm.acompletion(
                model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                messages=[
                    {"role": "user", "content": "First message"},
                    {"role": "assistant", "content": "First response"},
                    {"role": "user", "content": "Second message"},
                ],
                # index is a string "-1" (as stored in DB/config JSON)
                cache_control_injection_points=[
                    {"location": "message", "index": "-1"},
                ],
                client=client,
            )

            mock_post.assert_called_once()
            request_body = json.loads(mock_post.call_args.kwargs["data"])

            # The last user message should have cache control applied
            last_message = request_body["messages"][-1]
            last_message_content = last_message["content"]
            assert isinstance(last_message_content, list), f"Expected list content, got {type(last_message_content)}"
            has_cache_point = any(isinstance(item, dict) and "cachePoint" in item for item in last_message_content)
            assert has_cache_point, (
                f"Expected cachePoint in last message content, got: {last_message_content}. "
                "String index '-1' was not parsed correctly (str.isdigit() returns False for negative strings)."
            )


def _count_cache_control(messages: List[AllMessageValues]) -> int:
    """Count cache_control breakpoints across messages (message + content level)."""
    count = 0
    for message in messages:
        if message.get("cache_control") is not None:
            count += 1
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("cache_control") is not None:
                    count += 1
    return count


def _build_injection_points():
    return [
        {
            "location": "message",
            "role": "system",
            "control": {"type": "ephemeral", "ttl": "1h"},
        },
        {
            "location": "message",
            "index": -1,
            "control": {"type": "ephemeral", "ttl": "5m"},
        },
    ]


def test_cache_control_hook_caps_at_four_blocks_with_client_cache_control():
    """Regression for LIT-3667 / Anthropic 'A maximum of 4 blocks ... Found 5'.

    A Hermes-style request already carries 4 client cache_control breakpoints on
    its system messages. With both auto-inject points configured the hook must
    NOT add a 5th breakpoint, and must NOT overwrite the client's existing
    breakpoints (TTL must be preserved).
    """
    hook = AnthropicCacheControlHook()

    messages: List[AllMessageValues] = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": f"System block {i}",
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                }
            ],
        }
        for i in range(4)
    ]
    messages.append({"role": "user", "content": "hello"})

    _, processed, _ = hook.get_chat_completion_prompt(
        model="bedrock/us.anthropic.claude-opus-4-6-v1:0",
        messages=messages,
        non_default_params={"cache_control_injection_points": _build_injection_points()},
        prompt_id=None,
        prompt_variables=None,
        dynamic_callback_params={},
    )

    assert _count_cache_control(processed) == 4, "Hook must cap cache_control at Anthropic's limit of 4 blocks"

    # Client TTL on system blocks must be preserved (not overwritten by config).
    for i in range(4):
        assert processed[i]["content"][-1]["cache_control"] == {
            "type": "ephemeral",
            "ttl": "1h",
        }

    # The last (user) message must not receive a 5th breakpoint.
    user_message = processed[-1]
    assert user_message.get("cache_control") is None
    user_content = user_message.get("content")
    if isinstance(user_content, list):
        assert all(block.get("cache_control") is None for block in user_content if isinstance(block, dict))


def test_cache_control_hook_caps_at_four_blocks_without_client_cache_control():
    """Four plain system messages + role:system + index:-1 must stay at 4 blocks.

    role:system fills all four slots, so the index:-1 point is skipped.
    """
    hook = AnthropicCacheControlHook()

    messages: List[AllMessageValues] = [{"role": "system", "content": f"System {i}"} for i in range(4)]
    messages.append({"role": "user", "content": "hello"})

    _, processed, _ = hook.get_chat_completion_prompt(
        model="bedrock/us.anthropic.claude-opus-4-6-v1:0",
        messages=messages,
        non_default_params={"cache_control_injection_points": _build_injection_points()},
        prompt_id=None,
        prompt_variables=None,
        dynamic_callback_params={},
    )

    assert _count_cache_control(processed) == 4
    # All four system messages cached; user message skipped (limit reached).
    assert all(processed[i].get("cache_control") is not None for i in range(4))
    assert processed[-1].get("cache_control") is None


def test_cache_control_hook_does_not_overwrite_existing_cache_control():
    """If a targeted message already has client cache_control, do not inject."""
    hook = AnthropicCacheControlHook()

    messages: List[AllMessageValues] = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "Cached by client",
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                }
            ],
        },
        {"role": "user", "content": "hello"},
    ]

    _, processed, _ = hook.get_chat_completion_prompt(
        model="bedrock/us.anthropic.claude-opus-4-6-v1:0",
        messages=messages,
        # Target the already-cached system message with a different TTL.
        non_default_params={
            "cache_control_injection_points": [
                {
                    "location": "message",
                    "index": 0,
                    "control": {"type": "ephemeral", "ttl": "5m"},
                }
            ]
        },
        prompt_id=None,
        prompt_variables=None,
        dynamic_callback_params={},
    )

    # Client's 1h TTL must be preserved, not replaced by the config's 5m.
    assert processed[0]["content"][-1]["cache_control"] == {
        "type": "ephemeral",
        "ttl": "1h",
    }
    assert _count_cache_control(processed) == 1


@pytest.mark.asyncio
async def test_cache_control_hook_bedrock_payload_caps_cachepoints_at_four(monkeypatch: pytest.MonkeyPatch):
    """End-to-end: outgoing Bedrock payload must not exceed 4 cachePoint blocks.

    Reproduces the customer report where 4 client cache_control system blocks
    plus auto-inject produced 5 cachePoint blocks and Bedrock returned 400.
    """
    with patch.dict(
        os.environ,
        {
            "AWS_ACCESS_KEY_ID": "fake_access_key_id",
            "AWS_SECRET_ACCESS_KEY": "fake_secret_access_key",
            "AWS_REGION_NAME": "us-east-1",
        },
    ):
        monkeypatch.setattr(litellm, "callbacks", [AnthropicCacheControlHook()])

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "output": {"message": {"role": "assistant", "content": "ok"}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 100, "outputTokens": 4, "totalTokens": 104},
        }
        mock_response.status_code = 200

        client = AsyncHTTPHandler()
        with patch.object(client, "post", return_value=mock_response) as mock_post:
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": f"System block {i}",
                            "cache_control": {"type": "ephemeral", "ttl": "1h"},
                        }
                    ],
                }
                for i in range(4)
            ]
            messages.append({"role": "user", "content": "hello"})

            await litellm.acompletion(
                model="bedrock/us.anthropic.claude-opus-4-6-v1:0",
                messages=messages,
                max_tokens=32,
                cache_control_injection_points=_build_injection_points(),
                client=client,
            )

            request_body = json.loads(mock_post.call_args.kwargs["data"])

            cache_points = sum(
                1 for block in request_body.get("system", []) if isinstance(block, dict) and "cachePoint" in block
            )
            for msg in request_body.get("messages", []):
                content = msg.get("content", [])
                if isinstance(content, list):
                    cache_points += sum(1 for block in content if isinstance(block, dict) and "cachePoint" in block)

            assert cache_points <= 4, (
                f"Bedrock payload exceeded Anthropic's 4 cache_control block limit: "
                f"found {cache_points} cachePoint blocks"
            )


def test_cache_control_hook_reserves_slot_for_tool_config_point():
    """A tool_config injection point consumes one of the 4 slots downstream.

    With role:system targeting 4 system messages plus a tool_config point, the
    hook must inject at most 3 message-level blocks so the tool_config cachePoint
    appended by the Bedrock transform keeps the total at 4, not 5.
    """
    hook = AnthropicCacheControlHook()

    messages: List[AllMessageValues] = [{"role": "system", "content": f"System {i}"} for i in range(4)]
    messages.append({"role": "user", "content": "hello"})

    _, processed, non_default_params = hook.get_chat_completion_prompt(
        model="bedrock/us.anthropic.claude-opus-4-6-v1:0",
        messages=messages,
        non_default_params={
            "cache_control_injection_points": [
                {
                    "location": "message",
                    "role": "system",
                    "control": {"type": "ephemeral", "ttl": "1h"},
                },
                {"location": "tool_config"},
            ]
        },
        prompt_id=None,
        prompt_variables=None,
        dynamic_callback_params={},
    )

    assert _count_cache_control(processed) == 3
    # The tool_config point is passed through for the provider transform,
    # stamped so re-entries never re-judge it against litellm's own marks.
    assert non_default_params["cache_control_injection_points"] == [
        {"location": "tool_config", "_litellm_judged": True}
    ]


@pytest.mark.asyncio
async def test_cache_control_hook_bedrock_payload_caps_with_tool_config_point(monkeypatch: pytest.MonkeyPatch):
    """End-to-end: message + tool_config injection must not exceed 4 cachePoints."""
    with patch.dict(
        os.environ,
        {
            "AWS_ACCESS_KEY_ID": "fake_access_key_id",
            "AWS_SECRET_ACCESS_KEY": "fake_secret_access_key",
            "AWS_REGION_NAME": "us-east-1",
        },
    ):
        monkeypatch.setattr(litellm, "callbacks", [AnthropicCacheControlHook()])

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "output": {"message": {"role": "assistant", "content": "ok"}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 100, "outputTokens": 4, "totalTokens": 104},
        }
        mock_response.status_code = 200

        client = AsyncHTTPHandler()
        with patch.object(client, "post", return_value=mock_response) as mock_post:
            messages = [{"role": "system", "content": f"System block {i}"} for i in range(4)]
            messages.append({"role": "user", "content": "What is the weather?"})

            await litellm.acompletion(
                model="bedrock/us.anthropic.claude-opus-4-6-v1:0",
                messages=messages,
                max_tokens=32,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather for a location",
                            "parameters": {
                                "type": "object",
                                "properties": {"location": {"type": "string"}},
                                "required": ["location"],
                            },
                        },
                    }
                ],
                cache_control_injection_points=[
                    {
                        "location": "message",
                        "role": "system",
                        "control": {"type": "ephemeral", "ttl": "1h"},
                    },
                    {"location": "tool_config"},
                ],
                client=client,
            )

            request_body = json.loads(mock_post.call_args.kwargs["data"])

            cache_points = sum(
                1 for block in request_body.get("system", []) if isinstance(block, dict) and "cachePoint" in block
            )
            for msg in request_body.get("messages", []):
                content = msg.get("content", [])
                if isinstance(content, list):
                    cache_points += sum(1 for block in content if isinstance(block, dict) and "cachePoint" in block)
            for tool in request_body.get("toolConfig", {}).get("tools", []):
                if isinstance(tool, dict) and "cachePoint" in tool:
                    cache_points += 1

            assert cache_points <= 4, (
                f"Bedrock payload exceeded Anthropic's 4 cache_control block limit "
                f"when mixing message and tool_config injection: found {cache_points}"
            )


class TestApplyToAnthropicMessagesRequest:
    """Tests for apply_to_anthropic_messages_request (v1/messages cache control)."""

    def test_system_string_injection(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
        system = "You are helpful"
        injection_points = [{"location": "message", "role": "system"}]

        result_msgs, result_sys, remaining = AnthropicCacheControlHook.apply_to_anthropic_messages_request(
            messages=messages,
            system=system,
            injection_points=injection_points,
        )

        assert result_sys == [{"type": "text", "text": "You are helpful", "cache_control": {"type": "ephemeral"}}]
        assert result_msgs == messages
        assert remaining == []

    def test_system_list_injection(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
        system = [
            {"type": "text", "text": "Part 1"},
            {"type": "text", "text": "Part 2"},
        ]
        injection_points = [{"location": "message", "role": "system"}]

        _, result_sys, _ = AnthropicCacheControlHook.apply_to_anthropic_messages_request(
            messages=messages,
            system=system,
            injection_points=injection_points,
        )

        assert result_sys[0] == {"type": "text", "text": "Part 1"}
        assert result_sys[1] == {"type": "text", "text": "Part 2", "cache_control": {"type": "ephemeral"}}

    def test_user_message_injection_by_role(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "First"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Response"}]},
            {"role": "user", "content": [{"type": "text", "text": "Second"}]},
        ]
        injection_points = [{"location": "message", "role": "user"}]

        result_msgs, _, _ = AnthropicCacheControlHook.apply_to_anthropic_messages_request(
            messages=messages,
            system=None,
            injection_points=injection_points,
        )

        assert result_msgs[0]["content"][-1].get("cache_control") == {"type": "ephemeral"}
        assert result_msgs[2]["content"][-1].get("cache_control") == {"type": "ephemeral"}
        assert result_msgs[1]["content"][-1].get("cache_control") is None

    def test_message_injection_by_index(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "First"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Response"}]},
            {"role": "user", "content": [{"type": "text", "text": "Second"}]},
        ]
        injection_points = [{"location": "message", "index": -1}]

        result_msgs, _, _ = AnthropicCacheControlHook.apply_to_anthropic_messages_request(
            messages=messages,
            system=None,
            injection_points=injection_points,
        )

        assert result_msgs[2]["content"][-1].get("cache_control") == {"type": "ephemeral"}
        assert result_msgs[0]["content"][-1].get("cache_control") is None
        assert result_msgs[1]["content"][-1].get("cache_control") is None

    def test_mixed_system_and_message_injection(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
            {"role": "user", "content": [{"type": "text", "text": "Question"}]},
        ]
        system = "System prompt"
        injection_points = [
            {"location": "message", "role": "system"},
            {"location": "message", "index": -1},
        ]

        result_msgs, result_sys, _ = AnthropicCacheControlHook.apply_to_anthropic_messages_request(
            messages=messages,
            system=system,
            injection_points=injection_points,
        )

        assert result_sys[0]["cache_control"] == {"type": "ephemeral"}
        assert result_msgs[2]["content"][-1].get("cache_control") == {"type": "ephemeral"}

    def test_respects_max_4_blocks(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": f"Msg {i}"}]} for i in range(6)]
        system = "System"
        injection_points = [
            {"location": "message", "role": "system"},
            {"location": "message", "role": "user"},
        ]

        result_msgs, result_sys, _ = AnthropicCacheControlHook.apply_to_anthropic_messages_request(
            messages=messages,
            system=system,
            injection_points=injection_points,
        )

        sys_blocks = sum(1 for b in (result_sys or []) if isinstance(b, dict) and b.get("cache_control") is not None)
        total_blocks = sys_blocks + sum(AnthropicCacheControlHook._count_cache_control_blocks(m) for m in result_msgs)
        assert total_blocks <= 4

    def test_tool_config_points_forwarded_as_remaining(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
        injection_points = [
            {"location": "message", "role": "user"},
            {"location": "tool_config"},
        ]

        _, _, remaining = AnthropicCacheControlHook.apply_to_anthropic_messages_request(
            messages=messages,
            system=None,
            injection_points=injection_points,
        )

        assert remaining == [{"location": "tool_config"}]

    def test_no_injection_points_returns_unchanged(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
        system = "System"

        result_msgs, result_sys, remaining = AnthropicCacheControlHook.apply_to_anthropic_messages_request(
            messages=messages,
            system=system,
            injection_points=[],
        )

        assert result_msgs == messages
        assert result_sys == system
        assert remaining == []

    def test_does_not_mutate_input(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
        system = [{"type": "text", "text": "System"}]
        injection_points = [{"location": "message", "role": "system"}]

        original_system = copy.deepcopy(system)
        original_messages = copy.deepcopy(messages)

        AnthropicCacheControlHook.apply_to_anthropic_messages_request(
            messages=messages,
            system=system,
            injection_points=injection_points,
        )

        assert messages == original_messages
        assert system == original_system

    def test_system_none_with_system_point_skipped(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
        injection_points = [{"location": "message", "role": "system"}]

        result_msgs, result_sys, _ = AnthropicCacheControlHook.apply_to_anthropic_messages_request(
            messages=messages,
            system=None,
            injection_points=injection_points,
        )

        assert result_sys is None

    def test_existing_cache_control_counted_toward_limit(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "A", "cache_control": {"type": "ephemeral"}}]},
            {"role": "assistant", "content": [{"type": "text", "text": "B", "cache_control": {"type": "ephemeral"}}]},
            {"role": "user", "content": [{"type": "text", "text": "C", "cache_control": {"type": "ephemeral"}}]},
            {"role": "user", "content": [{"type": "text", "text": "D"}]},
            {"role": "user", "content": [{"type": "text", "text": "E"}]},
        ]
        system = "System"
        injection_points = [
            {"location": "message", "role": "system"},
            {"location": "message", "index": 3},
            {"location": "message", "index": 4},
        ]

        result_msgs, result_sys, _ = AnthropicCacheControlHook.apply_to_anthropic_messages_request(
            messages=messages,
            system=system,
            injection_points=injection_points,
        )

        sys_blocks = sum(1 for b in (result_sys or []) if isinstance(b, dict) and b.get("cache_control") is not None)
        total_blocks = sys_blocks + sum(AnthropicCacheControlHook._count_cache_control_blocks(m) for m in result_msgs)
        assert total_blocks <= 4


class TestEnableAnthropicPromptCaching:
    """Auto-injected default breakpoints via litellm.enable_anthropic_prompt_caching."""

    MESSAGES: List[AllMessageValues] = [
        {"role": "system", "content": "a long system prompt"},
        {"role": "user", "content": "first turn"},
        {"role": "assistant", "content": "a reply"},
        {"role": "user", "content": "latest turn"},
    ]

    def _points(self, model="claude-sonnet-4-5", provider="anthropic", messages=None, system=None, tools=None):
        return AnthropicCacheControlHook.get_default_injection_points(
            messages=copy.deepcopy(self.MESSAGES) if messages is None else messages,
            system=system,
            model=model,
            custom_llm_provider=provider,
            tools=tools,
        )

    def test_disabled_by_default(self):
        assert litellm.enable_anthropic_prompt_caching is False
        assert self._points() == []

    def test_injects_system_and_trailing_turn(self, monkeypatch):
        monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", True)
        assert self._points() == [
            {"location": "message", "role": "system", "index": None, "control": {"type": "ephemeral"}},
            {"location": "message", "role": None, "index": -1, "control": {"type": "ephemeral"}},
        ]

    def test_bedrock_claude_is_injected(self, monkeypatch):
        monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", True)
        points = self._points(model="us.anthropic.claude-sonnet-4-5-20250929-v1:0", provider="bedrock")
        assert [p["index"] for p in points] == [None, -1]

    @pytest.mark.parametrize("model, provider", [("gpt-4o", "openai"), ("gemini-2.0-flash", "gemini")])
    def test_non_anthropic_providers_never_injected(self, monkeypatch, model, provider):
        """These report supports_prompt_caching=True but never consume cache_control markers."""
        from litellm.utils import supports_prompt_caching

        monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", True)
        assert supports_prompt_caching(model=model, custom_llm_provider=provider) is True
        assert self._points(model=model, provider=provider) == []

    def test_databricks_claude_not_injected_despite_caching_support(self, monkeypatch, local_model_cost_map):
        from litellm.utils import supports_prompt_caching

        monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", True)
        model = "databricks/databricks-claude-sonnet-4-5"
        assert supports_prompt_caching(model=model, custom_llm_provider="databricks") is True
        assert self._points(model=model, provider="databricks") == []

    def test_model_without_caching_support_not_injected(self, monkeypatch):
        monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", True)
        assert self._points(model="anthropic.claude-3-5-sonnet-20240620-v1:0", provider="bedrock") == []

    def test_stands_down_when_client_sent_cache_control(self, monkeypatch):
        monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", True)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral"}}]},
            {"role": "user", "content": "latest turn"},
        ]
        assert self._points(messages=messages) == []

    def test_stands_down_when_system_block_has_cache_control(self, monkeypatch):
        monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", True)
        system = [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral"}}]
        assert self._points(messages=[{"role": "user", "content": "hi"}], system=system) == []

    @staticmethod
    def _tools(count: int, cached: bool) -> List[dict]:
        tool: dict = {"type": "function", "function": {"name": "t", "description": "d", "parameters": {}}}
        if cached:
            tool["cache_control"] = {"type": "ephemeral"}
        return [{**tool, "function": {**tool["function"], "name": f"t{i}"}} for i in range(count)]

    def test_stands_down_when_only_tools_carry_cache_control(self, monkeypatch):
        """Caching just the tool definitions is a normal client pattern, and those
        breakpoints count toward the provider's four-block limit. Three of them plus
        our two would be five, which Anthropic rejects outright."""
        monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", True)
        assert self._points(tools=self._tools(3, cached=True)) == []

    def test_injects_when_tools_carry_no_cache_control(self, monkeypatch):
        """Tools alone must not suppress injection; only client-marked ones do."""
        monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", True)
        assert [p["index"] for p in self._points(tools=self._tools(3, cached=False))] == [None, -1]

    @pytest.mark.parametrize("tools", [None, []])
    def test_absent_tools_do_not_suppress_injection(self, monkeypatch, tools):
        monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", True)
        assert [p["index"] for p in self._points(tools=tools)] == [None, -1]

    def test_stands_down_when_tool_function_carries_cache_control(self, monkeypatch):
        """OpenAI-shaped tools nest cache_control under ``function``; the Anthropic
        chat transform honors that location, so the stand-down must see it too."""
        monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", True)
        tools = [{"type": "function", "function": {"name": "t", "parameters": {}, "cache_control": {"type": "ephemeral"}}}]
        assert self._points(tools=tools) == []

    def test_seed_stands_down_when_only_tools_carry_cache_control(self, monkeypatch):
        """Same guard on the /chat/completions seeding path."""
        monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", True)
        params: dict = {}
        AnthropicCacheControlHook.maybe_seed_default_injection_points(
            non_default_params=params,
            messages=copy.deepcopy(self.MESSAGES),
            model="claude-sonnet-4-5",
            custom_llm_provider="anthropic",
            tools=self._tools(3, cached=True),
        )
        assert "cache_control_injection_points" not in params

    def test_v1_messages_stands_down_when_only_tools_carry_cache_control(self, monkeypatch):
        """Same guard on the /v1/messages path, where tools reach the hook directly."""
        monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", True)
        messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        result_msgs, result_sys = AnthropicCacheControlHook.maybe_inject_cache_control(
            copy.deepcopy(messages),
            "sys",
            {},
            model="claude-sonnet-4-5",
            custom_llm_provider="anthropic",
            tools=self._tools(3, cached=True),
        )
        assert result_sys == "sys"
        assert result_msgs == messages

    def test_default_ttl_is_anthropics_five_minute_cache(self, monkeypatch):
        monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", True)
        assert all(p["control"] == {"type": "ephemeral"} for p in self._points())

    @pytest.mark.parametrize("ttl", ["5m", "1h"])
    def test_ttl_override_applied(self, monkeypatch, ttl):
        monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", True)
        monkeypatch.setattr(litellm, "anthropic_prompt_caching_ttl", ttl)
        assert all(p["control"] == {"type": "ephemeral", "ttl": ttl} for p in self._points())

    def test_seed_does_not_override_configured_points(self, monkeypatch):
        monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", True)
        configured = [{"location": "message", "role": "user", "index": 0}]
        params = {"cache_control_injection_points": configured}
        AnthropicCacheControlHook.maybe_seed_default_injection_points(
            non_default_params=params,
            messages=copy.deepcopy(self.MESSAGES),
            model="claude-sonnet-4-5",
            custom_llm_provider="anthropic",
        )
        assert params["cache_control_injection_points"] is configured

    def test_seed_adds_defaults_when_enabled(self, monkeypatch):
        monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", True)
        params: dict = {}
        AnthropicCacheControlHook.maybe_seed_default_injection_points(
            non_default_params=params,
            messages=copy.deepcopy(self.MESSAGES),
            model="claude-sonnet-4-5",
            custom_llm_provider="anthropic",
        )
        assert [p["index"] for p in params["cache_control_injection_points"]] == [None, -1]

    def test_seed_is_noop_when_disabled(self):
        params: dict = {}
        AnthropicCacheControlHook.maybe_seed_default_injection_points(
            non_default_params=params,
            messages=copy.deepcopy(self.MESSAGES),
            model="claude-sonnet-4-5",
            custom_llm_provider="anthropic",
        )
        assert params == {}

    def test_v1_messages_applies_defaults_end_to_end(self, monkeypatch):
        monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", True)
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "first"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "reply"}]},
            {"role": "user", "content": [{"type": "text", "text": "latest"}]},
        ]
        result_msgs, result_sys = AnthropicCacheControlHook.maybe_inject_cache_control(
            messages,
            "a system prompt",
            {},
            model="claude-sonnet-4-5",
            custom_llm_provider="anthropic",
        )

        assert result_sys == [{"type": "text", "text": "a system prompt", "cache_control": {"type": "ephemeral"}}]
        assert result_msgs[-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in result_msgs[0]["content"][-1]

    def test_messages_with_default_injections_leaves_the_caller_list_untouched(self, monkeypatch):
        """
        Routing calls this on the live request's own message list to derive the affinity key, before
        the request is sent. Marking in place would leak litellm's breakpoints into the caller's
        messages, where the real injection pass later reads them back as client-supplied ones.
        """
        monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", True)
        messages = copy.deepcopy(self.MESSAGES)
        before = copy.deepcopy(messages)

        injected = AnthropicCacheControlHook.messages_with_default_injections(
            messages=messages, models=("claude-sonnet-4-5",)
        )

        assert injected != messages
        assert messages == before


class TestPerKeyEnablePromptCaching:
    """Per-request enable_prompt_caching override (stamped from key metadata) with the global flag off."""

    MESSAGES: List[AllMessageValues] = [
        {"role": "system", "content": "a long system prompt"},
        {"role": "user", "content": "latest turn"},
    ]

    def _points(self, enable_prompt_caching, model="claude-sonnet-4-5", provider="anthropic", messages=None):
        return AnthropicCacheControlHook.get_default_injection_points(
            messages=copy.deepcopy(self.MESSAGES) if messages is None else messages,
            system=None,
            model=model,
            custom_llm_provider=provider,
            enable_prompt_caching=enable_prompt_caching,
        )

    def test_true_injects_with_global_flag_off(self):
        assert litellm.enable_anthropic_prompt_caching is False
        assert self._points(True) == [
            {"location": "message", "role": "system", "index": None, "control": {"type": "ephemeral"}},
            {"location": "message", "role": None, "index": -1, "control": {"type": "ephemeral"}},
        ]

    @pytest.mark.parametrize("enable_prompt_caching", [False, None])
    def test_false_and_none_fall_back_to_global_flag(self, enable_prompt_caching):
        assert self._points(enable_prompt_caching) == []

    def test_false_does_not_suppress_global_flag(self, monkeypatch):
        monkeypatch.setattr(litellm, "enable_anthropic_prompt_caching", True)
        assert [p["index"] for p in self._points(False)] == [None, -1]

    def test_provider_gate_still_applies(self):
        assert self._points(True, model="gpt-4o", provider="openai") == []

    def test_unsupported_model_gate_still_applies(self):
        assert self._points(True, model="anthropic.claude-3-5-sonnet-20240620-v1:0", provider="bedrock") == []

    def test_client_markers_still_win(self):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral"}}]},
            {"role": "user", "content": "latest turn"},
        ]
        assert self._points(True, messages=messages) == []

    def test_seed_injects_with_global_flag_off(self):
        params: dict = {}
        AnthropicCacheControlHook.maybe_seed_default_injection_points(
            non_default_params=params,
            messages=copy.deepcopy(self.MESSAGES),
            model="claude-sonnet-4-5",
            custom_llm_provider="anthropic",
            enable_prompt_caching=True,
        )
        assert [p["index"] for p in params["cache_control_injection_points"]] == [None, -1]

    def test_v1_messages_injects_and_pops_flag_from_kwargs(self):
        kwargs: dict = {"enable_prompt_caching": True}
        result_msgs, result_sys = AnthropicCacheControlHook.maybe_inject_cache_control(
            [{"role": "user", "content": [{"type": "text", "text": "latest"}]}],
            "a system prompt",
            kwargs,
            model="claude-sonnet-4-5",
            custom_llm_provider="anthropic",
        )
        assert result_sys == [{"type": "text", "text": "a system prompt", "cache_control": {"type": "ephemeral"}}]
        assert result_msgs[-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
        assert "enable_prompt_caching" not in kwargs

    def test_v1_messages_pops_flag_even_when_noop(self):
        kwargs: dict = {"enable_prompt_caching": True}
        AnthropicCacheControlHook.maybe_inject_cache_control(
            [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            None,
            kwargs,
            model="gpt-4o",
            custom_llm_provider="openai",
        )
        assert "enable_prompt_caching" not in kwargs

    def test_v1_messages_is_noop_when_disabled(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        result_msgs, result_sys = AnthropicCacheControlHook.maybe_inject_cache_control(
            messages,
            "sys",
            {},
            model="claude-sonnet-4-5",
            custom_llm_provider="anthropic",
        )

        assert result_sys == "sys"
        assert result_msgs == messages


class TestConfiguredInjectionPointsStandDown:
    """Configured cache_control_injection_points must stand down entirely when the
    client already set its own cache_control anywhere in the request (LIT-4582);
    injecting alongside client breakpoints clashes with the client's caching
    strategy and can push the request past Anthropic's four-block limit."""

    CONFIGURED = [{"location": "message", "role": "system"}]

    CLEAN_MESSAGES: List[AllMessageValues] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]

    MARKED_MESSAGES: List[AllMessageValues] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]},
    ]

    V1_MESSAGES = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]

    def _seed(self, params, messages, tools=None):
        AnthropicCacheControlHook.maybe_seed_default_injection_points(
            non_default_params=params,
            messages=messages,
            model="claude-sonnet-4-5",
            custom_llm_provider="anthropic",
            tools=tools,
        )

    def _inject(self, messages, kwargs, system="sys", tools=None):
        return AnthropicCacheControlHook.maybe_inject_cache_control(
            messages,
            system,
            kwargs,
            model="claude-sonnet-4-5",
            custom_llm_provider="anthropic",
            tools=tools,
        )

    def test_configured_points_dropped_when_messages_carry_cache_control(self):
        params = {"cache_control_injection_points": copy.deepcopy(self.CONFIGURED)}
        self._seed(params, copy.deepcopy(self.MARKED_MESSAGES))
        assert "cache_control_injection_points" not in params

    @pytest.mark.parametrize(
        "tool",
        [
            {"type": "function", "function": {"name": "t", "parameters": {}}, "cache_control": {"type": "ephemeral"}},
            {"type": "function", "function": {"name": "t", "parameters": {}, "cache_control": {"type": "ephemeral"}}},
        ],
        ids=["top_level", "nested_in_function"],
    )
    def test_configured_points_dropped_when_tools_carry_cache_control(self, tool):
        params = {"cache_control_injection_points": copy.deepcopy(self.CONFIGURED)}
        self._seed(params, copy.deepcopy(self.CLEAN_MESSAGES), tools=[tool])
        assert "cache_control_injection_points" not in params

    def test_configured_points_kept_when_request_is_unmarked(self):
        configured = copy.deepcopy(self.CONFIGURED)
        params = {"cache_control_injection_points": configured}
        self._seed(params, copy.deepcopy(self.CLEAN_MESSAGES))
        assert params["cache_control_injection_points"] is configured

    def test_judged_remainder_survives_reentry_despite_injected_marks(self):
        """acompletion() re-enters completion() after injection ran, with only the
        stamped non-message points written back; the re-entry must not misread
        litellm's own marks as client ones and drop that remainder."""
        remainder = [{"location": "tool_config", "_litellm_judged": True}]
        params = {"cache_control_injection_points": remainder}
        self._seed(params, copy.deepcopy(self.MARKED_MESSAGES))
        assert params["cache_control_injection_points"] is remainder

    def test_v1_messages_stand_down_when_content_block_marked(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]}
        ]
        kwargs = {"cache_control_injection_points": copy.deepcopy(self.CONFIGURED)}
        result_msgs, result_sys = self._inject(copy.deepcopy(messages), kwargs)
        assert result_msgs == messages
        assert result_sys == "sys"
        assert "cache_control_injection_points" not in kwargs

    def test_v1_messages_stand_down_when_system_block_marked(self):
        """A configured point targeting a message must not fire when the client
        marked the system prompt; the old behavior injected into the message
        because only the exact targeted position was guarded."""
        system = [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral"}}]
        kwargs = {"cache_control_injection_points": [{"location": "message", "role": "user"}]}
        result_msgs, result_sys = self._inject(copy.deepcopy(self.V1_MESSAGES), kwargs, system=system)
        assert result_msgs == self.V1_MESSAGES
        assert result_sys == system
        assert "cache_control_injection_points" not in kwargs

    def test_v1_messages_stand_down_when_tools_marked(self):
        tools = [{"name": "t", "input_schema": {}, "cache_control": {"type": "ephemeral"}}]
        kwargs = {"cache_control_injection_points": copy.deepcopy(self.CONFIGURED)}
        result_msgs, result_sys = self._inject(copy.deepcopy(self.V1_MESSAGES), kwargs, tools=tools)
        assert result_msgs == self.V1_MESSAGES
        assert result_sys == "sys"
        assert "cache_control_injection_points" not in kwargs

    def test_v1_messages_configured_points_apply_when_unmarked(self):
        kwargs = {"cache_control_injection_points": copy.deepcopy(self.CONFIGURED)}
        _, result_sys = self._inject(copy.deepcopy(self.V1_MESSAGES), kwargs)
        assert result_sys == [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}]

    def test_v1_messages_reentry_flow_preserves_tool_config_remainder(self):
        """The advisor interceptor re-enters anthropic_messages() with the outer
        request's kwargs and post-injection messages. The first pass applies the
        message point and writes back a stamped tool_config remainder; the
        re-entry must keep that remainder even though the messages and system
        now carry litellm's own marks."""
        points = [{"location": "message", "role": "system"}, {"location": "tool_config"}]
        kwargs = {"cache_control_injection_points": copy.deepcopy(points)}
        msgs1, sys1 = self._inject(copy.deepcopy(self.V1_MESSAGES), kwargs)
        assert sys1[0]["cache_control"] == {"type": "ephemeral"}
        expected_remainder = [{"location": "tool_config", "_litellm_judged": True}]
        assert kwargs["cache_control_injection_points"] == expected_remainder

        msgs2, sys2 = self._inject(msgs1, kwargs, system=sys1)
        assert kwargs["cache_control_injection_points"] == expected_remainder
        assert msgs2 == msgs1
        assert sys2 == sys1


class TestAnthropicPromptCachingEnvVars:
    """Both settings are read from the environment at import, so an admin can enable
    auto-caching without a config file. Each case re-imports litellm in a subprocess
    so the env is read fresh without contaminating this process's module graph.
    """

    @staticmethod
    def _import_litellm_with_env(env_override: dict) -> Tuple[bool, Optional[str]]:
        env = os.environ.copy()
        env.pop("LITELLM_ENABLE_ANTHROPIC_PROMPT_CACHING", None)
        env.pop("LITELLM_ANTHROPIC_PROMPT_CACHING_TTL", None)
        env.update(env_override)
        script = textwrap.dedent(
            """
            import json, litellm
            print(json.dumps([litellm.enable_anthropic_prompt_caching, litellm.anthropic_prompt_caching_ttl]))
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=300
        )
        assert result.returncode == 0, result.stderr
        enabled, ttl = json.loads(result.stdout.strip().splitlines()[-1])
        return enabled, ttl

    def test_unset_env_leaves_auto_caching_off(self):
        assert self._import_litellm_with_env({}) == (False, None)

    @pytest.mark.parametrize("value", ["true", "True", "TRUE"])
    def test_env_enables_auto_caching_case_insensitively(self, value):
        enabled, _ = self._import_litellm_with_env({"LITELLM_ENABLE_ANTHROPIC_PROMPT_CACHING": value})
        assert enabled is True

    @pytest.mark.parametrize("value", ["false", "0", "yes", ""])
    def test_env_only_enables_on_true(self, value):
        enabled, _ = self._import_litellm_with_env({"LITELLM_ENABLE_ANTHROPIC_PROMPT_CACHING": value})
        assert enabled is False

    @pytest.mark.parametrize("value", ["5m", "1h"])
    def test_ttl_env_is_applied(self, value):
        _, ttl = self._import_litellm_with_env({"LITELLM_ANTHROPIC_PROMPT_CACHING_TTL": value})
        assert ttl == value

    @pytest.mark.parametrize("value", ["10m", "1H", "3600", "ephemeral"])
    def test_unsupported_ttl_env_falls_back_to_provider_default(self, value):
        """An unparseable TTL must fall back to Anthropic's 5m default, never reach the provider verbatim."""
        _, ttl = self._import_litellm_with_env({"LITELLM_ANTHROPIC_PROMPT_CACHING_TTL": value})
        assert ttl is None


def _contains_key(value, key) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(v, key) for v in value.values())
    if isinstance(value, list):
        return any(_contains_key(v, key) for v in value)
    return False


class TestOpenAIPromptCacheBreakpoint:
    """OpenAI GPT-5.6+ targets get content-block `prompt_cache_breakpoint` markers and a
    request-level `prompt_cache_options` instead of Anthropic `cache_control` (#37509)."""

    EXPLICIT = {"mode": "explicit"}
    SYSTEM_POINT = [{"location": "message", "role": "system"}]

    @staticmethod
    def _inject(messages, system, kwargs, model="openai/gpt-5.6", custom_llm_provider=None):
        return AnthropicCacheControlHook.maybe_inject_cache_control(
            copy.deepcopy(messages),
            copy.deepcopy(system),
            kwargs,
            model=model,
            custom_llm_provider=custom_llm_provider,
        )

    @staticmethod
    def _chat(messages, params, model="openai/gpt-5.6"):
        return AnthropicCacheControlHook().get_chat_completion_prompt(
            model=model,
            messages=copy.deepcopy(messages),
            non_default_params=params,
            prompt_id=None,
            prompt_variables=None,
            dynamic_callback_params={},
        )

    @pytest.mark.parametrize(
        "model,expected",
        [
            ("gpt-5.6", True),
            ("openai/gpt-5.6", True),
            ("gpt-5.6-sol", True),
            ("gpt-5.6-luna", True),
            ("gpt-5.7", True),
            ("gpt-6", True),
            ("GPT-5.6", True),
            ("gpt-5.5", False),
            ("gpt-5", False),
            ("gpt-5-chat-latest", False),
            ("gpt-4.1", False),
            ("o3", False),
            ("claude-sonnet-4-5", False),
        ],
    )
    def test_model_support_truth_table(self, model, expected):
        assert supports_openai_prompt_cache_breakpoint(model) is expected

    @pytest.mark.parametrize(
        "model,provider,expected",
        [
            ("openai/gpt-5.6", None, True),
            ("gpt-5.6", None, True),
            ("gpt-5.6", "openai", True),
            ("gpt-5.6", "azure", False),
            ("azure/gpt-5.6", None, False),
            ("openai/gpt-4.1", None, False),
            ("anthropic/claude-sonnet-4-5", None, False),
            ("no-provider-can-route-this-model", None, False),
            (None, "openai", False),
        ],
    )
    def test_dialect_resolution(self, model, provider, expected):
        assert AnthropicCacheControlHook._targets_openai_prompt_cache_breakpoint(model, provider) is expected

    def test_count_covers_both_marker_kinds(self):
        message = {
            "role": "user",
            "cache_control": {"type": "ephemeral"},
            "content": [
                {"type": "text", "text": "a", "prompt_cache_breakpoint": self.EXPLICIT},
                {"type": "text", "text": "b", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "c"},
            ],
        }
        assert AnthropicCacheControlHook._count_cache_control_blocks(message) == 3

    def test_v1_messages_string_system_gets_block_breakpoint(self):
        kwargs = {"cache_control_injection_points": copy.deepcopy(self.SYSTEM_POINT)}
        messages, system = self._inject([{"role": "user", "content": "hi"}], "sys", kwargs)
        assert system == [{"type": "text", "text": "sys", "prompt_cache_breakpoint": self.EXPLICIT}]
        assert messages == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        assert kwargs == {"prompt_cache_options": self.EXPLICIT}
        assert not _contains_key(system, "cache_control")

    def test_v1_messages_list_system_marks_last_block_only(self):
        kwargs = {"cache_control_injection_points": copy.deepcopy(self.SYSTEM_POINT)}
        system = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
        _, result_system = self._inject([{"role": "user", "content": "hi"}], system, kwargs)
        assert result_system == [
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b", "prompt_cache_breakpoint": self.EXPLICIT},
        ]
        assert kwargs["prompt_cache_options"] == self.EXPLICIT

    def test_v1_messages_targets_by_role(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "reply"}]},
            {"role": "user", "content": "last"},
        ]
        kwargs = {"cache_control_injection_points": [{"location": "message", "role": "user"}]}
        result, _ = self._inject(messages, None, kwargs)
        assert result[0]["content"] == [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second", "prompt_cache_breakpoint": self.EXPLICIT},
        ]
        assert result[1] == messages[1]
        assert result[2]["content"] == [{"type": "text", "text": "last", "prompt_cache_breakpoint": self.EXPLICIT}]
        assert kwargs["prompt_cache_options"] == self.EXPLICIT

    def test_v1_messages_targets_by_index(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "first"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "reply"}]},
            {"role": "user", "content": [{"type": "text", "text": "last"}]},
        ]
        kwargs = {"cache_control_injection_points": [{"location": "message", "index": -1}]}
        result, _ = self._inject(messages, None, kwargs)
        assert result[:2] == messages[:2]
        assert result[2]["content"] == [{"type": "text", "text": "last", "prompt_cache_breakpoint": self.EXPLICIT}]

    def test_v1_messages_control_field_is_ignored(self):
        ttl_control = {"type": "ephemeral", "ttl": "1h"}
        kwargs = {
            "cache_control_injection_points": [
                {"location": "message", "role": "system", "control": ttl_control},
                {"location": "message", "index": -1, "control": ttl_control},
            ]
        }
        messages, system = self._inject([{"role": "user", "content": "hi"}], "sys", kwargs)
        assert system[0]["prompt_cache_breakpoint"] == self.EXPLICIT
        assert messages[0]["content"][-1]["prompt_cache_breakpoint"] == self.EXPLICIT
        assert not _contains_key(system, "cache_control")
        assert not _contains_key(messages, "cache_control")

    def test_v1_messages_keeps_caller_prompt_cache_options(self):
        caller_options = {"mode": "explicit", "ttl": "30m"}
        kwargs = {
            "cache_control_injection_points": copy.deepcopy(self.SYSTEM_POINT),
            "prompt_cache_options": dict(caller_options),
        }
        _, system = self._inject([{"role": "user", "content": "hi"}], "sys", kwargs)
        assert system[0]["prompt_cache_breakpoint"] == self.EXPLICIT
        assert kwargs["prompt_cache_options"] == caller_options

    def test_v1_messages_no_prompt_cache_options_when_nothing_injected(self):
        kwargs = {"cache_control_injection_points": copy.deepcopy(self.SYSTEM_POINT)}
        messages, system = self._inject([{"role": "user", "content": "hi"}], None, kwargs)
        assert system is None
        assert "prompt_cache_options" not in kwargs
        assert not _contains_key(messages, "prompt_cache_breakpoint")

    def test_v1_messages_anthropic_target_unchanged(self):
        kwargs = {"cache_control_injection_points": copy.deepcopy(self.SYSTEM_POINT)}
        _, system = self._inject(
            [{"role": "user", "content": "hi"}],
            "sys",
            kwargs,
            model="anthropic/claude-sonnet-4-5",
            custom_llm_provider="anthropic",
        )
        assert system == [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}]
        assert kwargs == {}

    def test_v1_messages_older_openai_model_keeps_cache_control(self):
        kwargs = {"cache_control_injection_points": copy.deepcopy(self.SYSTEM_POINT)}
        _, system = self._inject([{"role": "user", "content": "hi"}], "sys", kwargs, model="openai/gpt-4.1")
        assert system == [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}]
        assert kwargs == {}

    def test_v1_messages_client_content_breakpoint_makes_configured_points_stand_down(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "hi", "prompt_cache_breakpoint": self.EXPLICIT}]}]
        kwargs = {"cache_control_injection_points": copy.deepcopy(self.SYSTEM_POINT)}
        result, system = self._inject(messages, "sys", kwargs)
        assert result == messages
        assert system == "sys"
        assert kwargs == {}

    def test_v1_messages_client_system_breakpoint_makes_configured_points_stand_down(self):
        system = [{"type": "text", "text": "sys", "prompt_cache_breakpoint": self.EXPLICIT}]
        messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        kwargs = {"cache_control_injection_points": [{"location": "message", "index": -1}]}
        result, result_system = self._inject(messages, system, kwargs)
        assert result == messages
        assert result_system == system
        assert kwargs == {}

    def test_chat_system_string_wrapped_with_block_breakpoint(self):
        params = {"cache_control_injection_points": copy.deepcopy(self.SYSTEM_POINT)}
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        _, processed, returned = self._chat(messages, params)
        assert processed[0] == {
            "role": "system",
            "content": [{"type": "text", "text": "sys", "prompt_cache_breakpoint": self.EXPLICIT}],
        }
        assert processed[1] == {"role": "user", "content": "hi"}
        assert returned is params
        assert returned == {"prompt_cache_options": self.EXPLICIT}

    def test_chat_list_content_marks_last_block(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
                ],
            }
        ]
        params = {"cache_control_injection_points": [{"location": "message", "index": -1}]}
        _, processed, _ = self._chat(messages, params)
        assert processed[0]["content"] == [
            {"type": "text", "text": "look"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/a.png"},
                "prompt_cache_breakpoint": self.EXPLICIT,
            },
        ]
        assert params["prompt_cache_options"] == self.EXPLICIT

    def test_chat_unprefixed_model_resolves_to_openai(self):
        params = {"cache_control_injection_points": copy.deepcopy(self.SYSTEM_POINT)}
        _, processed, _ = self._chat([{"role": "system", "content": "sys"}], params, model="gpt-5.6")
        assert processed[0]["content"] == [{"type": "text", "text": "sys", "prompt_cache_breakpoint": self.EXPLICIT}]
        assert params["prompt_cache_options"] == self.EXPLICIT

    def test_chat_keeps_caller_prompt_cache_options(self):
        params = {
            "cache_control_injection_points": copy.deepcopy(self.SYSTEM_POINT),
            "prompt_cache_options": {"mode": "implicit"},
        }
        self._chat([{"role": "system", "content": "sys"}], params)
        assert params["prompt_cache_options"] == {"mode": "implicit"}

    def test_chat_no_prompt_cache_options_when_nothing_injected(self):
        params = {"cache_control_injection_points": copy.deepcopy(self.SYSTEM_POINT)}
        messages = [{"role": "user", "content": "hi"}]
        _, processed, _ = self._chat(messages, params)
        assert processed == messages
        assert params == {}

    @pytest.mark.parametrize("model", ["openai/gpt-4.1", "anthropic/claude-sonnet-4-5"])
    def test_chat_other_targets_keep_message_level_cache_control(self, model):
        params = {"cache_control_injection_points": copy.deepcopy(self.SYSTEM_POINT)}
        _, processed, _ = self._chat([{"role": "system", "content": "sys"}], params, model=model)
        assert processed[0] == {"role": "system", "content": "sys", "cache_control": {"type": "ephemeral"}}
        assert params == {}

    def test_chat_client_breakpoint_makes_seeded_points_stand_down(self):
        params = {"cache_control_injection_points": copy.deepcopy(self.SYSTEM_POINT)}
        AnthropicCacheControlHook.maybe_seed_default_injection_points(
            non_default_params=params,
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": [{"type": "text", "text": "hi", "prompt_cache_breakpoint": self.EXPLICIT}]},
            ],
            model="openai/gpt-5.6",
            custom_llm_provider="openai",
        )
        assert params == {}

    def test_cap_counts_client_breakpoints_of_both_kinds(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "a", "prompt_cache_breakpoint": self.EXPLICIT}]},
            {"role": "user", "content": [{"type": "text", "text": "b", "cache_control": {"type": "ephemeral"}}]},
            {"role": "user", "content": [{"type": "text", "text": "c", "prompt_cache_breakpoint": self.EXPLICIT}]},
            {"role": "user", "content": "d"},
            {"role": "user", "content": "e"},
        ]
        result = AnthropicCacheControlHook._apply_message_injections(
            points=[{"location": "message", "role": "user"}],
            messages=copy.deepcopy(messages),
            max_blocks=4,
            openai_dialect=True,
        )
        assert result[:3] == messages[:3]
        assert result[3]["content"] == [{"type": "text", "text": "d", "prompt_cache_breakpoint": self.EXPLICIT}]
        assert result[4] == {"role": "user", "content": "e"}


class TestOpenAIPromptCacheBreakpointPlacementRules:
    """OpenAI dialect only marks blocks OpenAI (and the /v1/messages bridges) can carry (#37509)."""

    EXPLICIT = {"mode": "explicit"}

    def _chat(self, messages, points, model="openai/gpt-5.6"):
        params = {"cache_control_injection_points": copy.deepcopy(points)}
        _, out, params = AnthropicCacheControlHook().get_chat_completion_prompt(
            model=model,
            messages=copy.deepcopy(messages),
            non_default_params=params,
            prompt_id=None,
            prompt_variables=None,
            dynamic_callback_params={},
        )
        return out, params

    def test_assistant_message_is_never_marked_on_chat_path(self):
        messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
        out, params = self._chat(messages, [{"location": "message", "role": "assistant"}])
        assert out == messages
        assert "prompt_cache_options" not in params

    def test_tool_message_text_is_marked_on_chat_path(self):
        messages = [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "w", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
        ]
        out, params = self._chat(messages, [{"location": "message", "index": -1}])
        assert out[2]["content"] == [{"type": "text", "text": "sunny", "prompt_cache_breakpoint": self.EXPLICIT}]
        assert params["prompt_cache_options"] == self.EXPLICIT

    def test_tool_result_only_turn_is_skipped_on_v1_messages(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "q"}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "w", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "sunny"}]},
        ]
        kwargs = {"cache_control_injection_points": [{"location": "message", "index": -1}]}
        out, system = AnthropicCacheControlHook.maybe_inject_cache_control(
            copy.deepcopy(messages), None, kwargs, model="openai/gpt-5.6"
        )
        assert out == messages
        assert system is None
        assert "prompt_cache_options" not in kwargs

    def test_assistant_turn_is_skipped_on_v1_messages(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "q"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "a"}]},
        ]
        kwargs = {"cache_control_injection_points": [{"location": "message", "role": "assistant"}]}
        out, _ = AnthropicCacheControlHook.maybe_inject_cache_control(
            copy.deepcopy(messages), None, kwargs, model="openai/gpt-5.6"
        )
        assert out == messages
        assert "prompt_cache_options" not in kwargs

    def test_text_after_tool_result_is_marked(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "sunny"},
                    {"type": "text", "text": "thanks"},
                ],
            }
        ]
        kwargs = {"cache_control_injection_points": [{"location": "message", "index": -1}]}
        out, _ = AnthropicCacheControlHook.maybe_inject_cache_control(messages, None, kwargs, model="openai/gpt-5.6")
        assert out[0]["content"] == [
            {"type": "tool_result", "tool_use_id": "t1", "content": "sunny"},
            {"type": "text", "text": "thanks", "prompt_cache_breakpoint": self.EXPLICIT},
        ]
        assert kwargs["prompt_cache_options"] == self.EXPLICIT

    def test_marker_walks_back_to_last_eligible_block(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "read this"},
                    {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "doc"}},
                ],
            }
        ]
        kwargs = {"cache_control_injection_points": [{"location": "message", "index": -1}]}
        out, _ = AnthropicCacheControlHook.maybe_inject_cache_control(messages, None, kwargs, model="openai/gpt-5.6")
        assert out[0]["content"][0] == {"type": "text", "text": "read this", "prompt_cache_breakpoint": self.EXPLICIT}
        assert "prompt_cache_breakpoint" not in out[0]["content"][1]

    def test_skipped_block_does_not_consume_a_slot(self):
        messages = [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t0", "content": "r"}]}] + [
            {"role": "user", "content": [{"type": "text", "text": f"m{i}"}]} for i in range(4)
        ]
        kwargs = {"cache_control_injection_points": [{"location": "message", "role": "user"}]}
        out, _ = AnthropicCacheControlHook.maybe_inject_cache_control(messages, None, kwargs, model="openai/gpt-5.6")
        assert "prompt_cache_breakpoint" not in out[0]["content"][0]
        assert all(msg["content"][0]["prompt_cache_breakpoint"] == self.EXPLICIT for msg in out[1:])


class TestChatPathProviderStamp:
    """The chat path learns the dialect decision (provider, api_base, opt-in) through the seeded points (#37509)."""

    POINTS = [{"location": "message", "role": "system"}]
    MESSAGES = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    ANTHROPIC_STYLE = {"role": "system", "content": "sys", "cache_control": {"type": "ephemeral"}}
    OPENAI_STYLE = [{"type": "text", "text": "sys", "prompt_cache_breakpoint": {"mode": "explicit"}}]
    CUSTOM_API_BASE = "http://127.0.0.1:9/v1"

    def _seed_and_run(self, model, custom_llm_provider, api_base=None, prompt_cache_options=None):
        params = {"cache_control_injection_points": copy.deepcopy(self.POINTS)}
        if prompt_cache_options is not None:
            params["prompt_cache_options"] = prompt_cache_options
        AnthropicCacheControlHook.maybe_seed_default_injection_points(
            non_default_params=params,
            messages=copy.deepcopy(self.MESSAGES),
            model=model,
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
        )
        return self._run(params, model)

    def _run(self, params, model):
        _, out, params = AnthropicCacheControlHook().get_chat_completion_prompt(
            model=model,
            messages=copy.deepcopy(self.MESSAGES),
            non_default_params=params,
            prompt_id=None,
            prompt_variables=None,
            dynamic_callback_params={},
        )
        return out, params

    def test_openai_compatible_provider_keeps_anthropic_style_markers(self):
        out, params = self._seed_and_run("gpt-5.6", "hosted_vllm")
        assert out[0] == {"role": "system", "content": "sys", "cache_control": {"type": "ephemeral"}}
        assert "prompt_cache_options" not in params

    def test_explicit_openai_provider_uses_openai_dialect(self):
        out, params = self._seed_and_run("gpt-5.6", "openai")
        assert out[0]["content"] == [{"type": "text", "text": "sys", "prompt_cache_breakpoint": {"mode": "explicit"}}]
        assert params["prompt_cache_options"] == {"mode": "explicit"}

    def test_bare_gpt_model_without_provider_resolves_to_openai(self):
        out, params = self._seed_and_run("gpt-5.6", None)
        assert out[0]["content"] == [{"type": "text", "text": "sys", "prompt_cache_breakpoint": {"mode": "explicit"}}]
        assert params["prompt_cache_options"] == {"mode": "explicit"}

    def test_points_keep_identity_for_models_below_gpt_5_6(self):
        points = copy.deepcopy(self.POINTS)
        params = {"cache_control_injection_points": points}
        AnthropicCacheControlHook.maybe_seed_default_injection_points(
            non_default_params=params,
            messages=copy.deepcopy(self.MESSAGES),
            model="anthropic/claude-sonnet-4-5",
            custom_llm_provider="anthropic",
        )
        assert params["cache_control_injection_points"] is points

    def test_provider_lookup_skipped_for_models_below_gpt_5_6(self):
        from unittest.mock import patch

        with patch.object(AnthropicCacheControlHook, "_resolve_provider") as resolve:
            assert AnthropicCacheControlHook._targets_openai_prompt_cache_breakpoint("gpt-4.1", None) is False
            assert AnthropicCacheControlHook._targets_openai_prompt_cache_breakpoint("my-custom-model", None) is False
        resolve.assert_not_called()

    def test_litellm_proxy_target_keeps_anthropic_style_markers(self):
        out, params = self._seed_and_run("litellm_proxy/gpt-5.6", None)
        assert out[0] == self.ANTHROPIC_STYLE
        assert "prompt_cache_options" not in params

    def test_custom_api_base_keeps_anthropic_style_markers(self):
        out, params = self._seed_and_run("gpt-5.6", None, api_base=self.CUSTOM_API_BASE)
        assert out[0] == self.ANTHROPIC_STYLE
        assert "prompt_cache_options" not in params

    def test_custom_api_base_opts_in_through_prompt_cache_options(self):
        out, params = self._seed_and_run(
            "gpt-5.6", None, api_base=self.CUSTOM_API_BASE, prompt_cache_options={"mode": "explicit"}
        )
        assert out[0]["content"] == self.OPENAI_STYLE
        assert params["prompt_cache_options"] == {"mode": "explicit"}

    def test_regional_openai_api_base_uses_openai_dialect(self):
        out, params = self._seed_and_run("gpt-5.6", None, api_base="https://eu.api.openai.com/v1")
        assert out[0]["content"] == self.OPENAI_STYLE
        assert params["prompt_cache_options"] == {"mode": "explicit"}

    @pytest.mark.parametrize("env_var", ["OPENAI_BASE_URL", "OPENAI_API_BASE"])
    def test_env_api_base_override_keeps_anthropic_style_markers(self, monkeypatch, env_var):
        monkeypatch.setenv(env_var, self.CUSTOM_API_BASE)
        out, params = self._seed_and_run("gpt-5.6", None)
        assert out[0] == self.ANTHROPIC_STYLE
        assert "prompt_cache_options" not in params

    def test_global_litellm_api_base_keeps_anthropic_style_markers(self, monkeypatch):
        monkeypatch.setattr(litellm, "api_base", self.CUSTOM_API_BASE)
        out, params = self._seed_and_run("gpt-5.6", None)
        assert out[0] == self.ANTHROPIC_STYLE
        assert "prompt_cache_options" not in params

    def test_request_api_base_wins_over_env_override(self, monkeypatch):
        monkeypatch.setenv("OPENAI_BASE_URL", self.CUSTOM_API_BASE)
        out, params = self._seed_and_run("gpt-5.6", None, api_base="https://api.openai.com/v1")
        assert out[0]["content"] == self.OPENAI_STYLE
        assert params["prompt_cache_options"] == {"mode": "explicit"}

    @pytest.mark.parametrize(
        "api_base,expected",
        [(None, True), ("http://127.0.0.1:9/v1", False), ("https://eu.api.openai.com/v1", True)],
    )
    def test_seed_stamps_the_dialect_decision(self, api_base, expected):
        params = {"cache_control_injection_points": copy.deepcopy(self.POINTS)}
        AnthropicCacheControlHook.maybe_seed_default_injection_points(
            non_default_params=params,
            messages=copy.deepcopy(self.MESSAGES),
            model="gpt-5.6",
            custom_llm_provider=None,
            api_base=api_base,
        )
        assert params["cache_control_injection_points"][0]["_litellm_openai_dialect"] is expected

    def test_stamp_is_authoritative_over_request_params(self):
        points = [{**self.POINTS[0], "_litellm_openai_dialect": False}]
        out, params = self._run({"cache_control_injection_points": points, "custom_llm_provider": "openai"}, "gpt-5.6")
        assert out[0] == self.ANTHROPIC_STYLE
        assert "prompt_cache_options" not in params

    def test_unstamped_points_read_api_base_from_request_params(self):
        params = {"cache_control_injection_points": copy.deepcopy(self.POINTS), "api_base": self.CUSTOM_API_BASE}
        out, params = self._run(params, "gpt-5.6")
        assert out[0] == self.ANTHROPIC_STYLE
        assert "prompt_cache_options" not in params

    def test_unstamped_points_read_prompt_cache_options_from_request_params(self):
        params = {
            "cache_control_injection_points": copy.deepcopy(self.POINTS),
            "api_base": self.CUSTOM_API_BASE,
            "prompt_cache_options": {"mode": "explicit"},
        }
        out, params = self._run(params, "gpt-5.6")
        assert out[0]["content"] == self.OPENAI_STYLE
        assert params["prompt_cache_options"] == {"mode": "explicit"}


class TestClientBreakpointsCountedOnce:
    def test_client_message_breakpoints_are_not_double_counted(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "m0", "cache_control": {"type": "ephemeral"}}]}] + [
            {"role": "user", "content": [{"type": "text", "text": f"m{i}"}]} for i in range(1, 4)
        ]
        out, system, _ = AnthropicCacheControlHook.apply_to_anthropic_messages_request(
            messages=messages,
            system="sys",
            injection_points=[
                {"location": "message", "role": "system"},
                {"location": "message", "index": -1},
                {"location": "message", "index": -2},
                {"location": "message", "index": -3},
            ],
        )
        marked = [msg["content"][0].get("cache_control") is not None for msg in out]
        assert marked == [True, False, True, True]
        assert system[0]["cache_control"] == {"type": "ephemeral"}


class TestResponsesInputPartsEligible:
    """Responses API input parts can carry prompt_cache_breakpoint on GPT-5.6+ (#37509)."""

    EXPLICIT = {"mode": "explicit"}

    def _chat(self, messages, points, model="openai/gpt-5.6"):
        params = {"cache_control_injection_points": copy.deepcopy(points)}
        _, out, params = AnthropicCacheControlHook().get_chat_completion_prompt(
            model=model,
            messages=copy.deepcopy(messages),
            non_default_params=params,
            prompt_id=None,
            prompt_variables=None,
            dynamic_callback_params={},
        )
        return out, params

    def test_marker_lands_on_last_input_text_part(self):
        messages = [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "first"}, {"type": "input_text", "text": "second"}],
            }
        ]
        out, params = self._chat(messages, [{"location": "message", "index": -1}])
        assert out[0]["content"][0] == {"type": "input_text", "text": "first"}
        assert out[0]["content"][1] == {
            "type": "input_text",
            "text": "second",
            "prompt_cache_breakpoint": self.EXPLICIT,
        }
        assert params["prompt_cache_options"] == self.EXPLICIT

    @pytest.mark.parametrize(
        "part",
        [
            {"type": "input_image", "image_url": "https://example.com/a.png"},
            {"type": "input_file", "file_id": "file_1"},
        ],
    )
    def test_input_image_and_input_file_parts_are_eligible(self, part):
        out, params = self._chat([{"role": "user", "content": [part]}], [{"location": "message", "index": -1}])
        assert out[0]["content"][0] == {**part, "prompt_cache_breakpoint": self.EXPLICIT}
        assert params["prompt_cache_options"] == self.EXPLICIT


class TestMessagesPathApiBaseGate:
    """/v1/messages only speaks the OpenAI dialect when the request really targets api.openai.com (#37509)."""

    EXPLICIT = {"mode": "explicit"}
    USER_POINT = [{"location": "message", "role": "user"}]
    MESSAGES = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    CUSTOM_API_BASE = "http://127.0.0.1:9/v1"
    CACHE_CONTROL_BLOCK = {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}
    BREAKPOINT_BLOCK = {"type": "text", "text": "hi", "prompt_cache_breakpoint": {"mode": "explicit"}}

    def _inject(self, model, api_base=None, prompt_cache_options=None, custom_llm_provider=None):
        kwargs = {"cache_control_injection_points": copy.deepcopy(self.USER_POINT)}
        if prompt_cache_options is not None:
            kwargs["prompt_cache_options"] = prompt_cache_options
        out, _ = AnthropicCacheControlHook.maybe_inject_cache_control(
            copy.deepcopy(self.MESSAGES),
            None,
            kwargs,
            model=model,
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
        )
        return out[0]["content"][0], kwargs

    def test_litellm_proxy_target_keeps_cache_control(self):
        block, kwargs = self._inject("gpt-5.6", api_base=self.CUSTOM_API_BASE, custom_llm_provider="litellm_proxy")
        assert block == self.CACHE_CONTROL_BLOCK
        assert "prompt_cache_options" not in kwargs

    def test_custom_api_base_keeps_cache_control(self):
        block, kwargs = self._inject("gpt-5.6", api_base=self.CUSTOM_API_BASE)
        assert block == self.CACHE_CONTROL_BLOCK
        assert "prompt_cache_options" not in kwargs

    def test_custom_api_base_opts_in_through_prompt_cache_options(self):
        block, kwargs = self._inject("gpt-5.6", api_base=self.CUSTOM_API_BASE, prompt_cache_options=self.EXPLICIT)
        assert block == self.BREAKPOINT_BLOCK
        assert kwargs["prompt_cache_options"] == self.EXPLICIT

    def test_regional_openai_api_base_uses_openai_dialect(self):
        block, kwargs = self._inject("gpt-5.6", api_base="https://eu.api.openai.com/v1")
        assert block == self.BREAKPOINT_BLOCK
        assert kwargs["prompt_cache_options"] == self.EXPLICIT

    def test_default_api_base_uses_openai_dialect(self):
        block, kwargs = self._inject("openai/gpt-5.6")
        assert block == self.BREAKPOINT_BLOCK
        assert kwargs["prompt_cache_options"] == self.EXPLICIT


class TestToolConfigSlotInOpenAIDialect:
    """OpenAI has no tool_config cache block, so the dialect does not hold a slot for one (#37509)."""

    EXPLICIT = {"mode": "explicit"}
    MESSAGES = [{"role": "user", "content": [{"type": "text", "text": f"m{i}"}]} for i in range(4)]
    POINTS = [{"location": "message", "index": i} for i in range(4)] + [{"location": "tool_config"}]

    def test_chat_path_marks_all_four_messages(self):
        params = {"cache_control_injection_points": copy.deepcopy(self.POINTS)}
        _, out, params = AnthropicCacheControlHook().get_chat_completion_prompt(
            model="openai/gpt-5.6",
            messages=copy.deepcopy(self.MESSAGES),
            non_default_params=params,
            prompt_id=None,
            prompt_variables=None,
            dynamic_callback_params={},
        )
        assert [msg["content"][0].get("prompt_cache_breakpoint") for msg in out] == [self.EXPLICIT] * 4
        assert params["prompt_cache_options"] == self.EXPLICIT

    def test_messages_path_marks_all_four_messages(self):
        out, _, _ = AnthropicCacheControlHook.apply_to_anthropic_messages_request(
            copy.deepcopy(self.MESSAGES), None, copy.deepcopy(self.POINTS), openai_dialect=True
        )
        assert [msg["content"][0].get("prompt_cache_breakpoint") for msg in out] == [self.EXPLICIT] * 4

    def test_anthropic_dialect_still_reserves_the_tool_config_slot(self):
        out, _, _ = AnthropicCacheControlHook.apply_to_anthropic_messages_request(
            copy.deepcopy(self.MESSAGES), None, copy.deepcopy(self.POINTS)
        )
        assert sum(msg["content"][0].get("cache_control") is not None for msg in out) == 3


class TestPromptCacheBreakpointCapability:
    """Eligibility comes from the model map's supports_prompt_cache_breakpoint flag when the entry carries one,
    with the GPT version rule for unlisted models and for entries the published map has not flagged yet (#37509)."""

    @pytest.fixture(autouse=True)
    def _bundled_model_map(self, monkeypatch):
        bundled = os.path.join(os.path.dirname(litellm.__file__), "model_prices_and_context_window_backup.json")
        with open(bundled) as handle:
            monkeypatch.setattr(litellm, "model_cost", json.load(handle))
        litellm.utils._cached_get_model_info_helper.cache_clear()
        yield
        litellm.utils._cached_get_model_info_helper.cache_clear()

    def test_public_helper_reads_the_model_map(self):
        from litellm.utils import supports_prompt_cache_breakpoint

        assert supports_prompt_cache_breakpoint("gpt-5.6") is True
        assert supports_prompt_cache_breakpoint("openai/gpt-5.6-sol") is True
        assert supports_prompt_cache_breakpoint("gpt-5.6", custom_llm_provider="openai") is True
        assert supports_prompt_cache_breakpoint("gpt-4.1") is False

    @pytest.mark.parametrize("model", ["gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"])
    def test_model_map_flags_every_openai_gpt_5_6_entry(self, model):
        assert litellm.model_cost[model]["litellm_provider"] == "openai"
        assert litellm.model_cost[model]["supports_prompt_cache_breakpoint"] is True

    def test_listed_model_uses_the_model_map_flag(self, monkeypatch):
        flagged = {**litellm.model_cost["gpt-4.1"], "supports_prompt_cache_breakpoint": True}
        monkeypatch.setitem(litellm.model_cost, "gpt-4.1", flagged)
        assert supports_openai_prompt_cache_breakpoint("gpt-4.1") is True

    def test_listed_gpt_5_6_without_the_flag_falls_back_to_the_version_rule(self, monkeypatch):
        unflagged = {k: v for k, v in litellm.model_cost["gpt-5.6"].items() if k != "supports_prompt_cache_breakpoint"}
        monkeypatch.setitem(litellm.model_cost, "gpt-5.6", unflagged)
        assert supports_openai_prompt_cache_breakpoint("gpt-5.6") is True
        assert supports_openai_prompt_cache_breakpoint("openai/gpt-5.6") is True

    def test_listed_model_flagged_false_is_not_eligible(self, monkeypatch):
        monkeypatch.setitem(
            litellm.model_cost, "gpt-5.6", {**litellm.model_cost["gpt-5.6"], "supports_prompt_cache_breakpoint": False}
        )
        assert supports_openai_prompt_cache_breakpoint("gpt-5.6") is False

    def test_listed_gpt_model_without_the_flag_follows_the_version_rule(self):
        assert "supports_prompt_cache_breakpoint" not in litellm.model_cost["gpt-4.1"]
        assert supports_openai_prompt_cache_breakpoint("gpt-4.1") is False

    def test_published_map_without_the_flag_still_injects_on_gpt_5_6(self, monkeypatch):
        unflagged = {k: v for k, v in litellm.model_cost["gpt-5.6"].items() if k != "supports_prompt_cache_breakpoint"}
        monkeypatch.setitem(litellm.model_cost, "gpt-5.6", unflagged)
        points = [{"location": "message", "role": "system"}]

        _, chat_messages, chat_params = AnthropicCacheControlHook().get_chat_completion_prompt(
            model="openai/gpt-5.6",
            messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
            non_default_params={"cache_control_injection_points": copy.deepcopy(points)},
            prompt_id=None,
            prompt_variables=None,
            dynamic_callback_params={},
        )
        assert chat_messages[0]["content"] == [
            {"type": "text", "text": "sys", "prompt_cache_breakpoint": {"mode": "explicit"}}
        ]
        assert chat_params["prompt_cache_options"] == {"mode": "explicit"}

        kwargs = {"cache_control_injection_points": copy.deepcopy(points)}
        _, system = AnthropicCacheControlHook.maybe_inject_cache_control(
            [{"role": "user", "content": "hi"}], "sys", kwargs, model="gpt-5.6", custom_llm_provider="openai"
        )
        assert system == [{"type": "text", "text": "sys", "prompt_cache_breakpoint": {"mode": "explicit"}}]
        assert kwargs == {"prompt_cache_options": {"mode": "explicit"}}

    @pytest.mark.parametrize("model,expected", [("gpt-5.6-2026-01-01", True), ("gpt-5.5-preview-unlisted", False)])
    def test_unlisted_model_falls_back_to_the_version_rule(self, model, expected):
        assert model not in litellm.model_cost
        assert supports_openai_prompt_cache_breakpoint(model) is expected
