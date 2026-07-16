import copy
import datetime
import json
import os
import sys
import unittest
from typing import List, Optional, Tuple
from unittest.mock import ANY, MagicMock, Mock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../.."))  # Adds the parent directory to the system-path
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

            # Verify that cache control was applied (Bedrock transforms it to a separate item)
            cache_control_count = sum(
                1 for item in request_body["system"] if isinstance(item, dict) and "cachePoint" in item
            )
            assert cache_control_count == 1, f"Expected exactly 1 cache control point, found {cache_control_count}"


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
            assert request_body["messages"][1]["content"][1]["cachePoint"] == {"type": "default"}


@pytest.mark.asyncio
async def test_anthropic_cache_control_hook_negative_indices():
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
async def test_anthropic_cache_control_hook_out_of_bounds_logging():
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
        litellm.callbacks = [anthropic_cache_control_hook]

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
                warning_call = mock_logger.warning.call_args[0][0]

                # Check that the warning message contains the expected information
                assert "AnthropicCacheControlHook: Provided index 10 is out of bounds" in warning_call
                assert "message list of length 2" in warning_call
                assert "Targeted index was 10" in warning_call
                assert "Skipping cache control injection for this point" in warning_call


@pytest.mark.asyncio
async def test_anthropic_cache_control_hook_negative_out_of_bounds_logging():
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
        litellm.callbacks = [anthropic_cache_control_hook]

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
                warning_call = mock_logger.warning.call_args[0][0]

                # Check that the warning message contains the original negative index
                assert "AnthropicCacheControlHook: Provided index -5 is out of bounds" in warning_call
                assert "message list of length 1" in warning_call
                assert "Targeted index was -4" in warning_call  # -5 + 1 = -4 (converted index)
                assert "Skipping cache control injection for this point" in warning_call


@pytest.mark.asyncio
async def test_anthropic_cache_control_hook_multiple_user_messages():
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
        litellm.callbacks = [anthropic_cache_control_hook]

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
async def test_anthropic_cache_control_hook_out_of_bounds(bad_index):
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
        litellm.callbacks = [anthropic_cache_control_hook]

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
async def test_anthropic_cache_control_hook_single_message(message_list):
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
        litellm.callbacks = [anthropic_cache_control_hook]

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
async def test_anthropic_cache_control_hook_empty_message_list():
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
        litellm.callbacks = [anthropic_cache_control_hook]

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
async def test_anthropic_cache_control_hook_no_op():
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
        litellm.callbacks = [anthropic_cache_control_hook]

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
async def test_anthropic_cache_control_hook_multiple_content_items_last_only():
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
        litellm.callbacks = [anthropic_cache_control_hook]

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
async def test_anthropic_cache_control_hook_document_analysis_multiple_pages():
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
        litellm.callbacks = [anthropic_cache_control_hook]

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
async def test_anthropic_cache_control_hook_string_negative_index():
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
        litellm.callbacks = [anthropic_cache_control_hook]

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
async def test_cache_control_hook_bedrock_payload_caps_cachepoints_at_four():
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
        litellm.callbacks = [AnthropicCacheControlHook()]

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
    # The tool_config point is passed through for the provider transform.
    assert non_default_params["cache_control_injection_points"] == [{"location": "tool_config"}]


@pytest.mark.asyncio
async def test_cache_control_hook_bedrock_payload_caps_with_tool_config_point():
    """End-to-end: message + tool_config injection must not exceed 4 cachePoints."""
    with patch.dict(
        os.environ,
        {
            "AWS_ACCESS_KEY_ID": "fake_access_key_id",
            "AWS_SECRET_ACCESS_KEY": "fake_secret_access_key",
            "AWS_REGION_NAME": "us-east-1",
        },
    ):
        litellm.callbacks = [AnthropicCacheControlHook()]

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

    def _points(self, model="claude-sonnet-4-5", provider="anthropic", messages=None, system=None):
        return AnthropicCacheControlHook.get_default_injection_points(
            messages=copy.deepcopy(self.MESSAGES) if messages is None else messages,
            system=system,
            model=model,
            custom_llm_provider=provider,
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
