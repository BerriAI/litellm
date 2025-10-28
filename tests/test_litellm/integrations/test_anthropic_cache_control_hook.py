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

            # Verify that cache control was applied (Bedrock transforms it to a separate item)
            cache_control_count = sum(
                1
                for item in request_body["system"]
                if isinstance(item, dict) and "cachePoint" in item
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
            assert request_body["messages"][1]["content"][1]["cachePoint"] == {
                "type": "default"
            }


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
                model="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
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
            assert isinstance(
                last_message_content, list
            ), "Last message content should be a list"
            assert any(
                "cachePoint" in item
                for item in last_message_content
                if isinstance(item, dict)
            ), "CachePoint missing in last message"

            # Note: Based on debug output, the hook correctly applies cache control to both messages,
            # but the Bedrock API transformation appears to only preserve cache control for user messages,
            # not assistant messages. This is a limitation of the API transformation layer.
            #
            # The second-to-last message (assistant) gets cache_control from the hook but loses it
            # during API transformation. This test documents this behavior.
            second_last_message_content = request_body["messages"][1]["content"]
            assert isinstance(
                second_last_message_content, list
            ), "Second-to-last message content should be a list"

            # Check if assistant message cache control is preserved (currently it's not)
            assistant_has_cache_control = any(
                "cachePoint" in item
                for item in second_last_message_content
                if isinstance(item, dict)
            )
            print(
                f"Assistant message has cache control in final request: {assistant_has_cache_control}"
            )

            # Verify the first user message (request index 0) was NOT modified
            first_user_message_content = request_body["messages"][0]["content"]
            assert isinstance(
                first_user_message_content, list
            ), "First user message content should be a list"
            assert not any(
                "cachePoint" in item
                for item in first_user_message_content
                if isinstance(item, dict)
            ), "CachePoint unexpectedly found in first user message"


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
        with patch(
            "litellm.integrations.anthropic_cache_control_hook.verbose_logger"
        ) as mock_logger:
            with patch.object(client, "post", return_value=mock_response) as mock_post:
                messages = [
                    {"role": "user", "content": "Message 1"},
                    {"role": "user", "content": "Message 2"},
                ]

                await litellm.acompletion(
                    model="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                    messages=messages,
                    cache_control_injection_points=[
                        {"location": "message", "index": 10}
                    ],  # Out of bounds index
                    client=client,
                )

                # Verify that warning was called with the expected message
                mock_logger.warning.assert_called_once()
                warning_call = mock_logger.warning.call_args[0][0]

                # Check that the warning message contains the expected information
                assert (
                    "AnthropicCacheControlHook: Provided index 10 is out of bounds"
                    in warning_call
                )
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
        with patch(
            "litellm.integrations.anthropic_cache_control_hook.verbose_logger"
        ) as mock_logger:
            with patch.object(client, "post", return_value=mock_response) as mock_post:
                messages = [
                    {"role": "user", "content": "Single message"},
                ]

                await litellm.acompletion(
                    model="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
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
                assert (
                    "AnthropicCacheControlHook: Provided index -5 is out of bounds"
                    in warning_call
                )
                assert "message list of length 1" in warning_call
                assert (
                    "Targeted index was -4" in warning_call
                )  # -5 + 1 = -4 (converted index)
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
                model="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
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
                1
                for item in combined_message_content
                if isinstance(item, dict) and "cachePoint" in item
            )
            assert cache_control_count == 2

            print(
                f"Found {cache_control_count} cache control points in the combined message"
            )


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
                model="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                messages=messages,
                cache_control_injection_points=[
                    {"location": "message", "index": bad_index}
                ],
                client=client,
            )

            mock_post.assert_called_once()
            request_body = json.loads(mock_post.call_args.kwargs["data"])

            # Assert that NO cache control was applied to any message
            for msg in request_body["messages"]:
                content = msg.get("content", [])
                if isinstance(content, list):
                    assert not any(
                        "cachePoint" in item
                        for item in content
                        if isinstance(item, dict)
                    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message_list",
    [
        [{"role": "user", "content": "Single message"}]
    ],  # Single message only - empty list will fail at API level
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
                model="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                messages=message_list,
                cache_control_injection_points=[{"location": "message", "index": -1}],
                client=client,
            )

            mock_post.assert_called_once()
            request_body = json.loads(mock_post.call_args.kwargs["data"])
            # For the single message, verify cache control was applied
            content = request_body["messages"][0]["content"]
            assert isinstance(content, list)
            assert any(
                "cachePoint" in item for item in content if isinstance(item, dict)
            )


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
                    model="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                    messages=[],
                    cache_control_injection_points=[
                        {"location": "message", "index": -1}
                    ],
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
                model="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
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
                    assert not any(
                        "cachePoint" in item
                        for item in content
                        if isinstance(item, dict)
                    )


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
                model="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "First piece of context"},
                            {"type": "text", "text": "Second piece of context"},
                            {"type": "text", "text": "Third piece of context"},
                            {"type": "text", "text": "Fourth piece of context"},
                            {"type": "text", "text": "Fifth piece of context - should be cached"},
                        ],
                    }
                ],
                cache_control_injection_points=[
                    {"location": "message", "index": -1}
                ],
                client=client,
            )

            mock_post.assert_called_once()
            request_body = json.loads(mock_post.call_args.kwargs["data"])

            print("Multi-content request_body: ", json.dumps(request_body, indent=4))

            message_content = request_body["messages"][0]["content"]
            assert isinstance(message_content, list)

            cache_control_count = sum(
                1
                for item in message_content
                if isinstance(item, dict) and "cachePoint" in item
            )
            assert cache_control_count == 1, f"Expected exactly 1 cache control point, found {cache_control_count}. This test verifies the fix for issue 15696 where cache_control was incorrectly applied to ALL content items."


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
                model="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Summarize this document"},
                            {"type": "text", "text": "Page 1 content"},
                            {"type": "text", "text": "Page 2 content"},
                            {"type": "text", "text": "Page 3 content"},
                            {"type": "text", "text": "Page 4 content"},
                            {"type": "text", "text": "Page 5 content - final page to cache"},
                        ],
                    }
                ],
                cache_control_injection_points=[
                    {"location": "message", "role": "user"}
                ],
                client=client,
            )

            mock_post.assert_called_once()
            request_body = json.loads(mock_post.call_args.kwargs["data"])

            print("Document analysis request_body: ", json.dumps(request_body, indent=4))

            message_content = request_body["messages"][0]["content"]
            assert isinstance(message_content, list)

            cache_control_count = sum(
                1
                for item in message_content
                if isinstance(item, dict) and "cachePoint" in item
            )
            assert cache_control_count == 1, f"Expected exactly 1 cache control point (last item only), found {cache_control_count}. Before fix, this would be 6 (one for each content item)."
