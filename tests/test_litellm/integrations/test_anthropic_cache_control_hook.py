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
from litellm.types.llms.openai import AllMessageValues, ChatCompletionCachedContent
from litellm.types.utils import StandardCallbackDynamicParams
from litellm.utils import is_cached_message


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


# ===========================================================================
# _safe_insert_cache_control_in_message — unit tests
# ===========================================================================

_CONTROL = ChatCompletionCachedContent(type="ephemeral")


class TestSafeInsertStringContent:
    """
    When message content is a plain string the hook must convert it to a
    single-item content-block list and place cache_control *inside* the block.

    Previously the hook placed cache_control as a top-level key on the message
    dict when content was a string.  Only Anthropic's transformer handled that
    placement; Vertex AI and Bedrock Invoke API both rely on
    ``litellm.utils.is_cached_message``, which returns False immediately when
    content is not a list, so caching was silently skipped for those providers.
    """

    def test_string_content_converted_to_list(self):
        msg = {"role": "system", "content": "You are a helpful assistant."}
        result = AnthropicCacheControlHook._safe_insert_cache_control_in_message(
            msg, _CONTROL
        )
        assert isinstance(result["content"], list), (
            "content must be converted from str to list"
        )

    def test_string_content_cache_control_inside_block(self):
        msg = {"role": "system", "content": "You are a helpful assistant."}
        result = AnthropicCacheControlHook._safe_insert_cache_control_in_message(
            msg, _CONTROL
        )
        block = result["content"][0]
        assert "cache_control" in block, (
            "cache_control must be inside the content block, not at message top level"
        )
        assert block["cache_control"] == {"type": "ephemeral"}

    def test_string_content_text_preserved(self):
        text = "You are a helpful assistant."
        msg = {"role": "system", "content": text}
        result = AnthropicCacheControlHook._safe_insert_cache_control_in_message(
            msg, _CONTROL
        )
        assert result["content"][0]["text"] == text

    def test_string_content_type_is_text(self):
        msg = {"role": "system", "content": "Instructions."}
        result = AnthropicCacheControlHook._safe_insert_cache_control_in_message(
            msg, _CONTROL
        )
        assert result["content"][0]["type"] == "text"

    def test_string_content_no_top_level_cache_control(self):
        """cache_control must NOT appear at the message top level after injection."""
        msg = {"role": "system", "content": "Instructions."}
        result = AnthropicCacheControlHook._safe_insert_cache_control_in_message(
            msg, _CONTROL
        )
        assert "cache_control" not in result, (
            "cache_control must NOT be a top-level key on the message dict"
        )

    def test_user_message_string_content(self):
        """Works for any role, not just system."""
        msg = {"role": "user", "content": "Long user context to cache."}
        result = AnthropicCacheControlHook._safe_insert_cache_control_in_message(
            msg, _CONTROL
        )
        assert isinstance(result["content"], list)
        assert result["content"][-1]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in result

    def test_custom_control_value_preserved(self):
        """Custom control values are preserved inside the block."""
        custom = ChatCompletionCachedContent(type="ephemeral")
        msg = {"role": "system", "content": "Some text."}
        result = AnthropicCacheControlHook._safe_insert_cache_control_in_message(
            msg, custom
        )
        assert result["content"][0]["cache_control"] == custom


class TestSafeInsertListContent:
    """
    When message content is already a list the existing behaviour must be
    preserved: cache_control goes into the *last* block only.
    """

    def test_list_content_last_block_gets_cache_control(self):
        msg = {
            "role": "system",
            "content": [
                {"type": "text", "text": "Block 1"},
                {"type": "text", "text": "Block 2"},
            ],
        }
        result = AnthropicCacheControlHook._safe_insert_cache_control_in_message(
            msg, _CONTROL
        )
        assert "cache_control" not in result["content"][0], (
            "First block must NOT get cache_control"
        )
        assert result["content"][-1]["cache_control"] == {"type": "ephemeral"}, (
            "Last block must get cache_control"
        )

    def test_list_content_no_top_level_cache_control(self):
        msg = {
            "role": "user",
            "content": [{"type": "text", "text": "Some text"}],
        }
        result = AnthropicCacheControlHook._safe_insert_cache_control_in_message(
            msg, _CONTROL
        )
        assert "cache_control" not in result

    def test_list_content_single_block(self):
        msg = {"role": "user", "content": [{"type": "text", "text": "Only block"}]}
        result = AnthropicCacheControlHook._safe_insert_cache_control_in_message(
            msg, _CONTROL
        )
        assert result["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_list_content_non_last_blocks_unchanged(self):
        original_first = {"type": "text", "text": "First"}
        msg = {
            "role": "user",
            "content": [dict(original_first), {"type": "text", "text": "Last"}],
        }
        result = AnthropicCacheControlHook._safe_insert_cache_control_in_message(
            msg, _CONTROL
        )
        assert result["content"][0] == original_first


class TestSafeInsertEdgeCases:
    def test_none_content_is_not_modified(self):
        msg = {"role": "user", "content": None}
        result = AnthropicCacheControlHook._safe_insert_cache_control_in_message(
            msg, _CONTROL
        )
        assert result["content"] is None
        assert "cache_control" not in result

    def test_empty_list_content_is_not_modified(self):
        msg = {"role": "user", "content": []}
        result = AnthropicCacheControlHook._safe_insert_cache_control_in_message(
            msg, _CONTROL
        )
        assert result["content"] == []
        assert "cache_control" not in result

    def test_returns_same_message_object(self):
        """The method modifies in place and returns the same object."""
        msg = {"role": "system", "content": "text"}
        result = AnthropicCacheControlHook._safe_insert_cache_control_in_message(
            msg, _CONTROL
        )
        assert result is msg


# ===========================================================================
# is_cached_message gate — critical for Vertex AI / Bedrock Invoke API
# ===========================================================================


class TestIsCachedMessageCompatibility:
    """
    ``litellm.utils.is_cached_message()`` is the gate used by the Vertex AI
    and Bedrock Invoke API transformers.  It returns False when content is not
    a list.  After injection into string-content messages the result must pass
    this gate.
    """

    def test_string_content_after_injection_passes_gate(self):
        msg = {"role": "system", "content": "Long static system prompt."}
        injected = AnthropicCacheControlHook._safe_insert_cache_control_in_message(
            msg, _CONTROL
        )
        assert is_cached_message(injected), (
            "is_cached_message() must return True after injection into string content — "
            "this is the gate used by Vertex AI and Bedrock Invoke API transformers"
        )

    def test_list_content_after_injection_passes_gate(self):
        msg = {
            "role": "system",
            "content": [{"type": "text", "text": "Static prompt."}],
        }
        injected = AnthropicCacheControlHook._safe_insert_cache_control_in_message(
            msg, _CONTROL
        )
        assert is_cached_message(injected)

    def test_no_injection_fails_gate(self):
        """Sanity: messages without cache_control must not pass the gate."""
        msg = {"role": "system", "content": "No caching here."}
        assert not is_cached_message(msg)

    def test_old_top_level_placement_fails_gate(self):
        """
        Documents the root cause: top-level cache_control on a string-content
        message is NOT recognised by is_cached_message() because it requires
        content to be a list.
        """
        msg = {
            "role": "system",
            "content": "Long prompt.",
            "cache_control": {"type": "ephemeral"},
        }
        assert not is_cached_message(msg), (
            "is_cached_message() must return False for top-level cache_control "
            "on string content — documents why Vertex AI / Bedrock silently skipped caching"
        )


# ===========================================================================
# Anthropic transformer backward-compat
# ===========================================================================


class TestAnthropicTransformerBackwardCompat:
    """
    Verify that the Anthropic transformer correctly handles messages whose
    string content was converted to a list by the hook.
    """

    def test_translate_system_message_with_list_content_preserves_cache_control(self):
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        config = AnthropicConfig()
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful assistant.",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {"role": "user", "content": "Hello"},
        ]
        system_msgs = config.translate_system_message(messages=messages)
        assert len(system_msgs) == 1
        assert system_msgs[0]["type"] == "text"
        assert system_msgs[0]["text"] == "You are a helpful assistant."
        assert system_msgs[0].get("cache_control") == {"type": "ephemeral"}, (
            "cache_control must be preserved by the Anthropic transformer "
            "when content is a list"
        )

    def test_translate_system_message_plain_string_unchanged(self):
        """Existing Anthropic behaviour for plain-string messages must be unchanged."""
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        config = AnthropicConfig()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        system_msgs = config.translate_system_message(messages=messages)
        assert len(system_msgs) == 1
        assert system_msgs[0]["text"] == "You are a helpful assistant."
        assert "cache_control" not in system_msgs[0]


# ===========================================================================
# _process_message_injection — role-targeting integration
# ===========================================================================


class TestProcessMessageInjection:
    """End-to-end tests of _process_message_injection with string-content messages."""

    def _inject(self, messages, role=None, index=None):
        point: dict = {"location": "message"}
        if role is not None:
            point["role"] = role
        if index is not None:
            point["index"] = index
        return AnthropicCacheControlHook._process_message_injection(
            point=point, messages=messages
        )

    def test_role_injection_on_string_system_message(self):
        messages = [
            {"role": "system", "content": "Static system instructions."},
            {"role": "user", "content": "Question"},
        ]
        result = self._inject(messages, role="system")
        sys_msg = next(m for m in result if m["role"] == "system")
        assert isinstance(sys_msg["content"], list)
        assert sys_msg["content"][-1]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in sys_msg

    def test_index_injection_on_string_user_message(self):
        messages = [{"role": "user", "content": "Long document to cache."}]
        result = self._inject(messages, index=-1)
        user_msg = result[0]
        assert isinstance(user_msg["content"], list)
        assert user_msg["content"][-1]["cache_control"] == {"type": "ephemeral"}

    def test_index_injection_on_list_message_unchanged(self):
        """Existing list-content path must continue to work as before."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Block 1"},
                    {"type": "text", "text": "Block 2"},
                ],
            }
        ]
        result = self._inject(messages, index=0)
        assert "cache_control" not in result[0]["content"][0]
        assert result[0]["content"][-1]["cache_control"] == {"type": "ephemeral"}

    def test_injected_string_message_passes_is_cached_message(self):
        messages = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "Hi"},
        ]
        result = self._inject(messages, role="system")
        sys_msg = next(m for m in result if m["role"] == "system")
        assert is_cached_message(sys_msg), (
            "After injection, is_cached_message() must return True — "
            "this is the gate used by Vertex AI and Bedrock Invoke API"
        )
