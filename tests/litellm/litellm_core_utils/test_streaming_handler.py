import json
import os
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path
import asyncio
import traceback
from typing import Optional

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.utils import (
    Delta,
    ModelResponseStream,
    PromptTokensDetailsWrapper,
    StreamingChoices,
    Usage,
)
from litellm.utils import ModelResponseListIterator


@pytest.fixture
def initialized_custom_stream_wrapper() -> CustomStreamWrapper:
    streaming_handler = CustomStreamWrapper(
        completion_stream=None,
        model=None,
        logging_obj=MagicMock(),
        custom_llm_provider=None,
    )
    return streaming_handler


bedrock_chunks = [
    ModelResponseStream(
        id="chatcmpl-d249def8-a78b-464c-87b5-3a6f43565292",
        created=1742056047,
        model=None,
        object="chat.completion.chunk",
        system_fingerprint=None,
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(
                    provider_specific_fields=None,
                    content="I'm Claude",
                    role="assistant",
                    function_call=None,
                    tool_calls=None,
                    audio=None,
                ),
                logprobs=None,
            )
        ],
        provider_specific_fields={},
        usage=None,
    ),
    ModelResponseStream(
        id="chatcmpl-fe559823-b383-4249-ab87-52f6ad9d08c2",
        created=1742056047,
        model=None,
        object="chat.completion.chunk",
        system_fingerprint=None,
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(
                    provider_specific_fields=None,
                    content=", an AI",
                    role="assistant",
                    function_call=None,
                    tool_calls=None,
                    audio=None,
                ),
                logprobs=None,
            )
        ],
        provider_specific_fields={},
        usage=None,
    ),
    ModelResponseStream(
        id="chatcmpl-c1c6cc2f-75b9-4a24-88b9-4e5aacd0268b",
        created=1742056047,
        model=None,
        object="chat.completion.chunk",
        system_fingerprint=None,
        choices=[
            StreamingChoices(
                finish_reason="stop",
                index=0,
                delta=Delta(
                    provider_specific_fields=None,
                    content="",
                    role="assistant",
                    function_call=None,
                    tool_calls=None,
                    audio=None,
                ),
                logprobs=None,
            )
        ],
        provider_specific_fields={},
        usage=None,
    ),
]


def test_is_chunk_non_empty(initialized_custom_stream_wrapper: CustomStreamWrapper):
    """Unit test if non-empty when reasoning_content is present"""
    chunk = {
        "id": "e89b6501-8ac2-464c-9550-7cd3daf94350",
        "object": "chat.completion.chunk",
        "created": 1741037890,
        "model": "deepseek-reasoner",
        "system_fingerprint": "fp_5417b77867_prod0225",
        "choices": [
            {
                "index": 0,
                "delta": {"content": None, "reasoning_content": "."},
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    }
    assert initialized_custom_stream_wrapper.is_chunk_non_empty(
        completion_obj=MagicMock(),
        model_response=ModelResponseStream(**chunk),
        response_obj=MagicMock(),
    )


def test_optional_combine_thinking_block_in_choices(
    initialized_custom_stream_wrapper: CustomStreamWrapper,
):
    """Test that reasoning_content is properly combined with content using <think> tags"""
    # Setup the wrapper to use the merge feature
    initialized_custom_stream_wrapper.merge_reasoning_content_in_choices = True

    # First chunk with reasoning_content - should add <think> tag
    first_chunk = {
        "id": "chunk1",
        "object": "chat.completion.chunk",
        "created": 1741037890,
        "model": "deepseek-reasoner",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": "",
                    "reasoning_content": "Let me think about this",
                },
                "finish_reason": None,
            }
        ],
    }

    # Middle chunk with more reasoning_content
    middle_chunk = {
        "id": "chunk2",
        "object": "chat.completion.chunk",
        "created": 1741037891,
        "model": "deepseek-reasoner",
        "choices": [
            {
                "index": 0,
                "delta": {"content": "", "reasoning_content": " step by step"},
                "finish_reason": None,
            }
        ],
    }

    # Final chunk with actual content - should add </think> tag
    final_chunk = {
        "id": "chunk3",
        "object": "chat.completion.chunk",
        "created": 1741037892,
        "model": "deepseek-reasoner",
        "choices": [
            {
                "index": 0,
                "delta": {"content": "The answer is 42", "reasoning_content": None},
                "finish_reason": None,
            }
        ],
    }

    # Process first chunk
    first_response = ModelResponseStream(**first_chunk)
    initialized_custom_stream_wrapper._optional_combine_thinking_block_in_choices(
        first_response
    )
    print("first_response", json.dumps(first_response, indent=4, default=str))
    assert first_response.choices[0].delta.content == "<think>Let me think about this"
    # assert the response does not have attribute reasoning_content
    assert not hasattr(first_response.choices[0].delta, "reasoning_content")

    assert initialized_custom_stream_wrapper.sent_first_thinking_block is True

    # Process middle chunk
    middle_response = ModelResponseStream(**middle_chunk)
    initialized_custom_stream_wrapper._optional_combine_thinking_block_in_choices(
        middle_response
    )
    print("middle_response", json.dumps(middle_response, indent=4, default=str))
    assert middle_response.choices[0].delta.content == " step by step"
    assert not hasattr(middle_response.choices[0].delta, "reasoning_content")

    # Process final chunk
    final_response = ModelResponseStream(**final_chunk)
    initialized_custom_stream_wrapper._optional_combine_thinking_block_in_choices(
        final_response
    )
    print("final_response", json.dumps(final_response, indent=4, default=str))
    assert final_response.choices[0].delta.content == "</think>The answer is 42"
    assert initialized_custom_stream_wrapper.sent_last_thinking_block is True
    assert not hasattr(final_response.choices[0].delta, "reasoning_content")


def test_multi_chunk_reasoning_and_content(
    initialized_custom_stream_wrapper: CustomStreamWrapper,
):
    """Test handling of multiple reasoning chunks followed by multiple content chunks"""
    # Setup the wrapper to use the merge feature
    initialized_custom_stream_wrapper.merge_reasoning_content_in_choices = True
    initialized_custom_stream_wrapper.sent_first_thinking_block = False
    initialized_custom_stream_wrapper.sent_last_thinking_block = False

    # Create test chunks
    chunks = [
        # Chunk 1: First reasoning chunk
        {
            "id": "chunk1",
            "object": "chat.completion.chunk",
            "created": 1741037890,
            "model": "deepseek-reasoner",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": "",
                        "reasoning_content": "To solve this problem",
                    },
                    "finish_reason": None,
                }
            ],
        },
        # Chunk 2: Second reasoning chunk
        {
            "id": "chunk2",
            "object": "chat.completion.chunk",
            "created": 1741037891,
            "model": "deepseek-reasoner",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": "",
                        "reasoning_content": ", I need to calculate 6 * 7",
                    },
                    "finish_reason": None,
                }
            ],
        },
        # Chunk 3: Third reasoning chunk
        {
            "id": "chunk3",
            "object": "chat.completion.chunk",
            "created": 1741037892,
            "model": "deepseek-reasoner",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "", "reasoning_content": " which equals 42"},
                    "finish_reason": None,
                }
            ],
        },
        # Chunk 4: First content chunk (transition from reasoning to content)
        {
            "id": "chunk4",
            "object": "chat.completion.chunk",
            "created": 1741037893,
            "model": "deepseek-reasoner",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": "The answer to your question",
                        "reasoning_content": None,
                    },
                    "finish_reason": None,
                }
            ],
        },
        # Chunk 5: Second content chunk
        {
            "id": "chunk5",
            "object": "chat.completion.chunk",
            "created": 1741037894,
            "model": "deepseek-reasoner",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": " is 42.", "reasoning_content": None},
                    "finish_reason": None,
                }
            ],
        },
    ]

    # Expected content after processing each chunk
    expected_contents = [
        "<think>To solve this problem",
        ", I need to calculate 6 * 7",
        " which equals 42",
        "</think>The answer to your question",
        " is 42.",
    ]

    # Process each chunk and verify results
    for i, (chunk, expected_content) in enumerate(zip(chunks, expected_contents)):
        response = ModelResponseStream(**chunk)
        initialized_custom_stream_wrapper._optional_combine_thinking_block_in_choices(
            response
        )

        # Check content
        assert (
            response.choices[0].delta.content == expected_content
        ), f"Chunk {i+1}: content mismatch"

        # Check reasoning_content was removed
        assert not hasattr(
            response.choices[0].delta, "reasoning_content"
        ), f"Chunk {i+1}: reasoning_content should be removed"

    # Verify final state
    assert initialized_custom_stream_wrapper.sent_first_thinking_block is True
    assert initialized_custom_stream_wrapper.sent_last_thinking_block is True


def test_strip_sse_data_from_chunk():
    """Test the static method that strips 'data: ' prefix from SSE chunks"""
    # Test with string inputs
    assert CustomStreamWrapper._strip_sse_data_from_chunk("data: content") == "content"
    assert (
        CustomStreamWrapper._strip_sse_data_from_chunk("data:  spaced content")
        == " spaced content"
    )
    assert (
        CustomStreamWrapper._strip_sse_data_from_chunk("regular content")
        == "regular content"
    )
    assert (
        CustomStreamWrapper._strip_sse_data_from_chunk("regular content with data:")
        == "regular content with data:"
    )

    # Test with None input
    assert CustomStreamWrapper._strip_sse_data_from_chunk(None) is None


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_streaming_handler_with_usage(
    sync_mode: bool, final_usage_block: Optional[Usage] = None
):
    import time

    final_usage_block = final_usage_block or Usage(
        completion_tokens=392,
        prompt_tokens=1799,
        total_tokens=2191,
        completion_tokens_details=None,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            audio_tokens=None, cached_tokens=1796, text_tokens=None, image_tokens=None
        ),
    )
    final_chunk = ModelResponseStream(
        id="chatcmpl-87291500-d8c5-428e-b187-36fe5a4c97ab",
        created=1742056047,
        model=None,
        object="chat.completion.chunk",
        system_fingerprint=None,
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(
                    provider_specific_fields=None,
                    content="",
                    role="assistant",
                    function_call=None,
                    tool_calls=None,
                    audio=None,
                ),
                logprobs=None,
            )
        ],
        provider_specific_fields={},
        usage=final_usage_block,
    )
    test_chunks = bedrock_chunks + [final_chunk]
    completion_stream = ModelResponseListIterator(model_responses=test_chunks)

    response = CustomStreamWrapper(
        completion_stream=completion_stream,
        model="bedrock/claude-3-5-sonnet-20240620-v1:0",
        custom_llm_provider="bedrock",
        logging_obj=Logging(
            model="bedrock/claude-3-5-sonnet-20240620-v1:0",
            messages=[{"role": "user", "content": "Hey"}],
            stream=True,
            call_type="completion",
            start_time=time.time(),
            litellm_call_id="12345",
            function_id="1245",
        ),
        stream_options={"include_usage": True},
    )

    chunk_has_usage = False
    if sync_mode:
        for chunk in response:
            if hasattr(chunk, "usage"):
                assert chunk.usage == final_usage_block
            chunk_has_usage = True
    else:
        async for chunk in response:
            if hasattr(chunk, "usage"):
                assert chunk.usage == final_usage_block
            chunk_has_usage = True
    assert chunk_has_usage


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_streaming_with_usage_and_logging(sync_mode: bool):
    import time

    from litellm.integrations.custom_logger import CustomLogger

    class MockCallback(CustomLogger):
        pass

    mock_callback = MockCallback()
    litellm.success_callback = [mock_callback]
    litellm._async_success_callback = [mock_callback]

    final_usage_block = Usage(
        completion_tokens=392,
        prompt_tokens=1799,
        total_tokens=2191,
        completion_tokens_details=None,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            audio_tokens=None,
            cached_tokens=1796,
            text_tokens=None,
            image_tokens=None,
        ),
        cache_creation_input_tokens=0,
        cache_read_input_tokens=1796,
    )

    with patch.object(
        mock_callback, "log_success_event"
    ) as mock_log_success_event, patch.object(
        mock_callback, "log_stream_event"
    ) as mock_log_stream_event, patch.object(
        mock_callback, "async_log_success_event"
    ) as mock_async_log_success_event, patch.object(
        mock_callback, "async_log_stream_event"
    ) as mock_async_log_stream_event:
        await test_streaming_handler_with_usage(
            sync_mode=sync_mode, final_usage_block=final_usage_block
        )
        if sync_mode:
            time.sleep(1)
            mock_log_success_event.assert_called_once()
            # mock_log_stream_event.assert_called()
        else:
            await asyncio.sleep(1)
            mock_async_log_success_event.assert_called_once()
            # mock_async_log_stream_event.assert_called()

        print(mock_log_success_event.call_args.kwargs.keys())

        mock_log_success_event.call_args.kwargs[
            "response_obj"
        ].usage == final_usage_block


def test_streaming_handler_with_stop_chunk(
    initialized_custom_stream_wrapper: CustomStreamWrapper,
):
    args = {
        "completion_obj": {"content": ""},
        "response_obj": {
            "text": "",
            "is_finished": True,
            "finish_reason": "length",
            "logprobs": None,
            "original_chunk": ModelResponseStream(
                id="chatcmpl-ad517c2e-c197-48de-a2e6-a559cca48124",
                created=1742093326,
                model=None,
                object="chat.completion.chunk",
                system_fingerprint=None,
                choices=[
                    StreamingChoices(
                        finish_reason="length",
                        index=0,
                        delta=Delta(
                            provider_specific_fields=None,
                            content="",
                            role="assistant",
                            function_call=None,
                            tool_calls=None,
                            audio=None,
                        ),
                        logprobs=None,
                    )
                ],
                provider_specific_fields={},
                usage=None,
            ),
            "usage": None,
        },
    }

    returned_chunk = initialized_custom_stream_wrapper.return_processed_chunk_logic(
        **args, model_response=ModelResponseStream()
    )
    assert returned_chunk is None
