import json
import os
import sys
import time
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
from litellm.litellm_core_utils.streaming_handler import (
    AUDIO_ATTRIBUTE,
    CustomStreamWrapper,
)
from litellm.types.utils import (
    CompletionTokensDetailsWrapper,
    Delta,
    ModelResponseStream,
    PromptTokensDetailsWrapper,
    StandardLoggingPayload,
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


@pytest.fixture
def logging_obj() -> Logging:
    import time

    logging_obj = Logging(
        model="my-random-model",
        messages=[{"role": "user", "content": "Hey"}],
        stream=True,
        call_type="completion",
        start_time=time.time(),
        litellm_call_id="12345",
        function_id="1245",
    )
    return logging_obj


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


def test_is_chunk_non_empty_with_annotations(
    initialized_custom_stream_wrapper: CustomStreamWrapper,
):
    """Unit test if non-empty when annotations are present"""
    chunk = {
        "id": "e89b6501-8ac2-464c-9550-7cd3daf94350",
        "object": "chat.completion.chunk",
        "created": 1741037890,
        "model": "deepseek-reasoner",
        "system_fingerprint": "fp_5417b77867_prod0225",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": None,
                    "annotations": [
                        {"type": "url_citation", "url": "https://www.google.com"}
                    ],
                },
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    }
    assert (
        initialized_custom_stream_wrapper.is_chunk_non_empty(
            completion_obj=MagicMock(),
            model_response=ModelResponseStream(**chunk),
            response_obj=MagicMock(),
        )
        is True
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
        completion_tokens_details=CompletionTokensDetailsWrapper(  # <-- This has a value
            accepted_prediction_tokens=None,
            audio_tokens=None,
            reasoning_tokens=0,
            rejected_prediction_tokens=None,
            text_tokens=None,
        ),
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


@pytest.mark.parametrize("sync_mode", [False])
@pytest.mark.asyncio
@pytest.mark.flaky(reruns=3)
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
        completion_tokens_details=CompletionTokensDetailsWrapper(
            accepted_prediction_tokens=None,
            audio_tokens=None,
            reasoning_tokens=0,
            rejected_prediction_tokens=None,
            text_tokens=None,
        ),
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
            assert (
                mock_log_success_event.call_args.kwargs["response_obj"].usage
                == final_usage_block
            )
        else:
            await asyncio.sleep(1)
            mock_async_log_success_event.assert_called_once()
            # mock_async_log_stream_event.assert_called()
            assert (
                mock_async_log_success_event.call_args.kwargs["response_obj"].usage
                == final_usage_block
            )

        print(mock_log_success_event.call_args.kwargs.keys())


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


def test_set_response_id_propagation_empty_to_valid(
    initialized_custom_stream_wrapper: CustomStreamWrapper,
):
    """Test that response_id is properly set when first chunk has empty ID and second chunk has valid ID"""

    model_response1 = ModelResponseStream(id="", created=1742056047, model=None)
    model_response1 = initialized_custom_stream_wrapper.set_model_id(
        model_response1.id, model_response1
    )
    assert model_response1.id == ""

    model_response2 = ModelResponseStream(
        id="valid-id-123", created=1742056048, model=None
    )
    model_response2 = initialized_custom_stream_wrapper.set_model_id(
        "valid-id-123", model_response2
    )
    assert model_response2.id == "valid-id-123"
    assert initialized_custom_stream_wrapper.response_id == "valid-id-123"


def test_set_response_id_propagation_valid_to_invalid(
    initialized_custom_stream_wrapper: CustomStreamWrapper,
):
    """Test that response_id is maintained when first chunk has valid ID and second chunk has invalid ID"""

    model_response1 = ModelResponseStream(
        id="first-valid-id", created=1742056049, model=None
    )
    model_response1 = initialized_custom_stream_wrapper.set_model_id(
        "first-valid-id", model_response1
    )
    assert model_response1.id == "first-valid-id"
    assert initialized_custom_stream_wrapper.response_id == "first-valid-id"

    model_response2 = ModelResponseStream(id="", created=1742056050, model=None)
    model_response2 = initialized_custom_stream_wrapper.set_model_id(
        "", model_response2
    )
    assert model_response2.id == "first-valid-id"
    assert initialized_custom_stream_wrapper.response_id == "first-valid-id"


@pytest.mark.asyncio
async def test_streaming_completion_start_time(logging_obj: Logging):
    """Test that the start time is set correctly"""
    from litellm.integrations.custom_logger import CustomLogger

    class MockCallback(CustomLogger):
        pass

    mock_callback = MockCallback()
    litellm.success_callback = [mock_callback, "langfuse"]

    completion_stream = ModelResponseListIterator(
        model_responses=bedrock_chunks, delay=0.1
    )

    response = CustomStreamWrapper(
        completion_stream=completion_stream,
        model="bedrock/claude-3-5-sonnet-20240620-v1:0",
        logging_obj=logging_obj,
    )

    async for chunk in response:
        print(chunk)

    await asyncio.sleep(2)

    assert logging_obj.model_call_details["completion_start_time"] is not None
    assert (
        logging_obj.model_call_details["completion_start_time"]
        < logging_obj.model_call_details["end_time"]
    )


def test_streaming_handler_with_created_time_propagation(
    initialized_custom_stream_wrapper: CustomStreamWrapper, logging_obj: Logging
):
    """Test that the created time is consistent across chunks"""
    import time

    bad_chunk = ModelResponseStream(
        choices=[], created=int(time.time())
    )  # chunk with different created time

    completion_stream = ModelResponseListIterator(
        model_responses=bedrock_chunks + [bad_chunk]
    )

    response = CustomStreamWrapper(
        completion_stream=completion_stream,
        model="bedrock/claude-3-5-sonnet-20240620-v1:0",
        logging_obj=logging_obj,
    )

    created: Optional[int] = None
    for chunk in response:
        if created is None:
            created = chunk.created
        else:
            assert created == chunk.created


def test_streaming_handler_with_stream_options(
    initialized_custom_stream_wrapper: CustomStreamWrapper,
):
    """Test that the stream options are propagated to the response"""

    mr = initialized_custom_stream_wrapper.model_response_creator()
    mr_dict = mr.model_dump()
    print(mr_dict)
    assert "stream_options" not in mr_dict


def test_optional_combine_thinking_block_with_none_content(
    initialized_custom_stream_wrapper: CustomStreamWrapper,
):
    """Test that reasoning_content is properly combined when delta.content is None"""
    # Setup the wrapper to use the merge feature
    initialized_custom_stream_wrapper.merge_reasoning_content_in_choices = True

    # First chunk with reasoning_content and None content - should handle None gracefully
    first_chunk = {
        "id": "chunk1",
        "object": "chat.completion.chunk",
        "created": 1741037890,
        "model": "deepseek-reasoner",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": None,  # This is None, not empty string
                    "reasoning_content": "Let me think about this problem",
                },
                "finish_reason": None,
            }
        ],
    }

    # Second chunk with reasoning_content and None content
    second_chunk = {
        "id": "chunk2",
        "object": "chat.completion.chunk",
        "created": 1741037891,
        "model": "deepseek-reasoner",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": None,  # This is None, not empty string
                    "reasoning_content": " step by step",
                },
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

    # Process first chunk - should not raise TypeError
    first_response = ModelResponseStream(**first_chunk)
    initialized_custom_stream_wrapper._optional_combine_thinking_block_in_choices(
        first_response
    )
    assert (
        first_response.choices[0].delta.content
        == "<think>Let me think about this problem"
    )
    assert not hasattr(first_response.choices[0].delta, "reasoning_content")
    assert initialized_custom_stream_wrapper.sent_first_thinking_block is True

    # Process second chunk - should work with continued reasoning
    second_response = ModelResponseStream(**second_chunk)
    initialized_custom_stream_wrapper._optional_combine_thinking_block_in_choices(
        second_response
    )
    assert second_response.choices[0].delta.content == " step by step"
    assert not hasattr(second_response.choices[0].delta, "reasoning_content")

    # Process final chunk - should add </think> tag
    final_response = ModelResponseStream(**final_chunk)
    initialized_custom_stream_wrapper._optional_combine_thinking_block_in_choices(
        final_response
    )
    assert final_response.choices[0].delta.content == "</think>The answer is 42"
    assert initialized_custom_stream_wrapper.sent_last_thinking_block is True
    assert not hasattr(final_response.choices[0].delta, "reasoning_content")


def test_has_special_delta_content(
    initialized_custom_stream_wrapper: CustomStreamWrapper,
):
    """Test the _has_special_delta_content helper method"""

    # Test empty choices
    empty_response = ModelResponseStream(
        id="test", created=1742056047, model=None, choices=[]
    )
    assert not initialized_custom_stream_wrapper._has_special_delta_content(
        empty_response
    )

    # Test with tool_calls (simulate with mock object)
    tool_call_response = ModelResponseStream(
        id="test",
        created=1742056047,
        model=None,
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(
                    content=None,
                    tool_calls=[
                        {
                            "id": "test",
                            "function": {"arguments": "{}", "name": "test_func"},
                        }
                    ],
                ),
            )
        ],
    )
    assert initialized_custom_stream_wrapper._has_special_delta_content(
        tool_call_response
    )

    # Test with function_call (simulate with mock object)
    function_call_response = ModelResponseStream(
        id="test",
        created=1742056047,
        model=None,
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(
                    content=None, function_call={"name": "test_func", "arguments": "{}"}
                ),
            )
        ],
    )
    assert initialized_custom_stream_wrapper._has_special_delta_content(
        function_call_response
    )

    # Test with audio (simulate by adding audio attribute)
    audio_response = ModelResponseStream(
        id="test",
        created=1742056047,
        model=None,
        choices=[
            StreamingChoices(finish_reason=None, index=0, delta=Delta(content=None))
        ],
    )
    # Manually add audio attribute to delta
    audio_response.choices[0].delta.audio = {"transcript": "test"}
    assert initialized_custom_stream_wrapper._has_special_delta_content(audio_response)

    # Test with image (simulate by adding image attribute)
    image_response = ModelResponseStream(
        id="test",
        created=1742056047,
        model=None,
        choices=[
            StreamingChoices(finish_reason=None, index=0, delta=Delta(content=None))
        ],
    )
    # Manually add image attribute to delta
    image_response.choices[0].delta.images = [{"url": "test.jpg"}]
    assert initialized_custom_stream_wrapper._has_special_delta_content(image_response)

    # Test with regular content (should return False)
    regular_response = ModelResponseStream(
        id="test",
        created=1742056047,
        model=None,
        choices=[
            StreamingChoices(
                finish_reason=None, index=0, delta=Delta(content="Hello world")
            )
        ],
    )
    assert not initialized_custom_stream_wrapper._has_special_delta_content(
        regular_response
    )


def test_handle_special_delta_content(
    initialized_custom_stream_wrapper: CustomStreamWrapper,
):
    """Test the _handle_special_delta_content helper method"""
    test_response = ModelResponseStream(
        id="test",
        created=1742056047,
        model=None,
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(content="test", role="assistant"),
            )
        ],
    )

    # The method should call strip_role_from_delta
    result = initialized_custom_stream_wrapper._handle_special_delta_content(
        test_response
    )

    # Should return the same response object (modified)
    assert result is test_response

    # Should have set sent_first_chunk to True
    assert initialized_custom_stream_wrapper.sent_first_chunk is True


def test_has_any_special_delta_attributes(
    initialized_custom_stream_wrapper: CustomStreamWrapper,
):
    """Test the _has_any_special_delta_attributes helper method"""

    # Test with delta that has audio attribute
    class MockDelta:
        def __init__(self):
            self.audio = {"transcript": "Hello world"}

    audio_delta = MockDelta()
    result = initialized_custom_stream_wrapper._has_any_special_delta_attributes(
        audio_delta
    )
    assert result is True

    # Test with delta that has image attribute
    class MockDeltaImage:
        def __init__(self):
            self.images = [{"url": "test.jpg"}]

    image_delta = MockDeltaImage()
    result = initialized_custom_stream_wrapper._has_any_special_delta_attributes(
        image_delta
    )
    assert result is True

    # Test with delta that has no special attributes
    class MockDeltaRegular:
        def __init__(self):
            self.content = "regular content"

    regular_delta = MockDeltaRegular()
    result = initialized_custom_stream_wrapper._has_any_special_delta_attributes(
        regular_delta
    )
    assert result is False


def test_handle_special_delta_attributes(
    initialized_custom_stream_wrapper: CustomStreamWrapper,
):
    """Test the _handle_special_delta_attributes helper method"""

    # Create a model response
    model_response = ModelResponseStream(
        id="test",
        created=1742056047,
        model=None,
        choices=[
            StreamingChoices(finish_reason=None, index=0, delta=Delta(content="test"))
        ],
    )

    # Test with delta that has audio attribute
    class MockDelta:
        def __init__(self):
            self.audio = {"transcript": "Hello world"}

    audio_delta = MockDelta()
    initialized_custom_stream_wrapper._handle_special_delta_attributes(
        audio_delta, model_response
    )

    # Should copy the audio attribute
    assert hasattr(model_response.choices[0].delta, "audio")
    assert model_response.choices[0].delta.audio == {"transcript": "Hello world"}

    # Test with delta that has image attribute
    class MockDeltaImage:
        def __init__(self):
            self.images = [{"url": "test.jpg"}]

    image_delta = MockDeltaImage()
    model_response2 = ModelResponseStream(
        id="test",
        created=1742056047,
        model=None,
        choices=[
            StreamingChoices(finish_reason=None, index=0, delta=Delta(content="test"))
        ],
    )

    initialized_custom_stream_wrapper._handle_special_delta_attributes(
        image_delta, model_response2
    )

    # Should copy the image attribute
    print(f"delta: {model_response2.choices[0].delta}")
    assert hasattr(model_response2.choices[0].delta, "images")
    print(f"images: {model_response2.choices[0].delta.images}")
    assert model_response2.choices[0].delta.images[0] == {"url": "test.jpg"}


def test_has_special_delta_attribute(
    initialized_custom_stream_wrapper: CustomStreamWrapper,
):
    """Test the _has_special_delta_attribute helper method"""

    # Test with None delta
    assert not initialized_custom_stream_wrapper._has_special_delta_attribute(
        None, "audio"
    )

    # Test with delta that has the attribute
    class MockDelta:
        def __init__(self):
            self.audio = {"transcript": "test"}

    delta_with_audio = MockDelta()
    assert initialized_custom_stream_wrapper._has_special_delta_attribute(
        delta_with_audio, "audio"
    )

    # Test with delta that doesn't have the attribute
    class MockDeltaNoAudio:
        def __init__(self):
            self.content = "test"

    delta_without_audio = MockDeltaNoAudio()
    assert not initialized_custom_stream_wrapper._has_special_delta_attribute(
        delta_without_audio, "audio"
    )

    # Test with delta that has the attribute but it's None
    class MockDeltaNone:
        def __init__(self):
            self.audio = None

    delta_with_none = MockDeltaNone()
    assert not initialized_custom_stream_wrapper._has_special_delta_attribute(
        delta_with_none, "audio"
    )
