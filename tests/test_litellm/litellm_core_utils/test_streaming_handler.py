import json
import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch

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
    _ProviderChunkEarlyReturn,
    _ProviderChunkParsed,
)
from litellm.types.utils import (
    ChoiceLogprobs,
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
        model="bedrock/claude-haiku-4-5-20251001-v1:0",
        custom_llm_provider="bedrock",
        logging_obj=Logging(
            model="bedrock/claude-haiku-4-5-20251001-v1:0",
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

    with (
        patch.object(mock_callback, "log_success_event") as mock_log_success_event,
        patch.object(mock_callback, "log_stream_event") as mock_log_stream_event,
        patch.object(
            mock_callback, "async_log_success_event"
        ) as mock_async_log_success_event,
        patch.object(
            mock_callback, "async_log_stream_event"
        ) as mock_async_log_stream_event,
    ):
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


def test_finish_reason_chunk_preserves_non_openai_attributes(
    initialized_custom_stream_wrapper: CustomStreamWrapper,
):
    """
    Regression test for #23444:
    Preserve upstream non-OpenAI attributes on final finish_reason chunk.
    """
    initialized_custom_stream_wrapper.received_finish_reason = "stop"

    original_chunk = ModelResponseStream(
        id="chatcmpl-test",
        created=1742093326,
        model=None,
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason="stop",
                index=0,
                delta=Delta(content=""),
                logprobs=None,
            )
        ],
    )
    setattr(original_chunk, "custom_field", {"key": "value"})

    returned_chunk = initialized_custom_stream_wrapper.return_processed_chunk_logic(
        completion_obj={"content": ""},
        response_obj={"original_chunk": original_chunk},
        model_response=ModelResponseStream(),
    )

    assert returned_chunk is not None
    assert getattr(returned_chunk, "custom_field", None) == {"key": "value"}


def test_finish_reason_with_holding_chunk_preserves_non_openai_attributes(
    initialized_custom_stream_wrapper: CustomStreamWrapper,
):
    """
    Regression test for #23444 holding-chunk path:
    preserve custom attributes when _is_delta_empty is False after flushing
    holding_chunk.
    """
    initialized_custom_stream_wrapper.received_finish_reason = "stop"
    initialized_custom_stream_wrapper.holding_chunk = "filtered text"

    original_chunk = ModelResponseStream(
        id="chatcmpl-test-2",
        created=1742093327,
        model=None,
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason="stop",
                index=0,
                delta=Delta(content=""),
                logprobs=None,
            )
        ],
    )
    setattr(original_chunk, "custom_field", {"key": "value"})

    returned_chunk = initialized_custom_stream_wrapper.return_processed_chunk_logic(
        completion_obj={"content": ""},
        response_obj={"original_chunk": original_chunk},
        model_response=ModelResponseStream(),
    )

    assert returned_chunk is not None
    assert returned_chunk.choices[0].delta.content == "filtered text"
    assert getattr(returned_chunk, "custom_field", None) == {"key": "value"}


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
        model="bedrock/claude-haiku-4-5-20251001-v1:0",
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


@pytest.mark.asyncio
async def test_vertex_streaming_bad_request_not_midstream(logging_obj: Logging):
    """Ensure Vertex bad request errors surface as 400, not mid-stream fallbacks."""
    from litellm.llms.vertex_ai.common_utils import VertexAIError

    async def _raise_bad_request(**kwargs):
        raise VertexAIError(
            status_code=400, message="invalid maxOutputTokens", headers=None
        )

    response = CustomStreamWrapper(
        completion_stream=None,
        model="gemini-3-pro-preview",
        logging_obj=logging_obj,
        custom_llm_provider="vertex_ai_beta",
        make_call=_raise_bad_request,
    )

    with pytest.raises(litellm.BadRequestError) as excinfo:
        await response.__anext__()

    assert getattr(excinfo.value, "status_code", None) == 400
    assert "invalid maxOutputTokens" in str(excinfo.value)


@pytest.mark.asyncio
async def test_vertex_streaming_rate_limit_triggers_midstream_fallback(
    logging_obj: Logging,
):
    """Ensure Vertex 429 rate-limit errors raise MidStreamFallbackError, not RateLimitError.

    Regression test for https://github.com/BerriAI/litellm/issues/20870
    """
    from litellm.exceptions import MidStreamFallbackError
    from litellm.llms.vertex_ai.common_utils import VertexAIError

    async def _raise_rate_limit(**kwargs):
        raise VertexAIError(
            status_code=429, message="Resource exhausted.", headers=None
        )

    response = CustomStreamWrapper(
        completion_stream=None,
        model="gemini-3-flash-preview",
        logging_obj=logging_obj,
        custom_llm_provider="vertex_ai_beta",
        make_call=_raise_rate_limit,
    )

    with pytest.raises(MidStreamFallbackError) as excinfo:
        await response.__anext__()

    assert excinfo.value.is_pre_first_chunk is True
    assert excinfo.value.generated_content == ""


def test_sync_streaming_rate_limit_triggers_midstream_fallback(logging_obj: Logging):
    """Ensure __next__ raises MidStreamFallbackError on 429, not RateLimitError.

    This is the sync-streaming equivalent of the async test above.  Before
    this fix, __next__ would raise RateLimitError directly, bypassing the
    Router's fallback chain entirely.
    """
    from litellm.exceptions import MidStreamFallbackError
    from litellm.llms.vertex_ai.common_utils import VertexAIError

    def _raise_rate_limit(**kwargs):
        raise VertexAIError(
            status_code=429, message="Resource exhausted.", headers=None
        )

    response = CustomStreamWrapper(
        completion_stream=None,
        model="gemini-3-flash-preview",
        logging_obj=logging_obj,
        custom_llm_provider="vertex_ai_beta",
        make_call=_raise_rate_limit,
    )

    with pytest.raises(MidStreamFallbackError) as excinfo:
        next(response)

    assert excinfo.value.is_pre_first_chunk is True
    assert excinfo.value.generated_content == ""


def test_sync_streaming_bad_request_not_midstream(logging_obj: Logging):
    """Ensure __next__ raises BadRequestError (400) directly, not MidStreamFallbackError.

    Non-retriable 4xx errors should surface immediately to the caller.
    """
    from litellm.llms.vertex_ai.common_utils import VertexAIError

    def _raise_bad_request(**kwargs):
        raise VertexAIError(
            status_code=400, message="invalid maxOutputTokens", headers=None
        )

    response = CustomStreamWrapper(
        completion_stream=None,
        model="gemini-3-pro-preview",
        logging_obj=logging_obj,
        custom_llm_provider="vertex_ai_beta",
        make_call=_raise_bad_request,
    )

    with pytest.raises(litellm.BadRequestError) as excinfo:
        next(response)

    assert getattr(excinfo.value, "status_code", None) == 400
    assert "invalid maxOutputTokens" in str(excinfo.value)


def _bedrock_error_event(exception_type: str):
    """A mocked botocore event-stream error event: status_code is botocore's
    hard-coded 400, with the real type in the :exception-type header."""
    event = Mock()
    event.to_response_dict = Mock(
        return_value={
            "status_code": 400,
            "headers": {
                ":exception-type": exception_type,
                ":content-type": "application/json",
                ":message-type": "exception",
            },
            "body": b'{"message":"Bedrock had an internal error."}',
        }
    )
    return event


@pytest.mark.asyncio
async def test_bedrock_midstream_internal_server_error_wraps_for_fallback(
    logging_obj: Logging,
):
    """End-to-end regression for https://github.com/BerriAI/litellm/issues/24608:
    a Bedrock mid-stream internalServerException event (botocore stamps it 400)
    must flow through the real decoder, gain its modeled 500 status, and wrap
    into MidStreamFallbackError so the Router can run streaming fallback.

    Calls the real AWSEventStreamDecoder, so reverting the decoder status fix
    makes the decoder raise BedrockError(400) and the gate raises BadRequestError
    directly -> this test fails without the fix."""
    from litellm.exceptions import MidStreamFallbackError
    from litellm.llms.bedrock.chat.invoke_handler import AWSEventStreamDecoder

    decoder = AWSEventStreamDecoder(model="anthropic.claude-3-sonnet-20240229-v1:0")

    async def _bedrock_stream():
        decoder._parse_message_from_event(
            _bedrock_error_event("internalServerException")
        )
        yield  # unreachable; the line above raises

    async def _make_call(**kwargs):
        return _bedrock_stream()

    response = CustomStreamWrapper(
        completion_stream=None,
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        logging_obj=logging_obj,
        custom_llm_provider="bedrock",
        make_call=_make_call,
    )

    with pytest.raises(MidStreamFallbackError):
        await response.__anext__()


@pytest.mark.asyncio
async def test_bedrock_5xx_wraps_for_midstream_fallback(logging_obj: Logging):
    """Gate contract: a Bedrock 5xx (here 503 serviceUnavailableException) wraps
    into MidStreamFallbackError so the Router can run streaming fallback."""
    from litellm.exceptions import MidStreamFallbackError
    from litellm.llms.bedrock.chat.invoke_handler import BedrockError

    async def _raise_503(**kwargs):
        raise BedrockError(
            status_code=503,
            message="serviceUnavailableException Bedrock is unavailable.",
        )

    response = CustomStreamWrapper(
        completion_stream=None,
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        logging_obj=logging_obj,
        custom_llm_provider="bedrock",
        make_call=_raise_503,
    )

    with pytest.raises(MidStreamFallbackError):
        await response.__anext__()


@pytest.mark.asyncio
async def test_bedrock_validation_error_raises_directly(logging_obj: Logging):
    """Gate contract: a Bedrock validationException (400) is a client error and
    must surface directly, never wrapped into MidStreamFallbackError."""
    from litellm.exceptions import MidStreamFallbackError
    from litellm.llms.bedrock.chat.invoke_handler import BedrockError

    async def _raise_400(**kwargs):
        raise BedrockError(
            status_code=400,
            message="validationException malformed input.",
        )

    response = CustomStreamWrapper(
        completion_stream=None,
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        logging_obj=logging_obj,
        custom_llm_provider="bedrock",
        make_call=_raise_400,
    )

    with pytest.raises(Exception) as excinfo:
        await response.__anext__()
    assert not isinstance(excinfo.value, MidStreamFallbackError)
    assert getattr(excinfo.value, "status_code", None) == 400


def _hosted_vllm_stream_wrapper(logging_obj: Logging, error_payload: dict) -> CustomStreamWrapper:
    """A CustomStreamWrapper over the real OpenAI-compatible line iterator,
    fed an HTTP 200 SSE body that carries an in-body error payload the way
    vLLM/sglang emit it."""
    from litellm.llms.openai.chat.gpt_transformation import (
        OpenAIChatCompletionStreamingHandler,
    )

    async def _stream():
        yield f"data: {json.dumps(error_payload)}"
        yield "data: [DONE]"

    completion_stream = OpenAIChatCompletionStreamingHandler(
        streaming_response=_stream(), sync_stream=False
    )
    return CustomStreamWrapper(
        completion_stream=completion_stream,
        model="qwen-vl",
        logging_obj=logging_obj,
        custom_llm_provider="hosted_vllm",
    )


@pytest.mark.asyncio
async def test_in_body_stream_error_400_raises_bad_request(logging_obj: Logging):
    """Regression for https://github.com/BerriAI/litellm/issues/25492: a 400
    error returned inside a 200 SSE body must surface as BadRequestError with
    the provider's message, not be parsed as an empty chunk that silently
    ends the stream (and never as an internal MidStreamFallbackError)."""
    from litellm.exceptions import MidStreamFallbackError

    response = _hosted_vllm_stream_wrapper(
        logging_obj,
        {
            "error": {
                "object": "error",
                "message": "The model is not multimodal. Please remove image inputs.",
                "type": "BadRequestError",
                "param": None,
                "code": 400,
            }
        },
    )

    with pytest.raises(litellm.BadRequestError) as excinfo:
        await response.__anext__()

    assert not isinstance(excinfo.value, MidStreamFallbackError)
    assert excinfo.value.status_code == 400
    assert "not multimodal" in str(excinfo.value)


@pytest.mark.asyncio
async def test_in_body_stream_error_500_wraps_for_midstream_fallback(
    logging_obj: Logging,
):
    """An in-body 5xx error wraps into MidStreamFallbackError so the Router's
    FallbackStreamWrapper can switch to a configured fallback deployment."""
    from litellm.exceptions import MidStreamFallbackError

    response = _hosted_vllm_stream_wrapper(
        logging_obj,
        {
            "error": {
                "object": "error",
                "message": "internal engine crash",
                "type": "InternalServerError",
                "param": None,
                "code": 500,
            }
        },
    )

    with pytest.raises(MidStreamFallbackError) as excinfo:
        await response.__anext__()

    assert excinfo.value.is_pre_first_chunk is True
    assert "internal engine crash" in str(excinfo.value)


@pytest.mark.asyncio
async def test_async_streaming_read_timeout_triggers_midstream_fallback(
    logging_obj: Logging,
):
    """A mid-stream httpx.ReadTimeout must wrap into MidStreamFallbackError so
    the Router's FallbackStreamWrapper can switch to a fallback model.

    Previously __anext__ caught httpx.TimeoutException and re-raised it raw,
    which bypassed _handle_stream_fallback_error and prevented stream_timeout
    from triggering fallbacks the way connection-phase timeout does.
    """
    import httpx

    from litellm.exceptions import MidStreamFallbackError

    async def _raise_read_timeout(**kwargs):
        raise httpx.ReadTimeout("Timeout on reading data from socket")

    response = CustomStreamWrapper(
        completion_stream=None,
        model="gpt-4",
        logging_obj=logging_obj,
        custom_llm_provider="openai",
        make_call=_raise_read_timeout,
    )

    with pytest.raises(MidStreamFallbackError) as excinfo:
        await response.__anext__()

    assert excinfo.value.is_pre_first_chunk is True
    assert isinstance(excinfo.value.original_exception, Exception)


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
        model="bedrock/claude-haiku-4-5-20251001-v1:0",
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


def test_calculate_total_usage_with_cost():
    from litellm.litellm_core_utils.streaming_handler import calculate_total_usage

    chunk1_usage = Usage(completion_tokens=1, prompt_tokens=10, total_tokens=11)
    chunk1 = ModelResponseStream(
        id="test-1",
        created=1745513206,
        model="openrouter/test",
        choices=[
            StreamingChoices(finish_reason=None, index=0, delta=Delta(content="Hi"))
        ],
        usage=chunk1_usage,
    )

    chunk2_usage = Usage(
        completion_tokens=5, prompt_tokens=10, total_tokens=15, cost=0.00025
    )
    chunk2 = ModelResponseStream(
        id="test-1",
        created=1745513207,
        model="openrouter/test",
        choices=[
            StreamingChoices(finish_reason="stop", index=0, delta=Delta(content=""))
        ],
        usage=chunk2_usage,
    )

    usage = calculate_total_usage([chunk1, chunk2])

    assert hasattr(usage, "cost")
    assert usage.cost == 0.00025
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 5


def test_calculate_total_usage_with_dict_usage_cost():
    """Regression: dict-shaped `usage` with a `cost` key must still surface
    provider cost even though `hasattr` on a dict does not consult its keys."""
    from litellm.litellm_core_utils.streaming_handler import calculate_total_usage

    chunk = {
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "cost": 0.00025,
        }
    }

    usage = calculate_total_usage([chunk])

    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 5
    assert getattr(usage, "cost", None) == 0.00025


@pytest.mark.asyncio
async def test_openrouter_streaming_cost_after_finish_reason(logging_obj: Logging):
    from litellm.utils import ModelResponseListIterator

    chunk1 = ModelResponseStream(
        id="chatcmpl-or",
        created=1742056047,
        model="openrouter/claude",
        choices=[
            StreamingChoices(
                finish_reason=None, index=0, delta=Delta(content="Hi", role="assistant")
            )
        ],
        usage=None,
    )
    chunk2 = ModelResponseStream(
        id="chatcmpl-or",
        created=1742056048,
        model="openrouter/claude",
        choices=[
            StreamingChoices(finish_reason="stop", index=0, delta=Delta(content=""))
        ],
        usage=None,
    )
    chunk3_usage = Usage(
        completion_tokens=5, prompt_tokens=10, total_tokens=15, cost=0.00025
    )
    chunk3 = ModelResponseStream(
        id="chatcmpl-or",
        created=1742056049,
        model="openrouter/claude",
        choices=[
            StreamingChoices(finish_reason=None, index=0, delta=Delta(content=""))
        ],
        usage=chunk3_usage,
    )

    completion_stream = ModelResponseListIterator(
        model_responses=[chunk1, chunk2, chunk3]
    )
    response = CustomStreamWrapper(
        completion_stream=completion_stream,
        model="openrouter/claude",
        custom_llm_provider="openrouter",
        logging_obj=logging_obj,
        stream_options={"include_usage": True},
    )

    collected_chunks = []
    async for chunk in response:
        collected_chunks.append(chunk)

    usage_chunks = [c for c in collected_chunks if hasattr(c, "usage") and c.usage]
    assert len(usage_chunks) > 0
    assert hasattr(usage_chunks[-1].usage, "cost")
    assert usage_chunks[-1].usage.cost == 0.00025


def test_openrouter_streaming_cost_propagates_to_hidden_params():
    """
    Verify that provider-reported cost from usage.cost flows into
    _hidden_params["additional_headers"]["llm_provider-x-litellm-response-cost"]
    on the complete streaming response, so litellm's cost calculator uses it.
    """
    import litellm

    chunk1 = ModelResponseStream(
        id="chatcmpl-or",
        created=1742056047,
        model="openrouter/claude",
        choices=[
            StreamingChoices(
                finish_reason=None, index=0, delta=Delta(content="Hi", role="assistant")
            )
        ],
        usage=None,
    )
    chunk2 = ModelResponseStream(
        id="chatcmpl-or",
        created=1742056048,
        model="openrouter/claude",
        choices=[
            StreamingChoices(finish_reason="stop", index=0, delta=Delta(content=""))
        ],
        usage=None,
    )
    chunk3 = ModelResponseStream(
        id="chatcmpl-or",
        created=1742056049,
        model="openrouter/claude",
        choices=[
            StreamingChoices(finish_reason=None, index=0, delta=Delta(content=""))
        ],
        usage=Usage(
            completion_tokens=5, prompt_tokens=10, total_tokens=15, cost=0.00025
        ),
    )

    # Build the complete response as stream_chunk_builder does
    complete_response = litellm.stream_chunk_builder(
        chunks=[chunk1, chunk2, chunk3],
        messages=[{"role": "user", "content": "test"}],
    )

    assert complete_response is not None
    assert hasattr(complete_response.usage, "cost")
    assert complete_response.usage.cost == 0.00025

    # Use the real propagation method from CustomStreamWrapper
    CustomStreamWrapper._propagate_usage_cost_to_hidden_params(complete_response)

    assert "additional_headers" in complete_response._hidden_params
    assert (
        complete_response._hidden_params["additional_headers"][
            "llm_provider-x-litellm-response-cost"
        ]
        == 0.00025
    )

    # Verify the cost calculator would pick this up
    from litellm.cost_calculator import get_response_cost_from_hidden_params

    provider_cost = get_response_cost_from_hidden_params(
        complete_response._hidden_params
    )
    assert provider_cost == 0.00025


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


def test_is_chunk_non_empty_with_empty_tool_calls(
    initialized_custom_stream_wrapper: CustomStreamWrapper,
):
    """
    Test that is_chunk_non_empty returns False when tool_calls is an empty list.

    Regression test for https://github.com/BerriAI/litellm/issues/17425
    Empty tool_calls in delta should not be considered non-empty chunks.
    """
    chunk = {
        "id": "test-chunk-id",
        "object": "chat.completion.chunk",
        "created": 1741037890,
        "model": "claude-sonnet-4-20250514",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": None,
                    "tool_calls": [],  # Empty tool_calls list
                },
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    }
    # Empty tool_calls should return False
    assert (
        initialized_custom_stream_wrapper.is_chunk_non_empty(
            completion_obj={},  # completion_obj has no tool_calls
            model_response=ModelResponseStream(**chunk),
            response_obj={},
        )
        is False
    )


def test_is_chunk_non_empty_with_valid_tool_calls(
    initialized_custom_stream_wrapper: CustomStreamWrapper,
):
    """
    Test that is_chunk_non_empty returns True when tool_calls has valid entries.

    Companion test for https://github.com/BerriAI/litellm/issues/17425
    Non-empty tool_calls in delta should be considered non-empty chunks.
    """
    chunk = {
        "id": "test-chunk-id",
        "object": "chat.completion.chunk",
        "created": 1741037890,
        "model": "claude-sonnet-4-20250514",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "NYC"}',
                            },
                        }
                    ],
                },
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    }
    # Non-empty tool_calls should return True
    assert (
        initialized_custom_stream_wrapper.is_chunk_non_empty(
            completion_obj={},
            model_response=ModelResponseStream(**chunk),
            response_obj={},
        )
        is True
    )


def _make_chunk(content: Optional[str]) -> ModelResponseStream:
    return ModelResponseStream(
        id="test",
        created=1741037890,
        model="test-model",
        choices=[StreamingChoices(index=0, delta=Delta(content=content))],
    )


def _build_chunks(pattern: list[str], N: int) -> list[ModelResponseStream]:
    """
    Build a list of chunks based on a pattern specification.
    """
    chunks = []
    for i, p in enumerate(pattern):
        if p == "same":
            chunks.append(_make_chunk("same_chunk"))
        elif p == "diff":
            chunks.append(_make_chunk(f"chunk_{i}"))
        else:
            chunks.append(_make_chunk(p))
    return chunks


_REPETITION_TEST_CASES = [
    # Basic cases
    pytest.param(
        ["same"] * litellm.REPEATED_STREAMING_CHUNK_LIMIT,
        True,
        id="all_identical_raises",
    ),
    pytest.param(
        ["same"] * (litellm.REPEATED_STREAMING_CHUNK_LIMIT - 1),
        False,
        id="below_threshold_no_raise",
    ),
    pytest.param(
        [None] * litellm.REPEATED_STREAMING_CHUNK_LIMIT,
        False,
        id="none_content_no_raise",
    ),
    pytest.param(
        [""] * litellm.REPEATED_STREAMING_CHUNK_LIMIT,
        False,
        id="empty_content_no_raise",
    ),
    # Short content (len <= 2) should not raise
    pytest.param(
        ["##"] * litellm.REPEATED_STREAMING_CHUNK_LIMIT,
        False,
        id="short_content_2chars_no_raise",
    ),
    pytest.param(
        ["{"] * litellm.REPEATED_STREAMING_CHUNK_LIMIT,
        False,
        id="short_content_1char_no_raise",
    ),
    pytest.param(
        ["ab"] * litellm.REPEATED_STREAMING_CHUNK_LIMIT,
        False,
        id="short_content_2chars_ab_no_raise",
    ),
    # All different chunks
    pytest.param(
        ["diff"] * litellm.REPEATED_STREAMING_CHUNK_LIMIT,
        False,
        id="all_different_no_raise",
    ),
    # One chunk different at various positions
    pytest.param(
        ["different_first"] + ["same"] * (litellm.REPEATED_STREAMING_CHUNK_LIMIT - 1),
        False,
        id="first_chunk_different_no_raise",
    ),
    pytest.param(
        ["same"] * (litellm.REPEATED_STREAMING_CHUNK_LIMIT - 1) + ["different_last"],
        False,
        id="last_chunk_different_no_raise",
    ),
    pytest.param(
        ["same"] * (litellm.REPEATED_STREAMING_CHUNK_LIMIT // 2 + 1)
        + ["different_mid"]
        + ["same"]
        * (
            litellm.REPEATED_STREAMING_CHUNK_LIMIT
            - litellm.REPEATED_STREAMING_CHUNK_LIMIT // 2
            + 1
        ),
        False,
        id="middle_chunk_different_no_raise",
    ),
    pytest.param(
        ["same"] * (litellm.REPEATED_STREAMING_CHUNK_LIMIT - 2) + ["diff", "diff"],
        False,
        id="last_two_different_no_raise",
    ),
    pytest.param(
        ["diff"] * litellm.REPEATED_STREAMING_CHUNK_LIMIT
        + ["same"] * litellm.REPEATED_STREAMING_CHUNK_LIMIT
        + ["diff"],
        True,
        id="in_between_same_and_diff_raise",
    ),
]


@pytest.mark.parametrize("chunks_pattern,should_raise", _REPETITION_TEST_CASES)
def test_raise_on_model_repetition(
    initialized_custom_stream_wrapper: CustomStreamWrapper,
    chunks_pattern: list,
    should_raise: bool,
):
    wrapper = initialized_custom_stream_wrapper
    chunks = _build_chunks(chunks_pattern, len(chunks_pattern))

    if should_raise:
        with pytest.raises(litellm.InternalServerError) as exc_info:
            for chunk in chunks:
                wrapper.chunks.append(chunk)
                wrapper.raise_on_model_repetition()
        assert "repeating the same chunk" in str(exc_info.value)
    else:
        for chunk in chunks:
            wrapper.chunks.append(chunk)
            wrapper.raise_on_model_repetition()


@pytest.mark.parametrize(
    "empty_chunk_index",
    [-1, -2],
    ids=["last_chunk_empty", "second_to_last_chunk_empty"],
)
def test_raise_on_model_repetition_tolerates_empty_choices(
    initialized_custom_stream_wrapper: CustomStreamWrapper,
    empty_chunk_index: int,
):
    """
    Regression test for https://github.com/BerriAI/litellm/issues/28884

    Vertex Gemini Flash / Flash Lite with web search streaming emits
    metadata-only and usage-only chunks that carry no choices. These are
    appended to self.chunks, and raise_on_model_repetition() previously
    accessed choices[0] unconditionally, raising IndexError mid-stream
    (surfaced to users as MidStreamFallbackError -> APIConnectionError).
    """
    wrapper = initialized_custom_stream_wrapper

    chunks = [
        _make_chunk("hello world"),
        ModelResponseStream(
            id="usage-only",
            created=1741037890,
            model="vertex_ai/gemini-3.1-flash-lite",
            choices=[],
            usage=Usage(prompt_tokens=10, completion_tokens=0, total_tokens=10),
        ),
    ]
    if empty_chunk_index == -2:
        chunks.append(_make_chunk("hello world again"))

    for chunk in chunks:
        wrapper.chunks.append(chunk)
        wrapper.raise_on_model_repetition()


def test_usage_chunk_after_finish_reason_updates_hidden_params(logging_obj):
    """
    Test that provider-reported usage from a post-finish_reason chunk
    is surfaced in _hidden_params even when stream_options is NOT set.

    Reproduces issue #20760: OpenRouter sends a final chunk with usage data
    after the finish_reason chunk.  The hidden_params["usage"] on the last
    user-visible chunk was being calculated before this usage chunk arrived,
    resulting in zeros.  The fix recalculates it in the StopIteration handler
    after stream_chunk_builder processes all chunks.
    """
    # Simulate OpenRouter's actual streaming pattern:
    # 1) content chunk
    # 2) finish_reason chunk  (content="")
    # 3) usage chunk  (content="", finish_reason=None, usage={...})
    chunks = [
        ModelResponseStream(
            id="gen-abc",
            object="chat.completion.chunk",
            created=1000000,
            model="openrouter/openai/gpt-4o-mini",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(role="assistant", content="Hello"),
                    finish_reason=None,
                )
            ],
        ),
        ModelResponseStream(
            id="gen-abc",
            object="chat.completion.chunk",
            created=1000000,
            model="openrouter/openai/gpt-4o-mini",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(content=""),
                    finish_reason="stop",
                )
            ],
        ),
        ModelResponseStream(
            id="gen-abc",
            object="chat.completion.chunk",
            created=1000000,
            model="openrouter/openai/gpt-4o-mini",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(role="assistant", content=""),
                    finish_reason=None,
                )
            ],
            usage=Usage(
                prompt_tokens=20,
                completion_tokens=135,
                total_tokens=155,
            ),
        ),
    ]

    # Create a CustomStreamWrapper with NO stream_options
    wrapper = CustomStreamWrapper(
        completion_stream=ModelResponseListIterator(model_responses=chunks),
        model="openrouter/openai/gpt-4o-mini",
        logging_obj=logging_obj,
        custom_llm_provider="openrouter",
        stream_options=None,
    )

    # Consume the stream
    collected = []
    for chunk in wrapper:
        collected.append(chunk)

    # The last user-visible chunk's _hidden_params["usage"] should
    # contain the provider-reported values, not zeros.
    last_chunk = collected[-1]
    hidden_usage = last_chunk._hidden_params.get("usage")
    assert hidden_usage is not None, "Expected usage in _hidden_params"
    assert (
        hidden_usage.prompt_tokens == 20
    ), f"Expected prompt_tokens=20 from provider, got {hidden_usage.prompt_tokens}"
    assert (
        hidden_usage.completion_tokens == 135
    ), f"Expected completion_tokens=135 from provider, got {hidden_usage.completion_tokens}"


@pytest.mark.asyncio
async def test_custom_stream_wrapper_aclose():
    """Test that aclose() delegates to the underlying completion_stream's aclose()"""
    mock_stream = AsyncMock()
    mock_stream.aclose = AsyncMock()

    wrapper = CustomStreamWrapper(
        completion_stream=mock_stream,
        model=None,
        logging_obj=MagicMock(),
        custom_llm_provider=None,
    )

    await wrapper.aclose()
    mock_stream.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_custom_stream_wrapper_aclose_no_underlying():
    """Test that aclose() is safe when completion_stream has no aclose method"""
    mock_stream = MagicMock(spec=[])  # No aclose attribute

    wrapper = CustomStreamWrapper(
        completion_stream=mock_stream,
        model=None,
        logging_obj=MagicMock(),
        custom_llm_provider=None,
    )

    # Should not raise
    await wrapper.aclose()


@pytest.mark.asyncio
async def test_custom_stream_wrapper_aclose_none_stream():
    """Test that aclose() is safe when completion_stream is None"""
    wrapper = CustomStreamWrapper(
        completion_stream=None,
        model=None,
        logging_obj=MagicMock(),
        custom_llm_provider=None,
    )

    # Should not raise
    await wrapper.aclose()


def test_content_not_dropped_when_finish_reason_already_set(
    initialized_custom_stream_wrapper: CustomStreamWrapper,
):
    """
    Regression test for #22098: Vertex AI Claude streaming truncation.

    When content_block_delta and message_delta arrive in rapid succession,
    received_finish_reason can be set BEFORE the last content chunk is
    processed. The old code raised StopIteration unconditionally, dropping
    content. The fix checks for text/tool_use content before stopping.
    """
    initialized_custom_stream_wrapper.received_finish_reason = "stop"
    initialized_custom_stream_wrapper.custom_llm_provider = "anthropic"

    content_chunk = {
        "text": "world!",
        "tool_use": None,
        "is_finished": False,
        "finish_reason": "",
        "usage": None,
        "index": 0,
    }

    result = initialized_custom_stream_wrapper.chunk_creator(chunk=content_chunk)

    assert (
        result is not None
    ), "chunk_creator() returned None — content was dropped (issue #22098)"
    assert result.choices[0].delta.content == "world!"


def test_empty_chunk_still_stops_after_finish_reason_set(
    initialized_custom_stream_wrapper: CustomStreamWrapper,
):
    """
    Companion test for #22098: an empty GenericStreamingChunk must still
    raise StopIteration when received_finish_reason is already set.
    """
    initialized_custom_stream_wrapper.received_finish_reason = "stop"
    initialized_custom_stream_wrapper.custom_llm_provider = "anthropic"

    empty_chunk = {
        "text": "",
        "tool_use": None,
        "is_finished": False,
        "finish_reason": "",
        "usage": None,
        "index": 0,
    }

    with pytest.raises(StopIteration):
        initialized_custom_stream_wrapper.chunk_creator(chunk=empty_chunk)


def test_tool_use_not_dropped_when_finish_reason_already_set(
    initialized_custom_stream_wrapper: CustomStreamWrapper,
):
    """
    Regression test for #22098: tool_use-only chunks must not be dropped
    when received_finish_reason is already set.
    """
    initialized_custom_stream_wrapper.received_finish_reason = "stop"
    initialized_custom_stream_wrapper.custom_llm_provider = "anthropic"

    tool_chunk = {
        "text": "",
        "tool_use": {
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": "{}"},
        },
        "is_finished": False,
        "finish_reason": "",
        "usage": None,
        "index": 0,
    }

    result = initialized_custom_stream_wrapper.chunk_creator(chunk=tool_chunk)

    assert (
        result is not None
    ), "chunk_creator() returned None — tool_use data was dropped"

    tool_calls = result.choices[0].delta.tool_calls
    assert (
        tool_calls is not None and len(tool_calls) > 0
    ), "tool_calls should contain at least one tool call"
    assert tool_calls[0].id == "call_1"
    assert tool_calls[0].function.name == "get_weather"


def test_usage_only_chunk_not_dropped_when_finish_reason_already_set(
    initialized_custom_stream_wrapper: CustomStreamWrapper,
):
    """
    Regression test: usage-only chunks must not be dropped once finish_reason
    is already set. Dropping these chunks can lose terminal finish_reason in
    downstream Responses API streaming translation.
    """
    initialized_custom_stream_wrapper.received_finish_reason = "content_filter"
    initialized_custom_stream_wrapper.custom_llm_provider = "anthropic"

    usage_only_chunk = {
        "text": "",
        "tool_use": None,
        "is_finished": False,
        "finish_reason": "",
        "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        "index": 0,
    }

    result = initialized_custom_stream_wrapper.chunk_creator(chunk=usage_only_chunk)

    assert result is not None, "usage-only chunk should not be dropped"
    assert result.choices[0].finish_reason == "content_filter"
    assert result.usage is not None


def _run_dispatch(wrapper: CustomStreamWrapper, chunk):
    model_response = wrapper.model_response_creator()
    completion_obj = {"content": ""}
    result = wrapper._dispatch_provider_chunk(
        chunk=chunk,
        model_response=model_response,
        completion_obj=completion_obj,
    )
    return result, model_response, completion_obj


def test_dispatch_vllm_extracts_output_text(
    initialized_custom_stream_wrapper: CustomStreamWrapper,
):
    """vllm chunks expose text at chunk[0].outputs[0].text; the dispatch must
    surface that as the content and report a parsed result."""
    initialized_custom_stream_wrapper.custom_llm_provider = "vllm"

    class _Output:
        text = "hello from vllm"

    class _VLLMChunk:
        outputs = [_Output()]

    result, _, completion_obj = _run_dispatch(
        initialized_custom_stream_wrapper, [_VLLMChunk()]
    )

    assert isinstance(result, _ProviderChunkParsed)
    assert completion_obj["content"] == "hello from vllm"


def test_dispatch_petals_slices_completion_stream(
    initialized_custom_stream_wrapper: CustomStreamWrapper,
):
    """petals fakes streaming by slicing 30 chars off the buffered completion
    stream each call, leaving the remainder for the next chunk."""
    initialized_custom_stream_wrapper.custom_llm_provider = "petals"
    initialized_custom_stream_wrapper.completion_stream = "A" * 50

    result, _, completion_obj = _run_dispatch(
        initialized_custom_stream_wrapper, chunk=None
    )

    assert isinstance(result, _ProviderChunkParsed)
    assert completion_obj["content"] == "A" * 30
    assert initialized_custom_stream_wrapper.completion_stream == "A" * 20
    assert initialized_custom_stream_wrapper.received_finish_reason is None


def test_dispatch_petals_empty_stream_sets_stop(
    initialized_custom_stream_wrapper: CustomStreamWrapper,
):
    """An exhausted petals stream marks the turn finished with a stop reason."""
    initialized_custom_stream_wrapper.custom_llm_provider = "petals"
    initialized_custom_stream_wrapper.completion_stream = ""

    result, _, completion_obj = _run_dispatch(
        initialized_custom_stream_wrapper, chunk=None
    )

    assert isinstance(result, _ProviderChunkParsed)
    assert completion_obj["content"] == ""
    assert initialized_custom_stream_wrapper.received_finish_reason == "stop"


def test_dispatch_petals_empty_stream_after_finish_raises(
    initialized_custom_stream_wrapper: CustomStreamWrapper,
):
    """Once petals has already finished, an empty stream signals end-of-iteration."""
    initialized_custom_stream_wrapper.custom_llm_provider = "petals"
    initialized_custom_stream_wrapper.completion_stream = ""
    initialized_custom_stream_wrapper.received_finish_reason = "stop"

    with pytest.raises(StopIteration):
        _run_dispatch(initialized_custom_stream_wrapper, chunk=None)


def test_dispatch_palm_slices_completion_stream(
    initialized_custom_stream_wrapper: CustomStreamWrapper,
):
    """palm uses the same fake-streaming slice strategy as petals."""
    initialized_custom_stream_wrapper.custom_llm_provider = "palm"
    initialized_custom_stream_wrapper.completion_stream = "B" * 40

    result, _, completion_obj = _run_dispatch(
        initialized_custom_stream_wrapper, chunk=None
    )

    assert isinstance(result, _ProviderChunkParsed)
    assert completion_obj["content"] == "B" * 30
    assert initialized_custom_stream_wrapper.completion_stream == "B" * 10


def test_dispatch_cached_response_extracts_delta(
    initialized_custom_stream_wrapper: CustomStreamWrapper,
):
    """cached_response replays a stored ModelResponseStream; the dispatch lifts
    its delta content, finish_reason and id back onto the live response."""
    initialized_custom_stream_wrapper.custom_llm_provider = "cached_response"
    chunk = ModelResponseStream(
        id="chatcmpl-cache-1",
        choices=[
            StreamingChoices(
                index=0,
                delta=Delta(content="cached text"),
                finish_reason="stop",
            )
        ],
    )

    result, model_response, completion_obj = _run_dispatch(
        initialized_custom_stream_wrapper, chunk
    )

    assert isinstance(result, _ProviderChunkParsed)
    assert completion_obj["content"] == "cached text"
    assert initialized_custom_stream_wrapper.received_finish_reason == "stop"
    assert model_response.id == "chatcmpl-cache-1"
    assert initialized_custom_stream_wrapper.response_id == "chatcmpl-cache-1"


def test_dispatch_vertex_ai_legacy_text_and_finish_reason(
    initialized_custom_stream_wrapper: CustomStreamWrapper,
):
    """Legacy vertex_ai chunks (non-ModelResponseStream) expose .text and a
    candidate finish_reason enum that must be normalised to an OpenAI reason."""
    initialized_custom_stream_wrapper.custom_llm_provider = "vertex_ai"

    class _FinishReason:
        name = "STOP"

    class _Candidate:
        finish_reason = _FinishReason()

    class _VertexChunk:
        candidates = [_Candidate()]
        text = "vertex content"

    result, _, completion_obj = _run_dispatch(
        initialized_custom_stream_wrapper, _VertexChunk()
    )

    assert isinstance(result, _ProviderChunkParsed)
    assert completion_obj["content"] == "vertex content"
    assert initialized_custom_stream_wrapper.received_finish_reason == "stop"


def test_dispatch_vertex_ai_legacy_without_candidates_stringifies_chunk(
    initialized_custom_stream_wrapper: CustomStreamWrapper,
):
    """A legacy vertex_ai chunk with no candidates falls back to str(chunk)."""
    initialized_custom_stream_wrapper.custom_llm_provider = "vertex_ai"

    class _RawChunk:
        def __str__(self) -> str:
            return "raw vertex blob"

    result, _, completion_obj = _run_dispatch(
        initialized_custom_stream_wrapper, _RawChunk()
    )

    assert isinstance(result, _ProviderChunkParsed)
    assert completion_obj["content"] == "raw vertex blob"


def test_dispatch_vertex_ai_legacy_function_call(
    initialized_custom_stream_wrapper: CustomStreamWrapper,
):
    """A legacy vertex_ai chunk whose part has no text but carries a
    function_call is converted into an OpenAI tool-call delta."""
    initialized_custom_stream_wrapper.custom_llm_provider = "vertex_ai"

    class _FunctionCall:
        name = "get_weather"
        args = {"location": "SF"}

    class _Part:
        function_call = _FunctionCall()

    class _Content:
        parts = [_Part()]

    class _FinishReason:
        name = "STOP"

    class _Candidate:
        content = _Content()
        finish_reason = _FinishReason()

    class _VertexFunctionChunk:
        candidates = [_Candidate()]

        @property
        def text(self):
            raise RuntimeError("Part has no text.")

    result, _, _ = _run_dispatch(
        initialized_custom_stream_wrapper, _VertexFunctionChunk()
    )

    assert isinstance(result, _ProviderChunkParsed)
    tool_calls = result.response_obj["original_chunk"].choices[0].delta.tool_calls
    assert tool_calls[0].function.name == "get_weather"
    assert json.loads(tool_calls[0].function.arguments) == {"location": "SF"}
    assert initialized_custom_stream_wrapper.received_finish_reason == "stop"


def test_dispatch_custom_provider_returns_chunk_early(
    monkeypatch: pytest.MonkeyPatch,
    initialized_custom_stream_wrapper: CustomStreamWrapper,
):
    """A registered custom provider passes its already-OpenAI-shaped chunk
    straight through as an early return rather than re-parsing it."""
    monkeypatch.setattr(litellm, "_custom_providers", ["my-custom-llm"])
    initialized_custom_stream_wrapper.custom_llm_provider = "my-custom-llm"
    chunk = ModelResponseStream(
        choices=[
            StreamingChoices(index=0, delta=Delta(content="hi"), finish_reason=None)
        ]
    )

    result, _, _ = _run_dispatch(initialized_custom_stream_wrapper, chunk)

    assert isinstance(result, _ProviderChunkEarlyReturn)
    assert result.value is chunk


def test_dispatch_custom_provider_finish_only_returns_none_early(
    monkeypatch: pytest.MonkeyPatch,
    initialized_custom_stream_wrapper: CustomStreamWrapper,
):
    """A custom-provider chunk that carries only a finish_reason (no content)
    records the reason and returns None so no empty delta is emitted."""
    monkeypatch.setattr(litellm, "_custom_providers", ["my-custom-llm"])
    initialized_custom_stream_wrapper.custom_llm_provider = "my-custom-llm"
    chunk = ModelResponseStream(
        choices=[
            StreamingChoices(index=0, delta=Delta(content=None), finish_reason="stop")
        ]
    )

    result, _, _ = _run_dispatch(initialized_custom_stream_wrapper, chunk)

    assert isinstance(result, _ProviderChunkEarlyReturn)
    assert result.value is None
    assert initialized_custom_stream_wrapper.received_finish_reason == "stop"


def test_dispatch_text_completion_codestral_parses_chunk(
    initialized_custom_stream_wrapper: CustomStreamWrapper,
):
    """text-completion-codestral streams raw SSE JSON strings that the dispatch
    routes through CodestralTextCompletionConfig to extract content/finish."""
    initialized_custom_stream_wrapper.custom_llm_provider = "text-completion-codestral"
    chunk = json.dumps(
        {"choices": [{"delta": {"content": "codestral text"}, "finish_reason": "stop"}]}
    )

    result, _, completion_obj = _run_dispatch(initialized_custom_stream_wrapper, chunk)

    assert isinstance(result, _ProviderChunkParsed)
    assert completion_obj["content"] == "codestral text"
    assert initialized_custom_stream_wrapper.received_finish_reason == "stop"


def test_dispatch_text_completion_codestral_requires_string(
    initialized_custom_stream_wrapper: CustomStreamWrapper,
):
    """The codestral branch only knows how to parse raw strings; anything else
    is a programming error and must surface loudly."""
    initialized_custom_stream_wrapper.custom_llm_provider = "text-completion-codestral"

    with pytest.raises(ValueError):
        _run_dispatch(initialized_custom_stream_wrapper, {"not": "a string"})


def test_dispatch_triton_stream(
    initialized_custom_stream_wrapper: CustomStreamWrapper,
):
    """triton stream chunks arrive as dicts keyed by text_output/stop_reason."""
    initialized_custom_stream_wrapper.custom_llm_provider = "triton"
    chunk = {"text_output": "triton text", "is_finished": True, "stop_reason": "stop"}

    result, _, completion_obj = _run_dispatch(initialized_custom_stream_wrapper, chunk)

    assert isinstance(result, _ProviderChunkParsed)
    assert completion_obj["content"] == "triton text"
    assert initialized_custom_stream_wrapper.received_finish_reason == "stop"


def test_dispatch_ai21_decodes_completion(
    initialized_custom_stream_wrapper: CustomStreamWrapper,
):
    """ai21 does fake streaming over a single byte-encoded JSON completion."""
    initialized_custom_stream_wrapper.custom_llm_provider = "ai21"
    chunk = json.dumps({"completions": [{"data": {"text": "ai21 text"}}]}).encode(
        "utf-8"
    )

    result, _, completion_obj = _run_dispatch(initialized_custom_stream_wrapper, chunk)

    assert isinstance(result, _ProviderChunkParsed)
    assert completion_obj["content"] == "ai21 text"
    assert initialized_custom_stream_wrapper.received_finish_reason == "stop"


def test_dispatch_text_completion_openai_with_usage(
    initialized_custom_stream_wrapper: CustomStreamWrapper,
):
    """text-completion-openai chunks expose choices[].text and an optional usage
    block that the dispatch lifts onto the model response."""
    initialized_custom_stream_wrapper.custom_llm_provider = "text-completion-openai"

    class _Choice:
        text = "oai text"
        finish_reason = "stop"

    class _Usage:
        prompt_tokens = 5
        completion_tokens = 3
        total_tokens = 8

    class _TextChunk:
        choices = [_Choice()]
        usage = _Usage()

    result, model_response, completion_obj = _run_dispatch(
        initialized_custom_stream_wrapper, _TextChunk()
    )

    assert isinstance(result, _ProviderChunkParsed)
    assert completion_obj["content"] == "oai text"
    assert initialized_custom_stream_wrapper.received_finish_reason == "stop"
    assert model_response.usage.prompt_tokens == 5
    assert model_response.usage.total_tokens == 8


@pytest.mark.asyncio
async def test_custom_stream_wrapper_anext_does_not_block_event_loop_for_sync_iterators(
    logging_obj: Logging,
):
    """
    Regression test: __anext__ must not call blocking next() on a sync iterator on the
    event loop thread. This happens for some provider streams which are sync iterators
    but used in async contexts (e.g. boto3-style streaming).
    """

    class BlockingIterator:
        def __init__(self, chunks, delay_s: float):
            self._it = iter(chunks)
            self._delay_s = delay_s

        def __iter__(self):
            return self

        def __next__(self):
            time.sleep(self._delay_s)  # simulate blocking I/O
            return next(self._it)

    test_chunk = ModelResponseStream(
        id="chatcmpl-test",
        created=int(time.time()),
        model="test-model",
        object="chat.completion.chunk",
        system_fingerprint=None,
        choices=[
            StreamingChoices(
                finish_reason="stop",
                index=0,
                delta=Delta(
                    provider_specific_fields=None,
                    content="hello",
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
    )

    # Delay is intentionally > the wait_for timeout used to detect event loop blocking.
    wrapper = CustomStreamWrapper(
        completion_stream=BlockingIterator([test_chunk], delay_s=0.3),
        model="test-model",
        logging_obj=logging_obj,
        custom_llm_provider="cached_response",
    )

    tick_event = asyncio.Event()

    async def background_tick():
        await asyncio.sleep(0.05)
        tick_event.set()

    # Run the two coroutines concurrently and measure wall time.
    # If __anext__ blocks the event loop, background_tick can't run and the gather
    # takes the full 0.3 s delay; if non-blocking both finish within ~0.35 s total.
    start = asyncio.get_event_loop().time()

    out, _ = await asyncio.gather(
        wrapper.__anext__(),
        background_tick(),
    )

    elapsed = asyncio.get_event_loop().time() - start
    assert isinstance(out, ModelResponseStream)
    # background_tick sleeps 0.05 s; total must finish well under 2 × 0.3 s
    assert elapsed < 0.5, f"Event loop was likely blocked (elapsed={elapsed:.2f}s)"


@pytest.mark.asyncio
async def test_custom_stream_wrapper_anext_exhaustion_raises_stop_async_iteration(
    logging_obj: Logging,
):
    """
    PEP 479 regression: when a sync iterator is exhausted, asyncio.to_thread(next, it)
    raises StopIteration inside a coroutine, which Python converts to RuntimeError.
    The wrapper must catch StopIteration in the thread and raise StopAsyncIteration
    in the coroutine instead, so callers get clean stream termination.
    """

    class SingleChunkIterator:
        def __init__(self, chunk: ModelResponseStream):
            self._chunk = chunk
            self._done = False

        def __iter__(self):
            return self

        def __next__(self):
            if self._done:
                raise StopIteration
            self._done = True
            return self._chunk

    test_chunk = ModelResponseStream(
        id="chatcmpl-exhaustion-test",
        created=int(time.time()),
        model="test-model",
        object="chat.completion.chunk",
        system_fingerprint=None,
        choices=[
            StreamingChoices(
                finish_reason="stop",
                index=0,
                delta=Delta(
                    provider_specific_fields=None,
                    content="done",
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
    )

    wrapper = CustomStreamWrapper(
        completion_stream=SingleChunkIterator(test_chunk),
        model="test-model",
        logging_obj=logging_obj,
        custom_llm_provider="cached_response",
    )

    # Drain the wrapper fully.  The wrapper's except-handler calls finish_reason_handler()
    # on the first StopAsyncIteration (sent_last_chunk=False→True), then re-raises on the
    # next call.  What must NOT happen is a RuntimeError from PEP 479 converting
    # StopIteration (raised inside the thread) to RuntimeError inside the coroutine.
    try:
        while True:
            await wrapper.__anext__()
    except StopAsyncIteration:
        pass  # expected clean termination
    except RuntimeError as e:
        pytest.fail(f"PEP 479 regression: StopIteration leaked as RuntimeError: {e}")


# Azure streaming chunks that reproduce issue #24221:
# Azure sends an initial chunk with prompt_filter_results and choices=[],
# then a chunk with role='assistant' and content='', then content chunks.
# With stream_options.include_usage=True, the empty-choices chunk was
# forwarded with an inflated default choice, consuming the sent_first_chunk
# flag and causing strip_role_from_delta to strip the role from the real
# first chunk.
_AZURE_CHUNKS_WITH_PROMPT_FILTER = [
    # Chunk 1: prompt_filter_results, no choices (Azure-specific)
    ModelResponseStream(
        id="chatcmpl-abc123",
        created=1742056047,
        model=None,
        object="chat.completion.chunk",
        choices=[],
        usage=None,
    ),
    # Chunk 2: first real chunk with role='assistant' and empty content
    ModelResponseStream(
        id="chatcmpl-abc123",
        created=1742056047,
        model=None,
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(content="", role="assistant"),
            )
        ],
        usage=None,
    ),
    # Chunk 3: content
    ModelResponseStream(
        id="chatcmpl-abc123",
        created=1742056047,
        model=None,
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(content="Hello!"),
            )
        ],
        usage=None,
    ),
    # Chunk 4: finish_reason
    ModelResponseStream(
        id="chatcmpl-abc123",
        created=1742056047,
        model=None,
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason="stop",
                index=0,
                delta=Delta(),
            )
        ],
        usage=None,
    ),
    # Chunk 5: final usage chunk, no choices
    ModelResponseStream(
        id="chatcmpl-abc123",
        created=1742056047,
        model=None,
        object="chat.completion.chunk",
        choices=[],
        usage=Usage(
            completion_tokens=10,
            prompt_tokens=20,
            total_tokens=30,
        ),
    ),
]


@pytest.mark.parametrize("sync_mode", [True, False], ids=["sync", "async"])
@pytest.mark.asyncio
async def test_azure_streaming_role_preserved_with_include_usage(sync_mode: bool):
    """
    Regression test for https://github.com/BerriAI/litellm/issues/24221

    Azure sends an initial chunk with choices=[] (prompt_filter_results)
    before the first content chunk. With stream_options.include_usage=True,
    this chunk was forwarded with an inflated default choice, which:
    1. Consumed the sent_first_chunk flag
    2. Caused strip_role_from_delta to strip role from the real first chunk

    The fix ensures:
    - Chunks with choices=[] are forwarded faithfully (no inflated choices)
    - sent_first_chunk is only marked for chunks with real choices
    - Chunks with role in delta are not discarded as empty
    """
    completion_stream = ModelResponseListIterator(
        model_responses=_AZURE_CHUNKS_WITH_PROMPT_FILTER
    )

    response = CustomStreamWrapper(
        completion_stream=completion_stream,
        model="azure/gpt-5-nano",
        custom_llm_provider="azure",
        logging_obj=Logging(
            model="azure/gpt-5-nano",
            messages=[{"role": "user", "content": "Hey"}],
            stream=True,
            call_type="completion",
            start_time=time.time(),
            litellm_call_id="12345",
            function_id="1245",
        ),
        stream_options={"include_usage": True},
    )

    chunks = []
    if sync_mode:
        for chunk in response:
            chunks.append(chunk)
    else:
        async for chunk in response:
            chunks.append(chunk)

    # The prompt_filter chunk should be forwarded with choices=[]
    assert (
        len(chunks[0].choices) == 0
    ), f"Expected prompt_filter chunk with choices=[], got {len(chunks[0].choices)} choices"

    # At least one chunk must have role='assistant' in its delta
    has_role = any(
        len(c.choices) > 0 and getattr(c.choices[0].delta, "role", None) == "assistant"
        for c in chunks
    )
    assert has_role, (
        "No chunk contained role='assistant' in delta (issue #24221). "
        "Chunk deltas: "
        + str([c.choices[0].delta if c.choices else "no choices" for c in chunks])
    )


def test_gemini_legacy_vertex_stop_finish_reason_normalised():
    """
    The legacy vertex_ai SDK streaming path sets finish_reason from a proto enum
    whose .name attribute is an uppercase string (e.g. "STOP", "MAX_TOKENS").
    Before the fix, received_finish_reason was stored as "STOP" which never
    matched "stop" in finish_reason_handler, silently breaking the tool_calls
    override.  After the fix, map_finish_reason() is applied so the value is
    always an OpenAI-normalised lowercase string.
    """
    wrapper = CustomStreamWrapper(
        completion_stream=None,
        model="gemini-1.5-pro",
        logging_obj=MagicMock(),
        custom_llm_provider="vertex_ai",
    )

    # Simulate a proto-like chunk: .candidates[0].finish_reason.name == "STOP"
    mock_finish_reason = MagicMock()
    mock_finish_reason.name = "STOP"
    mock_candidate = MagicMock()
    mock_candidate.finish_reason = mock_finish_reason
    mock_chunk = MagicMock()
    mock_chunk.candidates = [mock_candidate]
    # Ensure the chunk is not treated as a ModelResponseStream
    mock_chunk.__class__ = type("FakeProtoChunk", (), {})

    with patch("litellm.litellm_core_utils.streaming_handler.proto", create=True):
        wrapper.chunk_creator(chunk=mock_chunk)

    assert wrapper.received_finish_reason == "stop", (
        f"Expected 'stop' but got {wrapper.received_finish_reason!r}. "
        "map_finish_reason() was not applied to the Gemini enum name."
    )


def test_gemini_legacy_vertex_tool_calls_finish_reason_with_stop_enum():
    """
    When Gemini emits finish_reason STOP alongside tool-call content, the final
    chunk must report finish_reason='tool_calls'.  This requires that the raw
    "STOP" enum name is first normalised to lowercase "stop" by map_finish_reason()
    so that finish_reason_handler's equality check fires correctly.
    """
    wrapper = CustomStreamWrapper(
        completion_stream=None,
        model="gemini-1.5-pro",
        logging_obj=MagicMock(),
        custom_llm_provider="vertex_ai",
    )

    mock_finish_reason = MagicMock()
    mock_finish_reason.name = "STOP"
    mock_candidate = MagicMock()
    mock_candidate.finish_reason = mock_finish_reason
    mock_chunk = MagicMock()
    mock_chunk.candidates = [mock_candidate]
    mock_chunk.__class__ = type("FakeProtoChunk", (), {})

    with patch("litellm.litellm_core_utils.streaming_handler.proto", create=True):
        wrapper.chunk_creator(chunk=mock_chunk)

    # Signal that tool_calls were present in the stream
    wrapper.tool_call = True

    final = wrapper.finish_reason_handler()
    assert final.choices[0].finish_reason == "tool_calls", (
        f"Expected 'tool_calls' but got {final.choices[0].finish_reason!r}. "
        "STOP enum was not normalised through map_finish_reason()."
    )


@pytest.mark.parametrize(
    "finish_reason", ["stop", "tool_calls", "length", "content_filter"]
)
def test_chunk_creator_passes_through_model_response_stream(
    initialized_custom_stream_wrapper: CustomStreamWrapper,
    finish_reason: str,
):
    """
    chunk_creator must pass ModelResponseStream chunks from custom providers
    straight through and preserve finish_reason exactly — not force-cast to GChunk.
    Regression test for issue #27389.
    """
    initialized_custom_stream_wrapper.custom_llm_provider = "my-custom-provider"
    litellm._custom_providers.append("my-custom-provider")

    chunk = ModelResponseStream(
        id="test-id",
        choices=[
            StreamingChoices(
                index=0,
                delta=Delta(content="Hello", role="assistant"),
                finish_reason=finish_reason,
            )
        ],
    )

    result = initialized_custom_stream_wrapper.chunk_creator(chunk=chunk)

    litellm._custom_providers.remove("my-custom-provider")

    assert result is not None
    assert initialized_custom_stream_wrapper.received_finish_reason == finish_reason


def test_chunk_creator_drops_empty_finish_chunk(
    initialized_custom_stream_wrapper: CustomStreamWrapper,
):
    """
    A ModelResponseStream chunk with finish_reason but no content should return
    None so finish_reason_handler() synthesises the final chunk — mirrors GChunk
    behaviour via is_chunk_non_empty.
    """
    initialized_custom_stream_wrapper.custom_llm_provider = "my-custom-provider"
    litellm._custom_providers.append("my-custom-provider")

    chunk = ModelResponseStream(
        id="test-id",
        choices=[
            StreamingChoices(
                index=0,
                delta=Delta(content=""),
                finish_reason="stop",
            )
        ],
    )

    result = initialized_custom_stream_wrapper.chunk_creator(chunk=chunk)

    litellm._custom_providers.remove("my-custom-provider")

    assert result is None
    assert initialized_custom_stream_wrapper.received_finish_reason == "stop"


def test_chunk_creator_stops_iteration_on_trailing_chunk(
    initialized_custom_stream_wrapper: CustomStreamWrapper,
):
    """
    After received_finish_reason is set, any empty trailing chunk (e.g. provider
    metadata flush) must raise StopIteration to end the stream cleanly.
    """
    initialized_custom_stream_wrapper.custom_llm_provider = "my-custom-provider"
    initialized_custom_stream_wrapper.received_finish_reason = "stop"
    litellm._custom_providers.append("my-custom-provider")

    trailing_chunk = ModelResponseStream(
        id="test-id",
        choices=[
            StreamingChoices(
                index=0,
                delta=Delta(content=None),
                finish_reason="stop",
            )
        ],
    )

    with pytest.raises(StopIteration):
        initialized_custom_stream_wrapper.chunk_creator(chunk=trailing_chunk)

    litellm._custom_providers.remove("my-custom-provider")


def test_chunk_creator_strips_finish_reason_from_content_chunk(
    initialized_custom_stream_wrapper: CustomStreamWrapper,
):
    """
    When content and finish_reason arrive in the same chunk, finish_reason must be
    stripped so finish_reason_handler() emits it on the synthetic terminal chunk —
    preventing two terminal chunks (double finish_reason bug).
    """
    initialized_custom_stream_wrapper.custom_llm_provider = "my-custom-provider"
    litellm._custom_providers.append("my-custom-provider")

    chunk = ModelResponseStream(
        id="test-id",
        choices=[
            StreamingChoices(
                index=0,
                delta=Delta(content="Hello"),
                finish_reason="stop",
            )
        ],
    )

    result = initialized_custom_stream_wrapper.chunk_creator(chunk=chunk)

    litellm._custom_providers.remove("my-custom-provider")

    assert result is not None
    assert (
        result.choices[0].finish_reason is None
    ), "finish_reason must be stripped from content chunks to avoid double terminal chunks"
    assert initialized_custom_stream_wrapper.received_finish_reason == "stop"


def test_chunk_creator_tool_calls_not_dropped_on_finish(
    initialized_custom_stream_wrapper: CustomStreamWrapper,
):
    """
    A terminal chunk with finish_reason="tool_calls" and delta.tool_calls must NOT
    be silently dropped — tool_calls counts as content so the chunk is passed through
    (with finish_reason stripped) rather than returning None.
    """
    from litellm.types.utils import ChatCompletionDeltaToolCall, Function

    initialized_custom_stream_wrapper.custom_llm_provider = "my-custom-provider"
    litellm._custom_providers.append("my-custom-provider")

    chunk = ModelResponseStream(
        id="test-id",
        choices=[
            StreamingChoices(
                index=0,
                delta=Delta(
                    content=None,
                    tool_calls=[
                        ChatCompletionDeltaToolCall(
                            id="call_abc",
                            function=Function(
                                name="get_weather", arguments='{"city":"NYC"}'
                            ),
                            type="function",
                            index=0,
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
    )

    result = initialized_custom_stream_wrapper.chunk_creator(chunk=chunk)

    litellm._custom_providers.remove("my-custom-provider")

    assert result is not None, "tool_calls chunk must not be dropped"
    assert result.choices[0].delta.tool_calls is not None
    assert result.choices[0].finish_reason is None
    assert initialized_custom_stream_wrapper.received_finish_reason == "tool_calls"


def test_record_partial_usage_for_failure_stashes_usage_and_cost():
    """A stream that breaks mid-flight must surface the usage assembled from the
    chunks already delivered, plus its cost, on the logging object so the
    failure handler records the real partial spend instead of zero.
    """
    logging_obj = Logging(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hey"}],
        stream=True,
        call_type="completion",
        start_time=time.time(),
        litellm_call_id="partial-usage-1",
        function_id="1245",
    )
    logging_obj.model_call_details["custom_llm_provider"] = "openai"

    wrapper = CustomStreamWrapper(
        completion_stream=None,
        model="gpt-4o-mini",
        logging_obj=logging_obj,
        custom_llm_provider="openai",
    )
    wrapper.chunks = [
        ModelResponseStream(
            id="chatcmpl-partial-1",
            created=1742056047,
            model="gpt-4o-mini",
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(
                        content="The Roman Empire began when", role="assistant"
                    ),
                )
            ],
            usage=Usage(prompt_tokens=30, completion_tokens=1, total_tokens=31),
        )
    ]

    wrapper._record_partial_usage_for_failure()

    stashed = logging_obj.model_call_details["combined_usage_object"]
    assert stashed.prompt_tokens == 30
    assert stashed.completion_tokens == 1
    assert stashed.total_tokens == 31
    assert isinstance(logging_obj.model_call_details["response_cost"], float)


def test_record_partial_usage_for_failure_noop_without_chunks():
    """With no chunks delivered there is nothing billed to recover, so the
    failure stash must stay absent and not force a zero-usage row.
    """
    logging_obj = Logging(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hey"}],
        stream=True,
        call_type="completion",
        start_time=time.time(),
        litellm_call_id="partial-usage-2",
        function_id="1245",
    )
    wrapper = CustomStreamWrapper(
        completion_stream=None,
        model="gpt-4o-mini",
        logging_obj=logging_obj,
        custom_llm_provider="openai",
    )
    wrapper.chunks = []

    wrapper._record_partial_usage_for_failure()

    assert "combined_usage_object" not in logging_obj.model_call_details


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_stream_chunk_builder_raise_at_end_of_stream_still_recovers_usage(
    sync_mode,
):
    """stream_chunk_builder re-raises (as APIError) on large agentic tool-use
    streams. That raise originates inside the except-StopIteration handler, so
    before the fix it escaped __next__/__anext__ and the request was dropped from
    SpendLogs while the provider billed the tokens. The wrapper must catch it and
    recover usage from the raw chunks so cost is still tracked."""
    final_usage_block = Usage(
        completion_tokens=392, prompt_tokens=1799, total_tokens=2191
    )
    final_chunk = ModelResponseStream(
        id="chatcmpl-raise-test",
        created=1742056047,
        model=None,
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason="stop",
                index=0,
                delta=Delta(content="", role="assistant"),
            )
        ],
        usage=final_usage_block,
    )
    test_chunks = bedrock_chunks + [final_chunk]

    logging_obj = Logging(
        model="bedrock/claude-haiku-4-5-20251001-v1:0",
        messages=[{"role": "user", "content": "Hey"}],
        stream=True,
        call_type="completion",
        start_time=time.time(),
        litellm_call_id="raise-test",
        function_id="1245",
    )

    response = CustomStreamWrapper(
        completion_stream=ModelResponseListIterator(model_responses=test_chunks),
        model="bedrock/claude-haiku-4-5-20251001-v1:0",
        custom_llm_provider="bedrock",
        logging_obj=logging_obj,
        stream_options={"include_usage": True},
    )

    seen_usage = []
    with patch.object(
        litellm,
        "stream_chunk_builder",
        side_effect=Exception("simulated assembly failure"),
    ):
        # before the fix this raised and dropped the request; it must not raise now
        if sync_mode:
            for chunk in response:
                if getattr(chunk, "usage", None) is not None:
                    seen_usage.append(chunk.usage)
        else:
            async for chunk in response:
                if getattr(chunk, "usage", None) is not None:
                    seen_usage.append(chunk.usage)

    assert any(
        u.total_tokens == final_usage_block.total_tokens for u in seen_usage
    ), "usage recovered from raw chunks was not emitted after stream_chunk_builder raised"


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_stream_chunk_builder_raise_and_usage_recovery_failure_does_not_crash(
    sync_mode,
):
    """If end-of-stream assembly raises AND best-effort usage recovery from the raw
    chunks also fails, the stream must still complete cleanly rather than propagate
    the exception to the consumer."""
    from litellm.litellm_core_utils import streaming_handler as sh_module

    final_chunk = ModelResponseStream(
        id="chatcmpl-raise-recover-fail",
        created=1742056047,
        model=None,
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason="stop",
                index=0,
                delta=Delta(content="", role="assistant"),
            )
        ],
        usage=Usage(completion_tokens=1, prompt_tokens=1, total_tokens=2),
    )

    response = CustomStreamWrapper(
        completion_stream=ModelResponseListIterator(
            model_responses=bedrock_chunks + [final_chunk]
        ),
        model="bedrock/claude-haiku-4-5-20251001-v1:0",
        custom_llm_provider="bedrock",
        logging_obj=Logging(
            model="bedrock/claude-haiku-4-5-20251001-v1:0",
            messages=[{"role": "user", "content": "Hey"}],
            stream=True,
            call_type="completion",
            start_time=time.time(),
            litellm_call_id="raise-recover-fail",
            function_id="1245",
        ),
        stream_options={"include_usage": True},
    )

    with (
        patch.object(
            litellm, "stream_chunk_builder", side_effect=Exception("assembly failed")
        ),
        patch.object(
            sh_module, "calculate_total_usage", side_effect=Exception("recovery failed")
        ),
    ):
        # must not raise even though both assembly and recovery fail
        if sync_mode:
            chunks = [c for c in response]
        else:
            chunks = [c async for c in response]

    assert len(chunks) > 0


def test_normalize_logprobs_converts_raw_sdk_object():
    """
    Regression test for https://github.com/BerriAI/litellm/issues/33456

    openai SDK models are built with defer_build=True, so an instance parsed via
    model_validate() (how the SDK parses every streaming chunk) keeps a MockValSer
    placeholder serializer. If such a raw object is stored on a litellm chunk, a
    later model_dump() blows up with
    "'MockValSer' object cannot be converted to 'SchemaSerializer'".

    _normalize_logprobs must convert it to litellm's own ChoiceLogprobs.
    """
    from openai.types.chat.chat_completion_chunk import (
        ChoiceLogprobs as OpenAIChoiceLogprobs,
    )

    raw = OpenAIChoiceLogprobs.model_validate(
        {
            "content": [
                {"token": "hi", "bytes": [104, 105], "logprob": -0.1, "top_logprobs": []}
            ],
            "refusal": None,
        }
    )

    normalized = CustomStreamWrapper._normalize_logprobs(raw)

    assert isinstance(normalized, ChoiceLogprobs)
    assert type(normalized).__module__.startswith("litellm")
    assert normalized.content is not None and normalized.content[0].token == "hi"

    assert isinstance(
        CustomStreamWrapper._normalize_logprobs({"content": None, "refusal": None}),
        ChoiceLogprobs,
    )
    assert CustomStreamWrapper._normalize_logprobs(normalized) is normalized
    assert CustomStreamWrapper._normalize_logprobs(None) is None


@pytest.mark.asyncio
async def test_streaming_logprobs_do_not_store_raw_sdk_object(logging_obj: Logging):
    """
    Regression test for https://github.com/BerriAI/litellm/issues/33456

    A provider (e.g. SAP AI Core) that emits a final chunk carrying logprobs +
    usage with an empty delta used to leave the raw openai SDK ChoiceLogprobs on
    the litellm chunk. The subsequent model_dump() in __anext__ then crashed with
    a MockValSer TypeError, which the router surfaced as MidStreamFallbackError and
    truncated the stream mid-way.

    The forwarded chunk's logprobs must be litellm's own ChoiceLogprobs and every
    chunk must stay serializable via model_dump()/model_dump_json().
    """
    from litellm.types.llms.openai import OpenAIChatCompletionChunk

    def _chunk(delta, finish=None, usage=None, logprobs=None):
        payload = {
            "id": "chatcmpl-33456",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish,
                    "logprobs": logprobs,
                }
            ],
        }
        if usage is not None:
            payload["usage"] = usage
        return OpenAIChatCompletionChunk.model_validate(payload)

    logprobs_payload = {
        "content": [
            {"token": "hi", "bytes": [104, 105], "logprob": -0.1, "top_logprobs": []}
        ],
        "refusal": None,
    }

    async def _stream():
        yield _chunk({"role": "assistant", "content": "hi"})
        yield _chunk(
            {},
            finish="stop",
            logprobs=logprobs_payload,
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )

    response = CustomStreamWrapper(
        completion_stream=_stream(),
        model="gpt-4o",
        custom_llm_provider="openai",
        logging_obj=logging_obj,
    )

    logprobs_chunks = []
    async for chunk in response:
        chunk.model_dump()
        chunk.model_dump_json()
        if chunk.choices and chunk.choices[0].logprobs is not None:
            logprobs_chunks.append(chunk.choices[0].logprobs)

    assert logprobs_chunks, "expected a forwarded chunk carrying logprobs"
    for lp in logprobs_chunks:
        assert isinstance(lp, ChoiceLogprobs)
        assert "openai" not in type(lp).__module__


class TransportErrorAfterChunksIterator:
    """Yields the given chunks, then raises the given exception once, then StopAsyncIteration."""

    def __init__(self, model_responses, exception):
        self.model_responses = model_responses
        self.exception = exception
        self.index = 0
        self.raised = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index < len(self.model_responses):
            chunk = self.model_responses[self.index]
            self.index += 1
            return chunk
        if not self.raised:
            self.raised = True
            raise self.exception
        raise StopAsyncIteration


def _reset_test_chunk(content: Optional[str] = None, finish_reason: Optional[str] = None) -> ModelResponseStream:
    return ModelResponseStream(
        id="chatcmpl-reset-test",
        created=1783458104,
        model="stub-model",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                index=0,
                delta=Delta(content=content),
                finish_reason=finish_reason,
            )
        ],
    )


@pytest.mark.asyncio
async def test_transport_read_error_after_finish_reason_ends_stream_gracefully(
    logging_obj: Logging,
):
    """A trailing connection reset after the provider's finish chunk must not fail the stream."""
    import httpx

    completion_stream = TransportErrorAfterChunksIterator(
        model_responses=[
            _reset_test_chunk(content="Hello"),
            _reset_test_chunk(finish_reason="stop"),
        ],
        exception=httpx.ReadError("Response payload is not completed"),
    )
    response = CustomStreamWrapper(
        completion_stream=completion_stream,
        model="hosted_vllm/stub-model",
        custom_llm_provider="hosted_vllm",
        logging_obj=logging_obj,
    )

    chunks = [chunk async for chunk in response]

    finish_reasons = [
        chunk.choices[0].finish_reason
        for chunk in chunks
        if chunk.choices and chunk.choices[0].finish_reason
    ]
    contents = [
        chunk.choices[0].delta.content
        for chunk in chunks
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content
    ]
    assert finish_reasons == ["stop"]
    assert contents == ["Hello"]


@pytest.mark.asyncio
async def test_transport_read_error_before_finish_reason_raises(logging_obj: Logging):
    """A connection reset before any finish chunk must surface, never end as a clean stop.

    Regression test for silent empty/truncated HTTP 200 streams: the aiohttp
    transport used to swallow mid-stream connection resets, so the wrapper saw a
    clean end-of-stream and fabricated finish_reason "stop".
    """
    import httpx

    from litellm.exceptions import MidStreamFallbackError

    completion_stream = TransportErrorAfterChunksIterator(
        model_responses=[_reset_test_chunk(content="Hel")],
        exception=httpx.ReadError("Response payload is not completed"),
    )
    response = CustomStreamWrapper(
        completion_stream=completion_stream,
        model="hosted_vllm/stub-model",
        custom_llm_provider="hosted_vllm",
        logging_obj=logging_obj,
    )

    received = []
    with pytest.raises(MidStreamFallbackError):
        async for chunk in response:
            received.append(chunk)

    fabricated_finish_reasons = [
        chunk.choices[0].finish_reason
        for chunk in received
        if chunk.choices and chunk.choices[0].finish_reason
    ]
    assert fabricated_finish_reasons == []
