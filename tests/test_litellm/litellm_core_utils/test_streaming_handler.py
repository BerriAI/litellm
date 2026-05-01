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
    assert len(chunks[0].choices) == 0, (
        f"Expected prompt_filter chunk with choices=[], got {len(chunks[0].choices)} choices"
    )

    # At least one chunk must have role='assistant' in its delta
    has_role = any(
        len(c.choices) > 0
        and getattr(c.choices[0].delta, "role", None) == "assistant"
        for c in chunks
    )
    assert has_role, (
        "No chunk contained role='assistant' in delta (issue #24221). "
        "Chunk deltas: "
        + str([
            c.choices[0].delta if c.choices else "no choices"
            for c in chunks
        ])
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
