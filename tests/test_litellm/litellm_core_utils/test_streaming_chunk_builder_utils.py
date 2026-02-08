import json
import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.litellm_core_utils.streaming_chunk_builder_utils import ChunkProcessor
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    ChatCompletionMessageToolCall,
    Delta,
    Function,
    ModelResponseStream,
    PromptTokensDetails,
    ServerToolUse,
    StreamingChoices,
    Usage,
)


def test_get_combined_tool_content():
    chunks = [
        ModelResponseStream(
            id="chatcmpl-8478099a-3724-42c7-9194-88d97ffd254b",
            created=1744771912,
            model="llama-3.3-70b-versatile",
            object="chat.completion.chunk",
            system_fingerprint=None,
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(
                        provider_specific_fields=None,
                        content=None,
                        role="assistant",
                        function_call=None,
                        tool_calls=[
                            ChatCompletionDeltaToolCall(
                                id="call_m87w",
                                function=Function(
                                    arguments='{"location": "San Francisco", "unit": "imperial"}',
                                    name="get_current_weather",
                                ),
                                type="function",
                                index=0,
                            )
                        ],
                        audio=None,
                    ),
                    logprobs=None,
                )
            ],
            provider_specific_fields=None,
            stream_options=None,
        ),
        ModelResponseStream(
            id="chatcmpl-8478099a-3724-42c7-9194-88d97ffd254b",
            created=1744771912,
            model="llama-3.3-70b-versatile",
            object="chat.completion.chunk",
            system_fingerprint=None,
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(
                        provider_specific_fields=None,
                        content=None,
                        role="assistant",
                        function_call=None,
                        tool_calls=[
                            ChatCompletionDeltaToolCall(
                                id="call_rrns",
                                function=Function(
                                    arguments='{"location": "Tokyo", "unit": "metric"}',
                                    name="get_current_weather",
                                ),
                                type="function",
                                index=1,
                            )
                        ],
                        audio=None,
                    ),
                    logprobs=None,
                )
            ],
            provider_specific_fields=None,
            stream_options=None,
        ),
        ModelResponseStream(
            id="chatcmpl-8478099a-3724-42c7-9194-88d97ffd254b",
            created=1744771912,
            model="llama-3.3-70b-versatile",
            object="chat.completion.chunk",
            system_fingerprint=None,
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(
                        provider_specific_fields=None,
                        content=None,
                        role="assistant",
                        function_call=None,
                        tool_calls=[
                            ChatCompletionDeltaToolCall(
                                id="call_0k29",
                                function=Function(
                                    arguments='{"location": "Paris", "unit": "metric"}',
                                    name="get_current_weather",
                                ),
                                type="function",
                                index=2,
                            )
                        ],
                        audio=None,
                    ),
                    logprobs=None,
                )
            ],
            provider_specific_fields=None,
            stream_options=None,
        ),
    ]
    chunk_processor = ChunkProcessor(chunks=chunks)

    tool_calls_list = chunk_processor.get_combined_tool_content(chunks)
    assert tool_calls_list == [
        ChatCompletionMessageToolCall(
            id="call_m87w",
            function=Function(
                arguments='{"location": "San Francisco", "unit": "imperial"}',
                name="get_current_weather",
            ),
            type="function",
        ),
        ChatCompletionMessageToolCall(
            id="call_rrns",
            function=Function(
                arguments='{"location": "Tokyo", "unit": "metric"}',
                name="get_current_weather",
            ),
            type="function",
        ),
        ChatCompletionMessageToolCall(
            id="call_0k29",
            function=Function(
                arguments='{"location": "Paris", "unit": "metric"}',
                name="get_current_weather",
            ),
            type="function",
        ),
    ]


def test_get_combined_thinking_content_preserves_interleaved_blocks():
    base_chunk = {
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": "claude-sonnet-4-20250514",
    }

    def make_chunk(**delta_kwargs):
        return ModelResponseStream(
            **base_chunk,
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(**delta_kwargs),
                    finish_reason=None,
                )
            ],
        )

    chunks = [
        make_chunk(role="assistant", content=None),
        make_chunk(
            thinking_blocks=[
                {"type": "thinking", "thinking": "Step 1 analysis...", "signature": None}
            ]
        ),
        make_chunk(
            thinking_blocks=[
                {"type": "thinking", "thinking": None, "signature": "sig_block1"}
            ]
        ),
        make_chunk(
            thinking_blocks=[
                {
                    "type": "redacted_thinking",
                    "data": "EuoBCoYBGAIi...encrypted...",
                }
            ]
        ),
        make_chunk(
            thinking_blocks=[
                {"type": "thinking", "thinking": "Step 2 analysis...", "signature": None}
            ]
        ),
        make_chunk(
            thinking_blocks=[
                {"type": "thinking", "thinking": None, "signature": "sig_block2"}
            ]
        ),
    ]

    thinking_chunks = [
        chunk for chunk in chunks if chunk["choices"][0]["delta"].get("thinking_blocks")
    ]
    processor = ChunkProcessor(chunks=chunks)
    result = processor.get_combined_thinking_content(thinking_chunks)

    assert result is not None
    assert len(result) == 3
    assert result[0]["type"] == "thinking"
    assert result[0]["thinking"] == "Step 1 analysis..."
    assert result[0]["signature"] == "sig_block1"
    assert result[1]["type"] == "redacted_thinking"
    assert result[1]["data"] == "EuoBCoYBGAIi...encrypted..."
    assert result[2]["type"] == "thinking"
    assert result[2]["thinking"] == "Step 2 analysis..."
    assert result[2]["signature"] == "sig_block2"


def test_cache_read_input_tokens_retained():
    chunk1 = ModelResponseStream(
        id="chatcmpl-95aabb85-c39f-443d-ae96-0370c404d70c",
        created=1745513206,
        model="claude-3-7-sonnet-20250219",
        object="chat.completion.chunk",
        system_fingerprint=None,
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(
                    provider_specific_fields=None,
                    content="",
                    role=None,
                    function_call=None,
                    tool_calls=None,
                    audio=None,
                ),
                logprobs=None,
            )
        ],
        provider_specific_fields=None,
        stream_options={"include_usage": True},
        usage=Usage(
            completion_tokens=5,
            prompt_tokens=11779,
            total_tokens=11784,
            completion_tokens_details=None,
            prompt_tokens_details=PromptTokensDetails(
                audio_tokens=None, cached_tokens=11775
            ),
            cache_creation_input_tokens=4,
            cache_read_input_tokens=11775,
        ),
    )

    chunk2 = ModelResponseStream(
        id="chatcmpl-95aabb85-c39f-443d-ae96-0370c404d70c",
        created=1745513207,
        model="claude-3-7-sonnet-20250219",
        object="chat.completion.chunk",
        system_fingerprint=None,
        choices=[
            StreamingChoices(
                finish_reason="stop",
                index=0,
                delta=Delta(
                    provider_specific_fields=None,
                    content=None,
                    role=None,
                    function_call=None,
                    tool_calls=None,
                    audio=None,
                ),
                logprobs=None,
            )
        ],
        provider_specific_fields=None,
        stream_options={"include_usage": True},
        usage=Usage(
            completion_tokens=214,
            prompt_tokens=0,
            total_tokens=214,
            completion_tokens_details=None,
            prompt_tokens_details=PromptTokensDetails(
                audio_tokens=None, cached_tokens=0
            ),
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )

    # Use dictionaries directly instead of ModelResponseStream
    chunks = [chunk1, chunk2]
    processor = ChunkProcessor(chunks=chunks)

    usage = processor.calculate_usage(
        chunks=chunks,
        model="claude-3-7-sonnet",
        completion_output="",
    )

    assert usage.cache_creation_input_tokens == 4
    assert usage.cache_read_input_tokens == 11775
    assert usage.prompt_tokens_details.cached_tokens == 11775


def test_stream_chunk_builder_litellm_usage_chunks():
    """
    Validate ChunkProcessor.calculate_usage uses provided usage fields from streaming chunks
    and reconstructs prompt and completion tokens without making any upstream API calls.
    """
    # Prepare two mocked streaming chunks with usage split across them
    chunk1 = ModelResponseStream(
        id="chatcmpl-mocked-usage-1",
        created=1745513206,
        model="gemini/gemini-2.5-flash-lite",
        object="chat.completion.chunk",
        system_fingerprint=None,
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(
                    provider_specific_fields=None,
                    content="",
                    role=None,
                    function_call=None,
                    tool_calls=None,
                    audio=None,
                ),
                logprobs=None,
            )
        ],
        provider_specific_fields=None,
        stream_options={"include_usage": True},
        usage=Usage(
            completion_tokens=0,
            prompt_tokens=50,
            total_tokens=50,
            completion_tokens_details=None,
            prompt_tokens_details=None,
        ),
    )

    chunk2 = ModelResponseStream(
        id="chatcmpl-mocked-usage-1",
        created=1745513207,
        model="gemini/gemini-2.5-flash-lite",
        object="chat.completion.chunk",
        system_fingerprint=None,
        choices=[
            StreamingChoices(
                finish_reason="stop",
                index=0,
                delta=Delta(
                    provider_specific_fields=None,
                    content=None,
                    role=None,
                    function_call=None,
                    tool_calls=None,
                    audio=None,
                ),
                logprobs=None,
            )
        ],
        provider_specific_fields=None,
        stream_options={"include_usage": True},
        usage=Usage(
            completion_tokens=27,
            prompt_tokens=0,
            total_tokens=27,
            completion_tokens_details=None,
            prompt_tokens_details=None,
        ),
    )

    chunks = [chunk1, chunk2]
    processor = ChunkProcessor(chunks=chunks)

    usage = processor.calculate_usage(
        chunks=chunks, model="gemini/gemini-2.5-flash-lite", completion_output=""
    )

    assert usage.prompt_tokens == 50
    assert usage.completion_tokens == 27
    assert usage.total_tokens == 77


def test_get_model_from_chunks_azure_model_router():
    """
    Test that _get_model_from_chunks finds the actual model from Azure Model Router chunks.
    
    Azure Model Router returns the request model (e.g., 'azure-model-router') in the first chunk,
    but subsequent chunks contain the actual model (e.g., 'gpt-4.1-nano-2025-04-14').
    This is important for accurate cost calculation.
    """
    # First chunk has request model, subsequent chunks have actual model
    chunks = [
        {"model": "azure-model-router", "id": "chatcmpl-123", "choices": []},
        {"model": "gpt-4.1-nano-2025-04-14", "id": "chatcmpl-123", "choices": []},
        {"model": "gpt-4.1-nano-2025-04-14", "id": "chatcmpl-123", "choices": []},
    ]
    
    result = ChunkProcessor._get_model_from_chunks(
        chunks=chunks, first_chunk_model="azure-model-router"
    )
    
    # Should return the actual model, not the request model
    assert result == "gpt-4.1-nano-2025-04-14"
    
    # Test when all chunks have the same model (non-router case)
    chunks_same_model = [
        {"model": "gpt-4", "id": "chatcmpl-456", "choices": []},
        {"model": "gpt-4", "id": "chatcmpl-456", "choices": []},
    ]
    
    result_same = ChunkProcessor._get_model_from_chunks(
        chunks=chunks_same_model, first_chunk_model="gpt-4"
    )
    
    # Should return the first chunk's model when all are the same
    assert result_same == "gpt-4"


def test_stream_chunk_builder_anthropic_web_search():
    # Prepare two mocked streaming chunks with usage split across them
    chunk1 = ModelResponseStream(
        id="chatcmpl-mocked-usage-1",
        created=1745513206,
        model="claude-sonnet-4-5-20250929",
        object="chat.completion.chunk",
        system_fingerprint=None,
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(
                    provider_specific_fields=None,
                    content="",
                    role=None,
                    function_call=None,
                    tool_calls=None,
                    audio=None,
                ),
                logprobs=None,
            )
        ],
        provider_specific_fields=None,
        stream_options={"include_usage": True},
        usage=Usage(
            completion_tokens=0,
            prompt_tokens=50,
            total_tokens=50,
            completion_tokens_details=None,
            server_tool_use=ServerToolUse(web_search_requests=2),
            prompt_tokens_details=None,
        ),
    )

    chunk2 = ModelResponseStream(
        id="chatcmpl-mocked-usage-1",
        created=1745513207,
        model="claude-sonnet-4-5-20250929",
        object="chat.completion.chunk",
        system_fingerprint=None,
        choices=[
            StreamingChoices(
                finish_reason="stop",
                index=0,
                delta=Delta(
                    provider_specific_fields=None,
                    content=None,
                    role=None,
                    function_call=None,
                    tool_calls=None,
                    audio=None,
                ),
                logprobs=None,
            )
        ],
        provider_specific_fields=None,
        stream_options={"include_usage": True},
        usage=Usage(
            completion_tokens=27,
            prompt_tokens=0,
            total_tokens=27,
            completion_tokens_details=None,
            prompt_tokens_details=None,
        ),
    )

    chunks = [chunk1, chunk2]
    processor = ChunkProcessor(chunks=chunks)

    usage = processor.calculate_usage(
        chunks=chunks, model="claude-sonnet-4-5-20250929", completion_output=""
    )

    assert usage.prompt_tokens == 50
    assert usage.completion_tokens == 27
    assert usage.total_tokens == 77    
    assert usage.server_tool_use['web_search_requests'] == 2
