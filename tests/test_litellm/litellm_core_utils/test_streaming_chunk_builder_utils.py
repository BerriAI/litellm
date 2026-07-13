import json
import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm import ChatCompletionUsageBlock, stream_chunk_builder
from litellm.types.utils import GenericStreamingChunk
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
                {
                    "type": "thinking",
                    "thinking": "Step 1 analysis...",
                    "signature": None,
                }
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
                {
                    "type": "thinking",
                    "thinking": "Step 2 analysis...",
                    "signature": None,
                }
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


def test_streaming_preserves_anthropic_1hr_cache_creation_breakdown():
    """
    Anthropic emits the cache-creation TTL breakdown (ephemeral 5m/1h split) only
    on the `message_start` SSE event; the later `message_delta` carries the flat
    cache-creation count but drops the nested `cache_creation` object. Because
    prompt_tokens_details is aggregated last-wins, the breakdown used to be
    clobbered by message_delta, leaving cost calc with no TTL split. It then fell
    back to the 5-minute write rate and undercounted 1-hour cache writes by ~37.5%.

    Reproduces the trace: input=3, cache_creation=50 (all 1h), cache_read=8728.
    Correct cache-write cost is 50 * 6e-06 (1h) = 0.0003, not 50 * 3.75e-06 = 0.0001875.
    """
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig
    from litellm.llms.anthropic.cost_calculation import cost_per_token

    config = AnthropicConfig()
    message_start_usage = config.calculate_usage(
        usage_object={
            "input_tokens": 3,
            "cache_creation_input_tokens": 50,
            "cache_read_input_tokens": 8728,
            "output_tokens": 1,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 0,
                "ephemeral_1h_input_tokens": 50,
            },
        },
        reasoning_content=None,
    )
    message_delta_usage = config.calculate_usage(
        usage_object={
            "input_tokens": 3,
            "cache_creation_input_tokens": 50,
            "cache_read_input_tokens": 8728,
            "output_tokens": 31,
        },
        reasoning_content=None,
    )
    # Sanity: the delta event genuinely lacks the breakdown - this is the input
    # condition that used to defeat cost calc.
    assert (
        getattr(message_delta_usage.prompt_tokens_details, "cache_creation_token_details", None)
        is None
    )

    def _usage_chunk(usage, finish_reason):
        return ModelResponseStream(
            id="chatcmpl-1hr-cache",
            created=1745513206,
            model="claude-sonnet-4-6",
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    finish_reason=finish_reason,
                    index=0,
                    delta=Delta(content="" if finish_reason is None else None),
                )
            ],
            stream_options={"include_usage": True},
            usage=usage,
        )

    chunks = [
        _usage_chunk(message_start_usage, None),
        _usage_chunk(message_delta_usage, "stop"),
    ]
    usage = ChunkProcessor(chunks=chunks).calculate_usage(
        chunks=chunks, model="claude-sonnet-4-6", completion_output="hi"
    )

    breakdown = getattr(usage.prompt_tokens_details, "cache_creation_token_details", None)
    assert breakdown is not None, "1h/5m cache-creation breakdown lost during aggregation"
    assert breakdown.ephemeral_1h_input_tokens == 50
    assert breakdown.ephemeral_5m_input_tokens == 0
    assert usage.cache_creation_input_tokens == 50
    assert usage.cache_read_input_tokens == 8728

    prompt_cost, _ = cost_per_token(model="claude-sonnet-4-6", usage=usage)
    # text 3*3e-06 + cache_read 8728*3e-07 + cache_write 50*6e-06 (1h rate)
    expected = 3 * 3e-06 + 8728 * 3e-07 + 50 * 6e-06
    assert prompt_cost == pytest.approx(expected)
    # Guard against the regression: 5m-rate fallback would shave the write cost.
    buggy = 3 * 3e-06 + 8728 * 3e-07 + 50 * 3.75e-06
    assert prompt_cost != pytest.approx(buggy)


def test_streaming_keeps_cache_creation_breakdown_from_final_chunk():
    """When the final usage chunk itself carries the cache-creation breakdown,
    aggregation must keep that breakdown instead of re-attaching a stale one
    captured from an earlier chunk."""
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    config = AnthropicConfig()
    message_start_usage = config.calculate_usage(
        usage_object={
            "input_tokens": 3,
            "cache_creation_input_tokens": 7,
            "cache_read_input_tokens": 0,
            "output_tokens": 1,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 7,
                "ephemeral_1h_input_tokens": 0,
            },
        },
        reasoning_content=None,
    )
    message_delta_usage = config.calculate_usage(
        usage_object={
            "input_tokens": 3,
            "cache_creation_input_tokens": 50,
            "cache_read_input_tokens": 0,
            "output_tokens": 31,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 0,
                "ephemeral_1h_input_tokens": 50,
            },
        },
        reasoning_content=None,
    )

    def _usage_chunk(usage, finish_reason):
        return ModelResponseStream(
            id="chatcmpl-final-breakdown",
            created=1745513206,
            model="claude-sonnet-4-6",
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    finish_reason=finish_reason,
                    index=0,
                    delta=Delta(content="" if finish_reason is None else None),
                )
            ],
            stream_options={"include_usage": True},
            usage=usage,
        )

    chunks = [
        _usage_chunk(message_start_usage, None),
        _usage_chunk(message_delta_usage, "stop"),
    ]
    usage = ChunkProcessor(chunks=chunks).calculate_usage(
        chunks=chunks, model="claude-sonnet-4-6", completion_output="hi"
    )

    breakdown = getattr(usage.prompt_tokens_details, "cache_creation_token_details", None)
    assert breakdown is not None
    assert breakdown.ephemeral_1h_input_tokens == 50
    assert breakdown.ephemeral_5m_input_tokens == 0
    assert usage.cache_creation_input_tokens == 50


def test_cache_read_input_tokens_retained_genericstreamingchunk():
    chunk1 = GenericStreamingChunk(
        text="Test1",
        is_finished=False,
        finish_reason="",
        usage=None,
        index=1,
    )

    chunk2 = GenericStreamingChunk(
        text="Test2",
        is_finished=True,
        finish_reason="stop",
        usage=ChatCompletionUsageBlock(
            completion_tokens=5,
            prompt_tokens=1234,
            total_tokens=1239,
            completion_tokens_details=None,
            prompt_tokens_details=PromptTokensDetails(
                audio_tokens=None, cached_tokens=543
            ).model_dump(),
        ),
        index=2,
    )

    # Use dictionaries directly instead of ModelResponseStream
    chunks = [chunk1, chunk2]
    processor = ChunkProcessor(chunks=chunks)

    usage = processor.calculate_usage(
        chunks=chunks,
        model="gpt-5.5",
        completion_output="",
    )

    assert usage.prompt_tokens_details.cached_tokens == 543

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
    # server_tool_use must be a ServerToolUse pydantic so downstream cost-calc
    # (which uses attribute access) works. See issue #26153.
    assert isinstance(usage.server_tool_use, ServerToolUse)
    assert usage.server_tool_use.web_search_requests == 2


def test_sort_chunks_handles_dict_hidden_params_created_at():
    chunks = [
        {
            "id": "chunk_2",
            "object": "chat.completion.chunk",
            "created": 2,
            "model": "gpt-4.1-mini",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "b"}}],
            "_hidden_params": {"created_at": 2},
        },
        {
            "id": "chunk_1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4.1-mini",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "a"}}],
            "_hidden_params": {"created_at": 1},
        },
    ]

    processor = ChunkProcessor(chunks=chunks)
    assert processor.chunks[0]["id"] == "chunk_1"
    assert processor.chunks[1]["id"] == "chunk_2"


def test_stream_chunk_builder_accepts_dict_snapshot_chunks():
    chunk1 = ModelResponseStream(
        id="chatcmpl-123",
        created=1,
        model="gpt-4.1-mini",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(content="Hello ", role="assistant"),
            )
        ],
    )
    chunk2 = ModelResponseStream(
        id="chatcmpl-123",
        created=2,
        model="gpt-4.1-mini",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason="stop",
                index=0,
                delta=Delta(content="world", role=None),
            )
        ],
    )
    chunk1._hidden_params = {"created_at": 1}
    chunk2._hidden_params = {"created_at": 2}

    chunks = []
    for chunk in [chunk2, chunk1]:
        chunk_dict = chunk.model_dump()
        chunk_dict["_hidden_params"] = chunk._hidden_params
        chunks.append(chunk_dict)

    response = stream_chunk_builder(chunks=chunks)
    assert response is not None
    assert response.choices[0].message.content == "Hello world"


def test_stream_chunk_builder_dict_snapshot_preserves_hidden_provider_fields():
    chunk = ModelResponseStream(
        id="chatcmpl-123",
        created=1,
        model="gpt-4.1-mini",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason="stop",
                index=0,
                delta=Delta(content="hi", role="assistant"),
            )
        ],
    )
    chunk_dict = chunk.model_dump()
    chunk_dict["_hidden_params"] = {
        "provider_specific_fields": {"traffic_type": "default"}
    }

    response = stream_chunk_builder(chunks=[chunk_dict])
    assert response is not None
    assert (
        response._hidden_params["provider_specific_fields"]["traffic_type"] == "default"
    )


def test_stream_chunk_builder_propagates_vertex_ai_metadata_from_chunks():
    """Vertex AI metadata on streaming chunks must appear on assembled response."""
    grounding_metadata = [{"webSearchQueries": ["weather in SF"]}]
    url_context_metadata = [{"urlMetadata": [{"retrievedUrl": "https://example.com"}]}]

    chunk1 = ModelResponseStream(
        id="chatcmpl-vertex-1",
        created=1,
        model="gemini-2.5-flash",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(content="The weather", role="assistant"),
            )
        ],
    )
    setattr(chunk1, "vertex_ai_grounding_metadata", grounding_metadata)
    chunk1._hidden_params["vertex_ai_grounding_metadata"] = grounding_metadata

    chunk2 = ModelResponseStream(
        id="chatcmpl-vertex-1",
        created=1,
        model="gemini-2.5-flash",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason="stop",
                index=0,
                delta=Delta(content=" is sunny.", role="assistant"),
            )
        ],
    )
    setattr(chunk2, "vertex_ai_url_context_metadata", url_context_metadata)
    chunk2._hidden_params["vertex_ai_url_context_metadata"] = url_context_metadata

    response = stream_chunk_builder(chunks=[chunk1, chunk2])
    assert response is not None
    assert getattr(response, "vertex_ai_grounding_metadata") == grounding_metadata
    assert getattr(response, "vertex_ai_url_context_metadata") == url_context_metadata
    assert response._hidden_params["vertex_ai_grounding_metadata"] == grounding_metadata
    assert (
        response._hidden_params["vertex_ai_url_context_metadata"]
        == url_context_metadata
    )

    dumped = response.model_dump()
    assert dumped["vertex_ai_grounding_metadata"] == grounding_metadata
    assert dumped["vertex_ai_url_context_metadata"] == url_context_metadata


def test_stream_chunk_builder_uses_assembled_model_for_provider_metadata():
    grounding_metadata = [{"webSearchQueries": ["weather in SF"]}]

    chunk1 = ModelResponseStream(
        id="chatcmpl-vertex-router",
        created=1,
        model="gpt-4o",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(content="The weather", role="assistant"),
            )
        ],
    )
    chunk2 = ModelResponseStream(
        id="chatcmpl-vertex-router",
        created=1,
        model="gemini-2.5-flash",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason="stop",
                index=0,
                delta=Delta(content=" is sunny.", role=None),
            )
        ],
    )
    setattr(chunk2, "vertex_ai_grounding_metadata", grounding_metadata)
    chunk2._hidden_params["vertex_ai_grounding_metadata"] = grounding_metadata

    response = stream_chunk_builder(chunks=[chunk1, chunk2])
    assert response is not None
    assert response.model == "gemini-2.5-flash"
    assert getattr(response, "vertex_ai_grounding_metadata") == grounding_metadata


def test_stream_chunk_builder_propagates_vertex_ai_safety_results():
    """Assembled response must expose safety data under the non-streaming field name."""
    safety_ratings = [
        [{"category": "HARM_CATEGORY_HATE_SPEECH", "probability": "NEGLIGIBLE"}]
    ]

    chunk = ModelResponseStream(
        id="chatcmpl-vertex-safety",
        created=1,
        model="gemini-2.5-flash",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason="stop",
                index=0,
                delta=Delta(content="hello", role="assistant"),
            )
        ],
    )
    setattr(chunk, "vertex_ai_safety_ratings", safety_ratings)
    setattr(chunk, "vertex_ai_safety_results", safety_ratings)
    chunk._hidden_params["vertex_ai_safety_ratings"] = safety_ratings
    chunk._hidden_params["vertex_ai_safety_results"] = safety_ratings

    response = stream_chunk_builder(chunks=[chunk])
    assert response is not None
    assert getattr(response, "vertex_ai_safety_results") == safety_ratings
    assert response._hidden_params["vertex_ai_safety_results"] == safety_ratings
    assert response.model_dump()["vertex_ai_safety_results"] == safety_ratings


def test_stream_chunk_builder_propagates_vertex_ai_metadata_from_dict_chunks():
    """Dict snapshot chunks (model_dump) should also propagate Vertex AI metadata."""
    chunk_dict = ModelResponseStream(
        id="chatcmpl-vertex-2",
        created=1,
        model="gemini-2.5-flash",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason="stop",
                index=0,
                delta=Delta(content="hello", role="assistant"),
            )
        ],
    ).model_dump()
    chunk_dict["_hidden_params"] = {
        "vertex_ai_grounding_metadata": [{"webSearchQueries": ["test query"]}]
    }

    response = stream_chunk_builder(chunks=[chunk_dict])
    assert response is not None
    assert getattr(response, "vertex_ai_grounding_metadata") == [
        {"webSearchQueries": ["test query"]}
    ]
    assert response.model_dump()["vertex_ai_grounding_metadata"] == [
        {"webSearchQueries": ["test query"]}
    ]
