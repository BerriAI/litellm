import json

import pytest


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


def test_calculate_usage_carries_google_maps_grounding_requests():
    """
    The Maps grounding counter set on a streamed usage chunk must survive the stream rebuild even
    when a later chunk carries its own prompt_tokens_details, or Maps grounding on streaming
    requests silently bills $0.
    """
    from litellm.types.utils import PromptTokensDetailsWrapper

    chunk1 = ModelResponseStream(
        id="chatcmpl-maps-usage-0",
        created=1745513207,
        model="gemini-2.5-flash",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(content="Here"),
                logprobs=None,
            )
        ],
        stream_options={"include_usage": True},
        usage=Usage(
            completion_tokens=0,
            prompt_tokens=15,
            total_tokens=15,
            prompt_tokens_details=PromptTokensDetailsWrapper(google_maps_grounding_requests=1),
        ),
    )

    chunk2 = ModelResponseStream(
        id="chatcmpl-maps-usage-0",
        created=1745513207,
        model="gemini-2.5-flash",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason="stop",
                index=0,
                delta=Delta(content=None),
                logprobs=None,
            )
        ],
        stream_options={"include_usage": True},
        usage=Usage(
            completion_tokens=27,
            prompt_tokens=0,
            total_tokens=27,
            prompt_tokens_details=PromptTokensDetailsWrapper(text_tokens=0),
        ),
    )

    chunks = [chunk1, chunk2]
    processor = ChunkProcessor(chunks=chunks)

    usage = processor.calculate_usage(chunks=chunks, model="gemini-2.5-flash", completion_output="")

    assert usage.prompt_tokens_details.google_maps_grounding_requests == 1


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


def test_cost_field_in_usage_chunks():
    chunk1_usage = Usage(completion_tokens=1, prompt_tokens=10, total_tokens=11)
    chunk1 = ModelResponseStream(
        id="chatcmpl-1",
        created=1745513206,
        model="openrouter/claude",
        choices=[
            StreamingChoices(finish_reason=None, index=0, delta=Delta(content="Hi"))
        ],
        usage=chunk1_usage,
    )

    chunk2_usage = Usage(
        completion_tokens=5, prompt_tokens=10, total_tokens=15, cost=0.00025
    )
    chunk2 = ModelResponseStream(
        id="chatcmpl-1",
        created=1745513207,
        model="openrouter/claude",
        choices=[
            StreamingChoices(finish_reason="stop", index=0, delta=Delta(content=""))
        ],
        usage=chunk2_usage,
    )

    processor = ChunkProcessor(chunks=[chunk1, chunk2])
    usage = processor.calculate_usage(
        chunks=[chunk1, chunk2], model="openrouter/claude", completion_output="Hi"
    )

    assert hasattr(usage, "cost")
    assert usage.cost == 0.00025
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 5


def test_stream_chunk_builder_tolerates_trailing_chunk_without_choices():
    """Regression for https://github.com/BerriAI/litellm/issues/32051

    The Responses-API bridge yields ModelResponseStream chunks with choices
    followed by a trailing event object that has no ``choices`` key. Building
    those chunks used to raise ``KeyError('choices')`` (surfaced as a 500
    APIError); it must now skip the choices-less chunk and assemble content.
    """
    from litellm.types.llms.base import BaseLiteLLMOpenAIResponseObject

    content_chunks = [
        ModelResponseStream(
            model="gpt-4o",
            choices=[StreamingChoices(index=0, delta=Delta(content=part))],
        )
        for part in ("Hello", " world")
    ]
    trailing_chunk = BaseLiteLLMOpenAIResponseObject()
    assert "choices" not in trailing_chunk

    response = stream_chunk_builder(chunks=content_chunks + [trailing_chunk])

    assert response is not None
    assert response.choices[0].message.content == "Hello world"


def test_anthropic_speed_and_geo_survive_stream_assembly():
    """Anthropic prices fast mode and non-global regions with a multiplier read off
    ``usage.speed`` / ``usage.inference_geo``. Dropping them while reassembling a stream
    bills streamed fast-mode calls at the standard rate."""
    from litellm.llms.anthropic.cost_calculation import cost_per_token

    def _usage(**extra):
        usage = Usage(completion_tokens=100, prompt_tokens=1000, total_tokens=1100)
        for key, value in extra.items():
            setattr(usage, key, value)
        return usage

    def _chunk(usage):
        return ModelResponseStream(
            id="chatcmpl-1",
            created=1745513206,
            model="claude-opus-4-8",
            choices=[StreamingChoices(finish_reason="stop", index=0, delta=Delta(content="Hi"))],
            usage=usage,
        )

    fast_chunk = _chunk(_usage(speed="fast", inference_geo="global"))
    fast_usage = ChunkProcessor(chunks=[fast_chunk]).calculate_usage(
        chunks=[fast_chunk], model="claude-opus-4-8", completion_output="Hi"
    )
    standard_chunk = _chunk(_usage(inference_geo="global"))
    standard_usage = ChunkProcessor(chunks=[standard_chunk]).calculate_usage(
        chunks=[standard_chunk], model="claude-opus-4-8", completion_output="Hi"
    )

    assert fast_usage.speed == "fast"
    assert fast_usage.inference_geo == "global"
    assert getattr(standard_usage, "speed", None) is None

    fast_cost = sum(cost_per_token(model="claude-opus-4-8", usage=fast_usage))
    standard_cost = sum(cost_per_token(model="claude-opus-4-8", usage=standard_usage))
    assert fast_cost == pytest.approx(standard_cost * 2.0)


def test_prompt_tokens_details_survive_later_usage_chunk_without_details():
    """Regression for #34801: a trailing usage chunk that omits
    `prompt_tokens_details` must not wipe the OpenAI cache-read/cache-write split,
    otherwise those tokens get re-priced at the uncached input rate."""
    from litellm.types.utils import PromptTokensDetailsWrapper

    chunk_with_details = ModelResponseStream(
        id="chatcmpl-1",
        created=1745513206,
        model="openai/gpt-5.6-sol",
        choices=[
            StreamingChoices(finish_reason=None, index=0, delta=Delta(content="Hi"))
        ],
        usage=Usage(
            prompt_tokens=6017,
            completion_tokens=4,
            total_tokens=6021,
            prompt_tokens_details=PromptTokensDetailsWrapper(
                cached_tokens=6004, cache_write_tokens=10
            ),
        ),
    )
    chunk_without_details = ModelResponseStream(
        id="chatcmpl-1",
        created=1745513207,
        model="openai/gpt-5.6-sol",
        choices=[
            StreamingChoices(finish_reason="stop", index=0, delta=Delta(content=""))
        ],
        usage=Usage(prompt_tokens=6017, completion_tokens=4, total_tokens=6021),
    )

    chunks = [chunk_with_details, chunk_without_details]
    usage = ChunkProcessor(chunks=chunks).calculate_usage(
        chunks=chunks, model="openai/gpt-5.6-sol", completion_output="Hi"
    )

    assert usage.prompt_tokens == 6017
    assert usage.prompt_tokens_details is not None
    assert usage.prompt_tokens_details.cached_tokens == 6004
    assert usage.prompt_tokens_details.cache_write_tokens == 10


def test_get_combined_tool_content_custom_tool_call():
    from litellm.litellm_core_utils.streaming_chunk_builder_utils import ChunkProcessor
    from litellm.types.utils import ChatCompletionMessageCustomToolCall

    processor = ChunkProcessor.__new__(ChunkProcessor)
    tool_call_chunks = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_TBs",
                                "type": "custom",
                                "custom": {"name": "ApplyPatch", "input": ""},
                            }
                        ]
                    }
                }
            ]
        },
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "custom": {"input": "*** Begin Patch\n"}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "custom": {"input": "*** End Patch\n"}}]}}]},
    ]
    combined = processor.get_combined_tool_content(tool_call_chunks)
    assert len(combined) == 1
    assert isinstance(combined[0], ChatCompletionMessageCustomToolCall)
    assert combined[0].model_dump() == {
        "id": "call_TBs",
        "type": "custom",
        "custom": {"name": "ApplyPatch", "input": "*** Begin Patch\n*** End Patch\n"},
    }


def test_get_combined_tool_content_custom_tool_call_without_type_field():
    """Delta coercion classifies a tool-call chunk as custom from its ``custom`` payload
    alone (``type`` may never arrive on any chunk). The assembler must use the same
    evidence; requiring ``type == "custom"`` dropped the whole tool call from the
    combined message (it matched neither the custom nor the function branch)."""
    from litellm.litellm_core_utils.streaming_chunk_builder_utils import ChunkProcessor
    from litellm.types.utils import ChatCompletionMessageCustomToolCall

    processor = ChunkProcessor.__new__(ChunkProcessor)
    tool_call_chunks = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_TBs",
                                "custom": {"name": "ApplyPatch", "input": "*** Begin"},
                            }
                        ]
                    }
                }
            ]
        },
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "custom": {"input": " Patch"}}]}}]},
    ]
    combined = processor.get_combined_tool_content(tool_call_chunks)
    assert len(combined) == 1
    assert isinstance(combined[0], ChatCompletionMessageCustomToolCall)
    assert combined[0].model_dump() == {
        "id": "call_TBs",
        "type": "custom",
        "custom": {"name": "ApplyPatch", "input": "*** Begin Patch"},
    }


def _tool_call_delta_chunk(tool_call: dict[str, object] | ChatCompletionDeltaToolCall) -> dict[str, object]:
    return {"choices": [{"delta": {"tool_calls": [tool_call]}}]}


def test_get_combined_tool_content_joins_many_dict_shaped_argument_fragments_in_order():
    processor = ChunkProcessor.__new__(ChunkProcessor)
    first_fragments = [f"a{i};" for i in range(300)]
    second_fragments = [f"b{i};" for i in range(300)]
    header_chunks = [
        _tool_call_delta_chunk({"index": 0, "id": "call_a", "type": "function", "function": {"name": "tool_a"}}),
        _tool_call_delta_chunk({"index": 1, "id": "call_b", "type": "function", "function": {"name": "tool_b"}}),
        _tool_call_delta_chunk({"index": 2, "id": "call_c", "type": "function", "function": {"name": "tool_c"}}),
    ]
    fragment_chunks = [
        _tool_call_delta_chunk({"index": index, "function": {"arguments": fragment}})
        for first, second in zip(first_fragments, second_fragments)
        for index, fragment in ((0, first), (1, second))
    ]

    combined = processor.get_combined_tool_content(header_chunks + fragment_chunks)

    assert [tool_call.id for tool_call in combined] == ["call_a", "call_b", "call_c"]
    assert combined[0].function.name == "tool_a"
    assert combined[0].function.arguments == "".join(first_fragments)
    assert combined[1].function.name == "tool_b"
    assert combined[1].function.arguments == "".join(second_fragments)
    assert combined[2].function.arguments == "{}"


def test_get_combined_tool_content_joins_fragments_across_many_parallel_tool_calls():
    processor = ChunkProcessor.__new__(ChunkProcessor)
    indexes = range(40)
    header_chunks = [
        _tool_call_delta_chunk(
            {"index": index, "id": f"call_{index}", "type": "function", "function": {"name": f"tool_{index}"}}
        )
        for index in indexes
    ]
    fragment_chunks = [
        _tool_call_delta_chunk({"index": index, "function": {"arguments": f"{index}.{position};"}})
        for position in range(5)
        for index in indexes
    ]

    combined = processor.get_combined_tool_content(header_chunks + fragment_chunks)

    assert [tool_call.id for tool_call in combined] == [f"call_{index}" for index in indexes]
    for index, tool_call in zip(indexes, combined):
        assert tool_call.function.arguments == "".join(f"{index}.{position};" for position in range(5))


def test_get_combined_tool_content_joins_many_object_shaped_argument_fragments_in_order():
    processor = ChunkProcessor.__new__(ChunkProcessor)
    first_fragments = [f"x{i}|" for i in range(300)]
    second_fragments = [f"y{i}|" for i in range(300)]
    header_chunks = [
        _tool_call_delta_chunk(
            ChatCompletionDeltaToolCall(
                id="call_x", type="function", index=0, function=Function(name="tool_x", arguments="")
            )
        ),
        _tool_call_delta_chunk(
            ChatCompletionDeltaToolCall(
                id="call_y", type="function", index=1, function=Function(name="tool_y", arguments="")
            )
        ),
    ]
    fragment_chunks = [
        _tool_call_delta_chunk(ChatCompletionDeltaToolCall(index=index, function=Function(arguments=fragment)))
        for first, second in zip(first_fragments, second_fragments)
        for index, fragment in ((0, first), (1, second))
    ]

    combined = processor.get_combined_tool_content(header_chunks + fragment_chunks)

    assert [tool_call.id for tool_call in combined] == ["call_x", "call_y"]
    assert combined[0].function.name == "tool_x"
    assert combined[0].function.arguments == "".join(first_fragments)
    assert combined[1].function.name == "tool_y"
    assert combined[1].function.arguments == "".join(second_fragments)


def test_get_combined_tool_content_joins_many_custom_tool_input_fragments_in_order():
    from types import SimpleNamespace

    from litellm.types.utils import ChatCompletionMessageCustomToolCall

    processor = ChunkProcessor.__new__(ChunkProcessor)
    dict_fragments = [f"d{i}," for i in range(200)]
    object_fragments = [f"o{i}," for i in range(200)]
    header_chunks = [
        _tool_call_delta_chunk({"index": 0, "id": "call_d", "type": "custom", "custom": {"name": "apply_patch"}}),
        _tool_call_delta_chunk(
            SimpleNamespace(index=1, id="call_o", type="custom", custom=SimpleNamespace(name="run_script", input=""))
        ),
    ]
    fragment_chunks = [
        _tool_call_delta_chunk(tool_call)
        for dict_fragment, object_fragment in zip(dict_fragments, object_fragments)
        for tool_call in (
            {"index": 0, "custom": {"input": dict_fragment}},
            SimpleNamespace(index=1, custom=SimpleNamespace(input=object_fragment)),
        )
    ]

    combined = processor.get_combined_tool_content(header_chunks + fragment_chunks)

    assert [tool_call.id for tool_call in combined] == ["call_d", "call_o"]
    assert isinstance(combined[0], ChatCompletionMessageCustomToolCall)
    assert combined[0].custom.name == "apply_patch"
    assert combined[0].custom.input == "".join(dict_fragments)
    assert isinstance(combined[1], ChatCompletionMessageCustomToolCall)
    assert combined[1].custom.name == "run_script"
    assert combined[1].custom.input == "".join(object_fragments)


def _reasoning_stream_chunk() -> ModelResponseStream:
    return ModelResponseStream(
        id="chatcmpl-reasoning",
        model="claude-opus-4-8",
        choices=[StreamingChoices(finish_reason=None, index=0, delta=Delta(content="10", role="assistant"))],
    )


def test_count_reasoning_tokens_returns_none_for_signature_only_thinking():
    from litellm.types.utils import Choices, Message, ModelResponse

    processor = ChunkProcessor(chunks=[_reasoning_stream_chunk()])
    response = ModelResponse(
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(content="10", role="assistant", reasoning_content=""),
            )
        ]
    )

    assert processor.count_reasoning_tokens(response) is None


def test_count_reasoning_tokens_counts_visible_reasoning():
    from litellm.types.utils import Choices, Message, ModelResponse

    processor = ChunkProcessor(chunks=[_reasoning_stream_chunk()])
    response = ModelResponse(
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(
                    content="10",
                    role="assistant",
                    reasoning_content="let me count the primes under thirty",
                ),
            )
        ]
    )

    assert processor.count_reasoning_tokens(response) > 0


@pytest.mark.parametrize(
    "estimated_reasoning_tokens, expected_reasoning_tokens, expected_text_tokens",
    [(40, 40, 60), (250, 100, 0)],
)
def test_calculate_usage_fills_unknown_split_from_reasoning_estimate(
    estimated_reasoning_tokens, expected_reasoning_tokens, expected_text_tokens
):
    from litellm.types.utils import CompletionTokensDetailsWrapper

    chunk = ModelResponseStream(
        id="chatcmpl-unknown-split",
        model="claude-opus-4-8",
        choices=[StreamingChoices(finish_reason="stop", index=0, delta=Delta(content=None, role=None))],
        usage=Usage(
            prompt_tokens=50,
            completion_tokens=100,
            total_tokens=150,
            completion_tokens_details=CompletionTokensDetailsWrapper(reasoning_tokens=None, text_tokens=None),
        ),
    )
    processor = ChunkProcessor(chunks=[chunk])

    usage = processor.calculate_usage(
        chunks=[chunk],
        model="claude-opus-4-8",
        completion_output="10",
        reasoning_tokens=estimated_reasoning_tokens,
    )

    assert usage.completion_tokens == 100
    assert usage.completion_tokens_details.reasoning_tokens == expected_reasoning_tokens
    assert usage.completion_tokens_details.text_tokens == expected_text_tokens
