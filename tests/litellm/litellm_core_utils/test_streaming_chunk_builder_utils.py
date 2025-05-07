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
    StreamingChoices,
    Usage,
    PromptTokensDetails,
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
