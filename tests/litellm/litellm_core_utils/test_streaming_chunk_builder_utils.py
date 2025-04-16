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
