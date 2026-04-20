import os
import sys
from typing import List, Sequence


sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
    AnthropicStreamWrapper,
)
from litellm.types.utils import (
    Delta,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
    Usage,
    ChatCompletionDeltaToolCall,
    Function,
)


class MockCompletionStream:
    def __init__(self, responses: Sequence[ModelResponseStream | ModelResponse]):
        self.responses = responses
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.responses):
            raise StopIteration
        response = self.responses[self.index]
        self.index += 1
        return response

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.responses):
            raise StopAsyncIteration
        response = self.responses[self.index]
        self.index += 1
        return response


def construct_text_chunk(text: str) -> ModelResponseStream:
    return ModelResponseStream(
        choices=[
            StreamingChoices(
                delta=Delta(content=text),
                index=0,
                finish_reason=None,
            )
        ],
    )


def construct_split_tool_call(
    id: str, function_name: str, function_arg_parts: List[str]
) -> List[ModelResponseStream]:
    return [
        # https://platform.openai.com/docs/guides/function-calling#streaming
        ModelResponseStream(
            choices=[
                StreamingChoices(
                    delta=Delta(
                        tool_calls=[
                            ChatCompletionDeltaToolCall(
                                id=id,
                                function=Function(arguments="", name=function_name),
                                index=0,
                                type="function",
                            )
                        ]
                    ),
                    index=0,
                    finish_reason=None,
                )
            ],
        ),
        *[
            ModelResponseStream(
                choices=[
                    StreamingChoices(
                        delta=Delta(
                            tool_calls=[
                                ChatCompletionDeltaToolCall(
                                    id=None,
                                    function=Function(arguments=part, name=None),
                                    index=0,
                                    type=None,
                                )
                            ]
                        ),
                        index=0,
                        finish_reason=None,
                    )
                ],
            )
            for part in function_arg_parts
        ],
    ]


def construct_parallel_tool_call_start(
    tool_calls: List[tuple[str, str, str]],
) -> ModelResponse:
    return ModelResponse(
        stream=True,
        choices=[
            StreamingChoices(
                delta=Delta(
                    tool_calls=[
                        ChatCompletionDeltaToolCall(
                            id=tool_id,
                            function=Function(arguments=arguments, name=function_name),
                            index=tool_index,
                            type="function",
                        )
                        for tool_index, (
                            tool_id,
                            function_name,
                            arguments,
                        ) in enumerate(tool_calls)
                    ]
                ),
                index=0,
                finish_reason=None,
            )
        ],
    )


def test_anthropic_stream_wrapper_single_tool_call():
    responses = [
        *construct_split_tool_call("tooluse_foo", "get_weather", ['{"city":', '"NY"}']),
        ModelResponseStream(
            choices=[
                StreamingChoices(
                    delta=Delta(content="", stop_reason="tool_calls"),
                    index=0,
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=230, completion_tokens=65, total_tokens=295),
        ),
    ]

    wrapper = AnthropicStreamWrapper(
        completion_stream=MockCompletionStream(responses),
        model="sonnet-4-5",
    )

    chunks = []
    chunk_types = []

    # Collect all chunks
    for chunk in wrapper:
        chunks.append(chunk)
        chunk_types.append(chunk.get("type"))

    # Verify the expected sequence of chunk types
    expected_types = [
        "message_start",  # Initial message start
        # TODO: for future contributors: if the initial content_block_start
        # respects the upstream's starting chunk, the initial empty text block
        # should be removed (and this test should be updated accordingly)
        # ---------------------------------------------------------------------
        "content_block_start",  # Initial empty text block start
        "content_block_stop",  # End of empty text block
        # ---------------------------------------------------------------------
        "content_block_start",  # Start of first tool_use content block
        "content_block_delta",  # {"city":
        "content_block_delta",  # "NY"}
        "content_block_stop",  # End of first tool_use content block
        "message_delta",  # Stop reason with merged usage
        "message_stop",  # Final message stop
    ]

    assert expected_types == chunk_types

    get_weather_calls = 0

    for chunk in chunks:
        if (
            chunk.get("type") == "content_block_start"
            and chunk["content_block"]["type"] == "tool_use"
        ):
            if chunk["content_block"]["name"] == "get_weather":
                get_weather_calls += 1

    assert get_weather_calls == 1


def test_anthropic_stream_wrapper_back_to_back_tool_calls():
    responses = [
        *construct_split_tool_call("tooluse_foo", "get_weather", ['{"city":', '"NY"}']),
        *construct_split_tool_call("tooluse_bar", "get_weather", ['{"city":', '"SF"}']),
        ModelResponseStream(
            choices=[
                StreamingChoices(
                    delta=Delta(content="", stop_reason="tool_calls"),
                    index=0,
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=230, completion_tokens=65, total_tokens=295),
        ),
    ]

    wrapper = AnthropicStreamWrapper(
        completion_stream=MockCompletionStream(responses),
        model="sonnet-4-5",
    )

    chunks = []
    chunk_types = []

    # Collect all chunks
    for chunk in wrapper:
        chunks.append(chunk)
        chunk_types.append(chunk.get("type"))

    # Verify the expected sequence of chunk types
    expected_types = [
        "message_start",  # Initial message start
        # TODO: for future contributors: if the initial content_block_start
        # respects the upstream's starting chunk, the initial empty text block
        # should be removed (and this test should be updated accordingly)
        # ---------------------------------------------------------------------
        "content_block_start",  # Initial empty text block start
        "content_block_stop",  # End of empty text block
        # ---------------------------------------------------------------------
        "content_block_start",  # Start of first tool_use content block
        "content_block_delta",  # {"city":
        "content_block_delta",  # "NY"}
        "content_block_stop",  # End of first tool_use content block
        "content_block_start",  # Start of second tool_use content block
        "content_block_delta",  # {"city":
        "content_block_delta",  # " SF"}
        "content_block_stop",  # End of second tool_use content block
        "message_delta",  # Stop reason with merged usage
        "message_stop",  # Final message stop
    ]

    assert expected_types == chunk_types

    get_weather_calls = 0

    for chunk in chunks:
        if (
            chunk.get("type") == "content_block_start"
            and chunk["content_block"]["type"] == "tool_use"
        ):
            if chunk["content_block"]["name"] == "get_weather":
                get_weather_calls += 1

    assert get_weather_calls == 2


def test_anthropic_stream_wrapper_interleaved_tool_calls_and_text():
    responses = [
        *construct_split_tool_call("tooluse_foo", "get_weather", ['{"city":', '"NY"}']),
        construct_text_chunk("The weather is nice today."),
        *construct_split_tool_call("tooluse_bar", "get_weather", ['{"city":', '"SF"}']),
        *construct_split_tool_call(
            "tooluse_bar", "get_weather", ['{"city":', '"CHI"}']
        ),
        construct_text_chunk("The weather is not so nice today."),
        ModelResponseStream(
            choices=[
                StreamingChoices(
                    delta=Delta(content="", stop_reason="tool_calls"),
                    index=0,
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=230, completion_tokens=65, total_tokens=295),
        ),
    ]

    wrapper = AnthropicStreamWrapper(
        completion_stream=MockCompletionStream(responses),
        model="sonnet-4-5",
    )

    chunks = []
    chunk_types = []

    # Collect all chunks
    for chunk in wrapper:
        chunks.append(chunk)
        chunk_types.append(chunk.get("type"))

    # Verify the expected sequence of chunk types
    expected_types = [
        "message_start",  # Initial message start
        # TODO: for future contributors: if the initial content_block_start
        # respects the upstream's starting chunk, the initial empty text block
        # should be removed (and this test should be updated accordingly)
        # ---------------------------------------------------------------------
        "content_block_start",  # Initial empty text block start
        "content_block_stop",  # End of empty text block
        # ---------------------------------------------------------------------
        "content_block_start",  # Start of first tool_use content block
        "content_block_delta",  # {"city":
        "content_block_delta",  # "NY"}
        "content_block_stop",  # End of first tool_use content block
        "content_block_start",  # "The weather is nice today"
        "content_block_delta",  # "The weather is nice today."
        "content_block_stop",
        "content_block_start",  # Start of second tool_use content block
        "content_block_delta",  # {"city":
        "content_block_delta",  # " SF"}
        "content_block_stop",  # End of second tool_use content block
        "content_block_start",  # Start of third tool_use content block
        "content_block_delta",  # {"city":
        "content_block_delta",  # " CHI"}
        "content_block_stop",  # End of third tool_use content block
        "content_block_start",  # "The weather is not so nice today"
        "content_block_delta",  # "The weather is not so nice today."
        "content_block_stop",
        "message_delta",  # Stop reason with merged usage
        "message_stop",  # Final message stop
    ]

    assert expected_types == chunk_types

    get_weather_calls = 0

    for chunk in chunks:
        if (
            chunk.get("type") == "content_block_start"
            and chunk["content_block"]["type"] == "tool_use"
        ):
            if chunk["content_block"]["name"] == "get_weather":
                get_weather_calls += 1

    assert get_weather_calls == 3


def test_anthropic_stream_wrapper_parallel_tool_calls_same_chunk():
    responses = [
        construct_parallel_tool_call_start(
            [
                ("tooluse_foo", "get_weather", '{"city":"NY"}'),
                ("tooluse_bar", "get_time", '{"timezone":"UTC"}'),
            ]
        ),
        ModelResponse(
            stream=True,
            choices=[
                StreamingChoices(
                    delta=Delta(content="", stop_reason="tool_calls"),
                    index=0,
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=230, completion_tokens=65, total_tokens=295),
        ),
    ]

    wrapper = AnthropicStreamWrapper(
        completion_stream=MockCompletionStream(responses),
        model="sonnet-4-5",
    )

    chunks = []
    chunk_types = []

    for chunk in wrapper:
        chunks.append(chunk)
        chunk_types.append(chunk.get("type"))

    expected_types = [
        "message_start",
        "content_block_start",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]

    assert expected_types == chunk_types

    tool_starts = [
        chunk
        for chunk in chunks
        if chunk.get("type") == "content_block_start"
        and chunk["content_block"]["type"] == "tool_use"
    ]

    assert [chunk["content_block"]["id"] for chunk in tool_starts] == [
        "tooluse_foo",
        "tooluse_bar",
    ]
    assert [chunk["content_block"]["name"] for chunk in tool_starts] == [
        "get_weather",
        "get_time",
    ]

    tool_deltas = [
        chunk
        for chunk in chunks
        if chunk.get("type") == "content_block_delta"
        and chunk.get("delta", {}).get("type") == "input_json_delta"
    ]
    assert [chunk["delta"]["partial_json"] for chunk in tool_deltas] == [
        '{"city":"NY"}',
        '{"timezone":"UTC"}',
    ]


def test_anthropic_stream_wrapper_parallel_tool_call_index_continuations():
    responses = [
        construct_parallel_tool_call_start(
            [
                ("tooluse_foo", "get_weather", '{"city":"NY"}'),
                ("tooluse_bar", "get_time", ""),
            ]
        ),
        ModelResponse(
            stream=True,
            choices=[
                StreamingChoices(
                    delta=Delta(
                        tool_calls=[
                            ChatCompletionDeltaToolCall(
                                id=None,
                                function=Function(
                                    arguments='{"timezone":"UTC"}', name=None
                                ),
                                index=1,
                                type=None,
                            )
                        ]
                    ),
                    index=0,
                    finish_reason=None,
                )
            ],
        ),
        ModelResponse(
            stream=True,
            choices=[
                StreamingChoices(
                    delta=Delta(
                        tool_calls=[
                            ChatCompletionDeltaToolCall(
                                id=None,
                                function=Function(arguments="", name=None),
                                index=1,
                                type=None,
                            )
                        ]
                    ),
                    index=0,
                    finish_reason=None,
                )
            ],
        ),
        ModelResponse(
            stream=True,
            choices=[
                StreamingChoices(
                    delta=Delta(content="", stop_reason="tool_calls"),
                    index=0,
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=230, completion_tokens=65, total_tokens=295),
        ),
    ]

    wrapper = AnthropicStreamWrapper(
        completion_stream=MockCompletionStream(responses),
        model="sonnet-4-5",
    )

    chunks = list(wrapper)

    tool_starts = [
        chunk
        for chunk in chunks
        if chunk.get("type") == "content_block_start"
        and chunk["content_block"]["type"] == "tool_use"
    ]

    assert [chunk["content_block"]["id"] for chunk in tool_starts] == [
        "tooluse_foo",
        "tooluse_bar",
    ]

    tool_deltas = [
        chunk
        for chunk in chunks
        if chunk.get("type") == "content_block_delta"
        and chunk.get("delta", {}).get("type") == "input_json_delta"
    ]

    assert [
        (chunk["index"], chunk["delta"]["partial_json"]) for chunk in tool_deltas
    ] == [
        (1, '{"city":"NY"}'),
        (2, '{"timezone":"UTC"}'),
    ]
