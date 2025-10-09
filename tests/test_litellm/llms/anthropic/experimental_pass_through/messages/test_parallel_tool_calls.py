import os
import sys
from typing import List


sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
    AnthropicStreamWrapper,
)
from litellm.types.utils import (
    Delta,
    ModelResponse,
    StreamingChoices,
    Usage,
    ChatCompletionDeltaToolCall,
    Function,
)


class MockCompletionStream:
    def __init__(self, responses: List[ModelResponse]):
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


def construct_text_chunk(text: str) -> ModelResponse:
    return ModelResponse(
        stream=True,
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
) -> List[ModelResponse]:
    return [
        # https://platform.openai.com/docs/guides/function-calling#streaming
        ModelResponse(
            stream=True,
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
            ModelResponse(
                stream=True,
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


def test_anthropic_stream_wrapper_single_tool_call():
    responses = [
        *construct_split_tool_call("tooluse_foo", "get_weather", ['{"city":', '"NY"}']),
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
