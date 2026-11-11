from collections.abc import AsyncIterator, Iterator, Sequence
from itertools import chain
from typing import Final

import pytest
from openai.types.completion import Completion

from litellm.exceptions import APIConnectionError
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.text_completions.streaming_iterator import TextCompletionStreamWrapper
from litellm.types.utils import Choices, Delta, ModelResponseStream, StreamingChoices, TextCompletionResponse, Usage


def test_reasoning_content_preserved_in_text_completion_wrapper():
    """Ensure reasoning_content is copied from delta to text_choices."""
    chunk = ModelResponseStream(
        id="test-id",
        created=1234567890,
        model="test-model",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(
                    content="Some answer text",
                    role="assistant",
                    reasoning_content="Here's my chain of thought...",
                ),
            )
        ],
    )

    wrapper = TextCompletionStreamWrapper(
        completion_stream=None,  # Not used in convert_to_text_completion_object
        model="test-model",
        stream_options=None,
    )

    transformed = wrapper.convert_to_text_completion_object(chunk)

    assert "choices" in transformed
    assert len(transformed["choices"]) == 1
    choice = transformed["choices"][0]
    assert choice["text"] == "Some answer text"
    assert choice["reasoning_content"] == "Here's my chain of thought..."


@pytest.mark.asyncio
@pytest.mark.parametrize("async_stream", (False, True))
@pytest.mark.parametrize("empty_choices", (False, True))
@pytest.mark.parametrize("stream_options", (None, {}, {"include_usage": False}, {"include_usage": True}))
@pytest.mark.parametrize("usage", (None, Usage(prompt_tokens=7, completion_tokens=2, total_tokens=9), Usage()))
async def test_text_completion_stream_usage_is_opt_in(
    async_stream: bool, empty_choices: bool, stream_options: dict[str, bool] | None, usage: Usage | None
) -> None:
    chunk: Final = ModelResponseStream(
        id="test-id",
        created=1234567890,
        model="test-model",
        choices=[] if empty_choices else [StreamingChoices(index=0, delta=Delta(content="Hello"))],
        usage=usage,
    )

    async def chunks() -> AsyncIterator[ModelResponseStream]:
        yield chunk

    wrapper: Final = TextCompletionStreamWrapper(
        completion_stream=chunks() if async_stream else iter((chunk,)),
        model="test-model",
        stream_options=stream_options,
    )
    responses: Final = [response async for response in wrapper] if async_stream else list(wrapper)
    expected_usage: Final = (
        {"usage": usage.model_dump(exclude_none=True)}
        if stream_options and stream_options.get("include_usage") is True and usage is not None
        else {}
    )
    assert [response.model_dump(exclude_none=True) for response in responses] == [
        {
            "id": "test-id",
            "object": "text_completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [] if empty_choices else [{"index": 0, "text": "Hello"}],
            **expected_usage,
        }
    ]


def _completion_chunk(text: str, finish_reason: str | None, **choice_extras: object) -> Completion:
    return Completion.construct(
        id="cmpl-1",
        created=1,
        model="m",
        object="text_completion",
        choices=[{"index": 0, "text": text, "finish_reason": finish_reason, "logprobs": None, **choice_extras}],
    )


def _text_completion_stream(upstream: Sequence[Completion]) -> list[TextCompletionResponse]:
    logging_obj: Final = Logging(
        model="m",
        messages=[{"role": "user", "content": "1, 2, 3, 4"}],
        stream=True,
        call_type="text_completion",
        start_time=0.0,
        litellm_call_id="call-1",
        function_id="function-1",
    )
    chat_stream: Final = CustomStreamWrapper(
        completion_stream=iter(upstream),
        model="m",
        custom_llm_provider="text-completion-openai",
        logging_obj=logging_obj,
    )
    return list(TextCompletionStreamWrapper(completion_stream=chat_stream, model="m"))


@pytest.mark.parametrize("final_text", [", 5, 6", ""], ids=["text-then-finish", "finish-only"])
def test_text_completion_stream_keeps_stop_reason_on_the_finish_chunk(final_text: str) -> None:
    responses: Final = _text_completion_stream(
        (_completion_chunk(", 5", None), _completion_chunk(final_text, "stop", stop_reason=", 7"))
    )

    assert responses[-1].choices[0].model_dump(exclude_none=True) == {
        "finish_reason": "stop",
        "index": 0,
        "stop_reason": ", 7",
    }


def test_text_completion_stream_puts_stop_reason_on_the_final_text_chunk() -> None:
    responses: Final = _text_completion_stream(
        (_completion_chunk(", 5", None), _completion_chunk(", 6", "stop", stop_reason=", 7"))
    )

    assert [(response.choices[0].get("text"), response.choices[0].get("stop_reason")) for response in responses] == [
        (", 5", None),
        (", 6", ", 7"),
        (None, ", 7"),
    ]


@pytest.mark.parametrize("final_text", [", 6", ""], ids=["text-then-finish", "finish-only"])
def test_text_completion_stream_keeps_held_back_token_ids_and_the_stop_reason(final_text: str) -> None:
    responses: Final = _text_completion_stream(
        (
            _completion_chunk("", None, token_ids=[11]),
            _completion_chunk("", None, token_ids=[220]),
            _completion_chunk(", 5", None, token_ids=[20]),
            _completion_chunk(final_text, "stop", stop_reason=", 7", token_ids=[11, 220, 21]),
        )
    )
    choices: Final = [response.choices[0] for response in responses]

    assert list(chain.from_iterable(choice.get("token_ids") or () for choice in choices)) == [11, 220, 20, 11, 220, 21]
    assert [choice.get("stop_reason") for choice in choices if choice.finish_reason is not None] == [", 7"]


def test_text_completion_stream_sends_empty_text_on_token_id_only_chunks() -> None:
    responses: Final = _text_completion_stream(
        (
            _completion_chunk("", None, token_ids=[11]),
            _completion_chunk("", None, token_ids=[220]),
            _completion_chunk(", 5", None, token_ids=[20]),
        )
    )

    choices: Final = [response.choices[0] for response in responses]

    assert [(choice.text, choice.get("token_ids")) for choice in choices if choice.get("token_ids")] == [
        ("", [11]),
        ("", [220]),
        (", 5", [20]),
    ]


def test_text_completion_stream_keeps_each_choices_text_and_token_ids_on_its_own_index() -> None:
    responses: Final = _text_completion_stream(
        (
            _completion_chunk(" 5", None, index=0, token_ids=[20]),
            _completion_chunk(" 7", None, index=1, token_ids=[22]),
            _completion_chunk(", 6", None, index=0, token_ids=[11, 21]),
            _completion_chunk("", None, index=1, token_ids=[11]),
            _completion_chunk("", "stop", index=0, stop_reason=", 9", token_ids=[11, 220]),
            _completion_chunk(", 8", "length", index=1, token_ids=[220, 23]),
        )
    )
    choices: Final = tuple(chain.from_iterable(response.choices for response in responses))

    assert {
        index: (
            "".join(choice.text or "" for choice in choices if choice.index == index),
            list(chain.from_iterable(choice.get("token_ids") or () for choice in choices if choice.index == index)),
        )
        for index in (0, 1)
    } == {0: (" 5, 6", [20, 11, 21, 11, 220]), 1: (" 7, 8", [22, 11, 220, 23])}


@pytest.mark.parametrize(
    "upstream",
    [
        (
            _completion_chunk(" a", None, index=0),
            _completion_chunk(" b", None, index=1),
            _completion_chunk("", "length", index=2),
            _completion_chunk(" red", "stop", index=1, stop_reason=","),
            _completion_chunk(" s", "stop", index=0, stop_reason=","),
        ),
        (
            _completion_chunk(" a", None, index=0),
            _completion_chunk(" b", None, index=1),
            _completion_chunk(" s", "stop", index=0, stop_reason=","),
            _completion_chunk("", "length", index=2),
            _completion_chunk(" red", "stop", index=1, stop_reason=","),
        ),
    ],
    ids=["other-choices-finish-first", "choice-0-finishes-first"],
)
def test_text_completion_stream_finishes_each_choice_on_its_own_index(upstream: Sequence[Completion]) -> None:
    finished: Final = [
        (choice.index, choice.finish_reason, choice.get("stop_reason"))
        for choice in chain.from_iterable(response.choices for response in _text_completion_stream(upstream))
        if choice.finish_reason is not None
    ]

    assert sorted(finished) == [(0, "stop", ","), (1, "stop", ","), (2, "length", None)]


def test_text_completion_conversion_rejects_non_streaming_choices() -> None:
    chunk: Final = Choices()
    wrapper: Final = TextCompletionStreamWrapper(completion_stream=None, model="test-model")

    with pytest.raises(Exception, match="Error occurred converting to text completion object") as error:
        wrapper.convert_to_text_completion_object(chunk)

    assert str(chunk) in str(error.value)


@pytest.mark.parametrize("chunk", (None, "None"))
def test_sync_text_completion_stream_rejects_invalid_chunks(chunk: str | None) -> None:
    wrapper: Final = TextCompletionStreamWrapper(
        completion_stream=iter((chunk,)), model="test-model", custom_llm_provider="openai"
    )

    with pytest.raises(APIConnectionError, match="OpenAIException") as error:
        next(wrapper)

    assert (error.value.model, error.value.llm_provider) == ("test-model", "openai")


@pytest.mark.asyncio
@pytest.mark.parametrize("chunk", (None, "None"))
async def test_async_text_completion_stream_rejects_invalid_chunks(chunk: str | None) -> None:
    async def chunks() -> AsyncIterator[str | None]:
        yield chunk

    wrapper: Final = TextCompletionStreamWrapper(completion_stream=chunks(), model="test-model")

    with pytest.raises(Exception, match=r"^$") as error:
        await anext(wrapper)

    assert not isinstance(error.value, StopAsyncIteration)


def test_sync_text_completion_stream_maps_upstream_failure_after_content() -> None:
    chunk: Final = ModelResponseStream(
        id="test-id",
        created=1234567890,
        model="test-model",
        choices=[StreamingChoices(index=0, delta=Delta(content="Hello"))],
    )

    def chunks() -> Iterator[ModelResponseStream]:
        yield chunk
        raise RuntimeError("upstream disconnected")

    wrapper: Final = TextCompletionStreamWrapper(
        completion_stream=chunks(), model="test-model", custom_llm_provider="openai"
    )
    assert next(wrapper).choices[0].text == "Hello"

    with pytest.raises(APIConnectionError, match="upstream disconnected") as error:
        next(wrapper)

    assert (error.value.model, error.value.llm_provider) == ("test-model", "openai")


def _stream_with_trailing_usage_chunk() -> tuple[ModelResponseStream, ...]:
    return (
        ModelResponseStream(choices=[StreamingChoices(index=0, delta=Delta(content="Hi"))]),
        ModelResponseStream(choices=[StreamingChoices(index=0, delta=Delta(), finish_reason="length")]),
        ModelResponseStream(
            choices=[StreamingChoices(index=0, delta=Delta())],
            usage=Usage(prompt_tokens=19, completion_tokens=5, total_tokens=24),
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("async_stream", (False, True))
@pytest.mark.parametrize(
    ("usage_injected_by_proxy", "expected_frames", "expected_usage"),
    (
        (True, [("Hi", None), (None, "length")], [None, None]),
        (False, [("Hi", None), (None, "length"), (None, None)], [None, None, (19, 5)]),
    ),
)
async def test_text_completion_stream_hides_only_the_usage_chunk_the_proxy_asked_for(
    async_stream: bool,
    usage_injected_by_proxy: bool,
    expected_frames: list[tuple[str | None, str | None]],
    expected_usage: list[tuple[int, int] | None],
) -> None:
    async def chunks() -> AsyncIterator[ModelResponseStream]:
        for chunk in _stream_with_trailing_usage_chunk():
            yield chunk

    wrapper: Final = TextCompletionStreamWrapper(
        completion_stream=chunks() if async_stream else iter(_stream_with_trailing_usage_chunk()),
        model="gpt-5.4-mini",
        stream_options={"include_usage": True},
        usage_injected_by_proxy=usage_injected_by_proxy,
    )
    frames: Final = [frame async for frame in wrapper] if async_stream else list(wrapper)

    assert [(frame.choices[0].text, frame.choices[0].finish_reason) for frame in frames] == expected_frames
    assert [
        None if frame.usage is None else (frame.usage.prompt_tokens, frame.usage.completion_tokens) for frame in frames
    ] == expected_usage


def test_text_completion_stream_keeps_a_finish_frame_but_hides_the_usage_the_proxy_asked_for() -> None:
    finish_with_usage: Final = ModelResponseStream(
        choices=[StreamingChoices(index=0, delta=Delta(), finish_reason="stop")],
        usage=Usage(prompt_tokens=27, completion_tokens=5, total_tokens=32),
    )
    wrapper: Final = TextCompletionStreamWrapper(
        completion_stream=iter((finish_with_usage,)),
        model="claude-haiku-4-5",
        stream_options={"include_usage": True},
        usage_injected_by_proxy=True,
    )

    frames: Final = list(wrapper)

    assert [(frame.choices[0].finish_reason, frame.usage) for frame in frames] == [("stop", None)]


@pytest.mark.asyncio
@pytest.mark.parametrize("async_stream", (False, True))
async def test_text_completion_stream_hides_the_empty_prompt_filter_frame_when_the_proxy_asked_for_usage(
    async_stream: bool,
) -> None:
    prompt_filter_frame: Final = ModelResponseStream(id="", choices=[])
    upstream: Final = (prompt_filter_frame, *_stream_with_trailing_usage_chunk())

    async def chunks() -> AsyncIterator[ModelResponseStream]:
        for chunk in upstream:
            yield chunk

    wrapper: Final = TextCompletionStreamWrapper(
        completion_stream=chunks() if async_stream else iter(upstream),
        model="gpt-5.4-mini",
        stream_options={"include_usage": True},
        usage_injected_by_proxy=True,
    )
    frames: Final = [frame async for frame in wrapper] if async_stream else list(wrapper)

    assert [(frame.choices[0].text, frame.choices[0].finish_reason) for frame in frames] == [
        ("Hi", None),
        (None, "length"),
    ]
