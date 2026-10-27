from collections.abc import AsyncIterator, Iterator
from typing import Final

import pytest

from litellm.exceptions import APIConnectionError
from litellm.text_completions.streaming_iterator import TextCompletionStreamWrapper
from litellm.types.utils import Choices, Delta, ModelResponseStream, StreamingChoices, Usage


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
