"""The streams a Rust-routed attempt hands back, read until the attempt has committed.

An attempt's stream is read until its first content (or, for the Responses API, its first
event past the opening lifecycle events) before the attempt reports success, so a failure
before then can still go to another deployment. What was read replays ahead of the rest.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Iterator, Sequence
from typing import Final, Literal, cast

from litellm.exceptions import MidStreamFallbackError
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.router_backends.python_router import (
    FallbackAwareStreamWrapper,
    PythonRouter,
    _responses_stream_holds_event,  # pyright: ignore[reportPrivateUsage]  # the hold rule Python's Responses wrapper applies
    _stream_chunks_have_generated_content,  # pyright: ignore[reportPrivateUsage]  # the content rule Python's stream wrapper applies
)
from litellm.router_utils.responses_fallback_stream import FallbackResponsesStreamWrapper
from litellm.types.llms.openai import ResponseAPIUsage
from litellm.types.utils import ModelResponseStream

StreamFailureKind = Literal["before_content", "terminal"]


class StreamFailed(Exception):
    """A stream that failed before it committed. `partial_usage` is what the failed Responses
    stream had used, which Python adds to whatever falls back from it."""

    def __init__(
        self, error: BaseException, kind: StreamFailureKind, partial_usage: ResponseAPIUsage | None = None
    ) -> None:
        super().__init__(str(error))
        self.error: Final = error
        self.kind: Final = kind
        self.partial_usage: Final = partial_usage


def unwrapped(error: BaseException) -> BaseException:
    """The provider error behind a mid-stream fallback error, which Python's stream wrappers raise in its place."""
    if isinstance(error, MidStreamFallbackError) and error.original_exception is not None:
        return error.original_exception
    return error


class ReplayStream(FallbackAwareStreamWrapper):
    """A chat attempt's stream, replaying the chunks read while waiting for its first content.

    After content has reached the caller the stream cannot fall back, so a mid-stream
    fallback error surfaces as the error that caused it, as the Python router's wrapper does.
    """

    def __init__(self, inner: CustomStreamWrapper, buffered: Sequence[ModelResponseStream]) -> None:
        super().__init__(  # pyright: ignore[reportUnknownMemberType]  # CustomStreamWrapper's untyped constructor
            completion_stream=cast(object, inner.completion_stream),  # cast-ok: untyped attribute
            model=cast(object, inner.model),  # cast-ok: untyped attribute
            custom_llm_provider=inner.custom_llm_provider,
            logging_obj=inner.logging_obj,
            _response_headers=getattr(inner, "_response_headers", None),
        )
        self._inner: Final = inner
        self._replay: Final = iter(tuple(buffered))
        self.chunks = cast(list[ModelResponseStream], inner.chunks)  # cast-ok: untyped attribute
        self._hidden_params = dict(cast(dict[str, object], inner._hidden_params))  # pyright: ignore[reportPrivateUsage]  # cast-ok: the stream's own untyped metadata

    def __aiter__(self) -> ReplayStream:
        return self

    async def __anext__(self) -> ModelResponseStream:
        buffered: Final = next(self._replay, None)
        if buffered is not None:
            return buffered
        try:
            return await self._inner.__anext__()
        except MidStreamFallbackError as error:
            raise unwrapped(error) from error

    def __iter__(self) -> Iterator[ModelResponseStream]:
        return self

    def __next__(self) -> ModelResponseStream:
        buffered: Final = next(self._replay, None)
        if buffered is not None:
            return buffered
        try:
            return self._inner.__next__()
        except MidStreamFallbackError as error:
            raise unwrapped(error) from error

    async def aclose(self) -> None:
        await self._inner.aclose()


async def chat_until_content(stream: CustomStreamWrapper) -> ReplayStream:
    buffered: Final[list[ModelResponseStream]] = []  # mutable-ok: chunks read before the first content
    try:
        async for chunk in stream:
            buffered.append(chunk)
            if _stream_chunks_have_generated_content([chunk]):
                break
    except MidStreamFallbackError as error:
        raise StreamFailed(error, "before_content") from error
    except Exception as error:
        raise StreamFailed(error, "terminal") from error
    return ReplayStream(stream, buffered)


def chat_until_content_sync(stream: CustomStreamWrapper) -> ReplayStream:
    buffered: Final[list[ModelResponseStream]] = []  # mutable-ok: chunks read before the first content
    try:
        for chunk in stream:
            buffered.append(chunk)
            if _stream_chunks_have_generated_content([chunk]):
                break
    except MidStreamFallbackError as error:
        raise StreamFailed(error, "before_content") from error
    except Exception as error:
        raise StreamFailed(error, "terminal") from error
    return ReplayStream(stream, buffered)


ResponsesFallback = Callable[[MidStreamFallbackError], Awaitable[object]]


async def responses_until_output(
    stream: BaseResponsesAPIStreamingIterator,
    fallback: ResponsesFallback,
    prior_usage: Sequence[ResponseAPIUsage],
) -> ResponsesReplay:
    """Holds the opening lifecycle events back until the stream's first output, as Python's
    Responses wrapper does, so a failure before then falls back without announcing a response."""
    held: tuple[object, ...] = ()  # rebind-ok: grows until the first output
    try:
        async for item in _items(stream):
            held = (*held, item)
            if not _responses_stream_holds_event(item, len(held) - 1):
                break
    except MidStreamFallbackError as error:
        raise StreamFailed(
            error,
            "before_content",
            PythonRouter._extract_partial_responses_usage(stream),  # pyright: ignore[reportPrivateUsage]  # Python's partial-usage reader
        ) from error
    except Exception as error:
        raise StreamFailed(error, "terminal") from error
    return ResponsesReplay(held, stream, fallback, prior_usage)


class ResponsesReplay(FallbackResponsesStreamWrapper):
    """A Responses attempt's stream, replaying what was held. After output a mid-stream fallback
    error still falls back through `fallback`, with the text generated so far to continue from."""

    def __init__(
        self,
        held: Sequence[object],
        source: BaseResponsesAPIStreamingIterator,
        fallback: ResponsesFallback,
        prior_usage: Sequence[ResponseAPIUsage],
    ) -> None:
        super().__init__(self._replay(held, source, fallback, prior_usage), source)

    async def _replay(
        self,
        held: Sequence[object],
        source: BaseResponsesAPIStreamingIterator,
        fallback: ResponsesFallback,
        prior_usage: Sequence[ResponseAPIUsage],
    ) -> AsyncGenerator[object, None]:
        for item in held:
            yield _with_usage(item, prior_usage)
        try:
            async for item in _items(source):
                yield _with_usage(item, prior_usage)
        except MidStreamFallbackError as error:
            partial: Final = PythonRouter._extract_partial_responses_usage(source)  # pyright: ignore[reportPrivateUsage]  # Python's partial-usage reader
            usage: Final = (*prior_usage, partial) if partial is not None else tuple(prior_usage)
            try:
                response: Final = await fallback(error)
            except Exception as fallback_error:
                raise unwrapped(fallback_error) from fallback_error
            prepared: Final = self.adopt_fallback_headers(response)
            if not hasattr(response, "__aiter__"):
                yield response
                return
            async for item in cast(AsyncIterator[object], response):  # cast-ok: checked above
                PythonRouter._apply_fallback_hidden_params_to_item(item, prepared)  # pyright: ignore[reportPrivateUsage]  # as Python's wrapper
                yield _with_usage(item, usage)
        finally:
            close: Final[object] = getattr(source, "aclose", None)
            if callable(close):
                await cast(Callable[[], Awaitable[object]], close)()  # cast-ok: an async iterator's aclose


def _items(stream: BaseResponsesAPIStreamingIterator) -> AsyncIterator[object]:
    return cast(AsyncIterator[object], stream)  # cast-ok: every Responses stream iterator is async-iterable


def _with_usage(item: object, usage: Sequence[ResponseAPIUsage]) -> object:
    for partial in usage:
        PythonRouter._combine_responses_fallback_usage(item, partial)  # pyright: ignore[reportPrivateUsage, reportArgumentType]  # Python's usage merge, a no-op on events without usage
    return item
