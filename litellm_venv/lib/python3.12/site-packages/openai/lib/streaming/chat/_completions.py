from __future__ import annotations

import inspect
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Callable,
    Iterable,
    Awaitable,
    AsyncIterator,
    cast,
)
from typing_extensions import Self, Iterator, assert_never

from jiter import from_json

from ._types import (
    ParsedChoiceSnapshot,
    ParsedChatCompletionSnapshot,
    ParsedChatCompletionMessageSnapshot,
)
from ._events import (
    ChunkEvent,
    ContentDoneEvent,
    RefusalDoneEvent,
    ContentDeltaEvent,
    RefusalDeltaEvent,
    LogprobsContentDoneEvent,
    LogprobsRefusalDoneEvent,
    ChatCompletionStreamEvent,
    LogprobsContentDeltaEvent,
    LogprobsRefusalDeltaEvent,
    FunctionToolCallArgumentsDoneEvent,
    FunctionToolCallArgumentsDeltaEvent,
)
from .._deltas import accumulate_delta
from ...._types import NOT_GIVEN, IncEx, NotGiven
from ...._utils import is_given, consume_sync_iterator, consume_async_iterator
from ...._compat import model_dump
from ...._models import build, construct_type
from ..._parsing import (
    ResponseFormatT,
    has_parseable_input,
    maybe_parse_content,
    parse_chat_completion,
    get_input_tool_by_name,
    solve_response_format_t,
    parse_function_tool_arguments,
)
from ...._streaming import Stream, AsyncStream
from ....types.chat import (
    ChatCompletionChunk,
    ParsedChatCompletion,
    ChatCompletionToolUnionParam,
)
from ...._exceptions import LengthFinishReasonError, ContentFilterFinishReasonError
from ....types.chat.chat_completion import ChoiceLogprobs
from ....types.chat.chat_completion_chunk import Choice as ChoiceChunk
from ....types.chat.completion_create_params import (
    ResponseFormat as ResponseFormatParam,
)


class ChatCompletionStream(Generic[ResponseFormatT]):
    """Wrapper over the Chat Completions streaming API that adds helpful
    events such as `content.done`, supports automatically parsing
    responses & tool calls and accumulates a `ChatCompletion` object
    from each individual chunk.

    https://platform.openai.com/docs/api-reference/streaming
    """

    def __init__(
        self,
        *,
        raw_stream: Stream[ChatCompletionChunk],
        response_format: type[ResponseFormatT] | ResponseFormatParam | NotGiven,
        input_tools: Iterable[ChatCompletionToolUnionParam] | NotGiven,
    ) -> None:
        self._raw_stream = raw_stream
        self._response = raw_stream.response
        self._iterator = self.__stream__()
        self._state = ChatCompletionStreamState(
            response_format=response_format, input_tools=input_tools
        )

    def __next__(self) -> ChatCompletionStreamEvent[ResponseFormatT]:
        return self._iterator.__next__()

    def __iter__(self) -> Iterator[ChatCompletionStreamEvent[ResponseFormatT]]:
        for item in self._iterator:
            yield item

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """
        Close the response and release the connection.

        Automatically called if the response body is read to completion.
        """
        self._response.close()

    def get_final_completion(self) -> ParsedChatCompletion[ResponseFormatT]:
        """Waits until the stream has been read to completion and returns
        the accumulated `ParsedChatCompletion` object.

        If you passed a class type to `.stream()`, the `completion.choices[0].message.parsed`
        property will be the content deserialised into that class, if there was any content returned
        by the API.
        """
        self.until_done()
        return self._state.get_final_completion()

    def until_done(self) -> Self:
        """Blocks until the stream has been consumed."""
        consume_sync_iterator(self)
        return self

    @property
    def current_completion_snapshot(self) -> ParsedChatCompletionSnapshot:
        return self._state.current_completion_snapshot

    def __stream__(self) -> Iterator[ChatCompletionStreamEvent[ResponseFormatT]]:
        for sse_event in self._raw_stream:
            if not _is_valid_chat_completion_chunk_weak(sse_event):
                continue
            events_to_fire = self._state.handle_chunk(sse_event)
            for event in events_to_fire:
                yield event


class ChatCompletionStreamManager(Generic[ResponseFormatT]):
    """Context manager over a `ChatCompletionStream` that is returned by `.stream()`.

    This context manager ensures the response cannot be leaked if you don't read
    the stream to completion.

    Usage:
    ```py
    with client.chat.completions.stream(...) as stream:
        for event in stream:
            ...
    ```
    """

    def __init__(
        self,
        api_request: Callable[[], Stream[ChatCompletionChunk]],
        *,
        response_format: type[ResponseFormatT] | ResponseFormatParam | NotGiven,
        input_tools: Iterable[ChatCompletionToolUnionParam] | NotGiven,
    ) -> None:
        self.__stream: ChatCompletionStream[ResponseFormatT] | None = None
        self.__api_request = api_request
        self.__response_format = response_format
        self.__input_tools = input_tools

    def __enter__(self) -> ChatCompletionStream[ResponseFormatT]:
        raw_stream = self.__api_request()

        self.__stream = ChatCompletionStream(
            raw_stream=raw_stream,
            response_format=self.__response_format,
            input_tools=self.__input_tools,
        )

        return self.__stream

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self.__stream is not None:
            self.__stream.close()


class AsyncChatCompletionStream(Generic[ResponseFormatT]):
    """Wrapper over the Chat Completions streaming API that adds helpful
    events such as `content.done`, supports automatically parsing
    responses & tool calls and accumulates a `ChatCompletion` object
    from each individual chunk.

    https://platform.openai.com/docs/api-reference/streaming
    """

    def __init__(
        self,
        *,
        raw_stream: AsyncStream[ChatCompletionChunk],
        response_format: type[ResponseFormatT] | ResponseFormatParam | NotGiven,
        input_tools: Iterable[ChatCompletionToolUnionParam] | NotGiven,
    ) -> None:
        self._raw_stream = raw_stream
        self._response = raw_stream.response
        self._iterator = self.__stream__()
        self._state = ChatCompletionStreamState(
            response_format=response_format, input_tools=input_tools
        )

    async def __anext__(self) -> ChatCompletionStreamEvent[ResponseFormatT]:
        return await self._iterator.__anext__()

    async def __aiter__(
        self,
    ) -> AsyncIterator[ChatCompletionStreamEvent[ResponseFormatT]]:
        async for item in self._iterator:
            yield item

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        """
        Close the response and release the connection.

        Automatically called if the response body is read to completion.
        """
        await self._response.aclose()

    async def get_final_completion(self) -> ParsedChatCompletion[ResponseFormatT]:
        """Waits until the stream has been read to completion and returns
        the accumulated `ParsedChatCompletion` object.

        If you passed a class type to `.stream()`, the `completion.choices[0].message.parsed`
        property will be the content deserialised into that class, if there was any content returned
        by the API.
        """
        await self.until_done()
        return self._state.get_final_completion()

    async def until_done(self) -> Self:
        """Blocks until the stream has been consumed."""
        await consume_async_iterator(self)
        return self

    @property
    def current_completion_snapshot(self) -> ParsedChatCompletionSnapshot:
        return self._state.current_completion_snapshot

    async def __stream__(
        self,
    ) -> AsyncIterator[ChatCompletionStreamEvent[ResponseFormatT]]:
        async for sse_event in self._raw_stream:
            if not _is_valid_chat_completion_chunk_weak(sse_event):
                continue
            events_to_fire = self._state.handle_chunk(sse_event)
            for event in events_to_fire:
                yield event


class AsyncChatCompletionStreamManager(Generic[ResponseFormatT]):
    """Context manager over a `AsyncChatCompletionStream` that is returned by `.stream()`.

    This context manager ensures the response cannot be leaked if you don't read
    the stream to completion.

    Usage:
    ```py
    async with client.chat.completions.stream(...) as stream:
        for event in stream:
            ...
    ```
    """

    def __init__(
        self,
        api_request: Awaitable[AsyncStream[ChatCompletionChunk]],
        *,
        response_format: type[ResponseFormatT] | ResponseFormatParam | NotGiven,
        input_tools: Iterable[ChatCompletionToolUnionParam] | NotGiven,
    ) -> None:
        self.__stream: AsyncChatCompletionStream[ResponseFormatT] | None = None
        self.__api_request = api_request
        self.__response_format = response_format
        self.__input_tools = input_tools

    async def __aenter__(self) -> AsyncChatCompletionStream[ResponseFormatT]:
        raw_stream = await self.__api_request

        self.__stream = AsyncChatCompletionStream(
            raw_stream=raw_stream,
            response_format=self.__response_format,
            input_tools=self.__input_tools,
        )

        return self.__stream

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self.__stream is not None:
            await self.__stream.close()


class ChatCompletionStreamState(Generic[ResponseFormatT]):
    """Helper class for manually accumulating `ChatCompletionChunk`s into a final `ChatCompletion` object.

    This is useful in cases where you can't always use the `.stream()` method, e.g.

    ```py
    from openai.lib.streaming.chat import ChatCompletionStreamState

    state = ChatCompletionStreamState()

    stream = client.chat.completions.create(..., stream=True)
    for chunk in response:
        state.handle_chunk(chunk)

        # can also access the accumulated `ChatCompletion` mid-stream
        state.current_completion_snapshot

    print(state.get_final_completion())
    ```
    """

    def __init__(
        self,
        *,
        input_tools: Iterable[ChatCompletionToolUnionParam] | NotGiven = NOT_GIVEN,
        response_format: type[ResponseFormatT]
        | ResponseFormatParam
        | NotGiven = NOT_GIVEN,
    ) -> None:
        self.__current_completion_snapshot: ParsedChatCompletionSnapshot | None = None
        self.__choice_event_states: list[ChoiceEventState] = []

        self._input_tools = (
            [tool for tool in input_tools] if is_given(input_tools) else []
        )
        self._response_format = response_format
        self._rich_response_format: type | NotGiven = (
            response_format if inspect.isclass(response_format) else NOT_GIVEN
        )

    def get_final_completion(self) -> ParsedChatCompletion[ResponseFormatT]:
        """Parse the final completion object.

        Note this does not provide any guarantees that the stream has actually finished, you must
        only call this method when the stream is finished.
        """
        return parse_chat_completion(
            chat_completion=self.current_completion_snapshot,
            response_format=self._rich_response_format,
            input_tools=self._input_tools,
        )

    @property
    def current_completion_snapshot(self) -> ParsedChatCompletionSnapshot:
        assert self.__current_completion_snapshot is not None
        return self.__current_completion_snapshot

    def handle_chunk(
        self, chunk: ChatCompletionChunk
    ) -> Iterable[ChatCompletionStreamEvent[ResponseFormatT]]:
        """Accumulate a new chunk into the snapshot and returns an iterable of events to yield."""
        self.__current_completion_snapshot = self._accumulate_chunk(chunk)

        return self._build_events(
            chunk=chunk,
            completion_snapshot=self.__current_completion_snapshot,
        )

    def _get_choice_state(self, choice: ChoiceChunk) -> ChoiceEventState:
        try:
            return self.__choice_event_states[choice.index]
        except IndexError:
            choice_state = ChoiceEventState(input_tools=self._input_tools)
            self.__choice_event_states.append(choice_state)
            return choice_state

    def _accumulate_chunk(
        self, chunk: ChatCompletionChunk
    ) -> ParsedChatCompletionSnapshot:
        completion_snapshot = self.__current_completion_snapshot

        if completion_snapshot is None:
            return _convert_initial_chunk_into_snapshot(chunk)

        for choice in chunk.choices:
            try:
                choice_snapshot = completion_snapshot.choices[choice.index]
                previous_tool_calls = choice_snapshot.message.tool_calls or []

                choice_snapshot.message = cast(
                    ParsedChatCompletionMessageSnapshot,
                    construct_type(
                        type_=ParsedChatCompletionMessageSnapshot,
                        value=accumulate_delta(
                            cast(
                                "dict[object, object]",
                                model_dump(
                                    choice_snapshot.message,
                                    # we don't want to serialise / deserialise our custom properties
                                    # as they won't appear in the delta and we don't want to have to
                                    # continuosly reparse the content
                                    exclude=cast(
                                        # cast required as mypy isn't smart enough to infer `True` here to `Literal[True]`
                                        IncEx,
                                        {
                                            "parsed": True,
                                            "tool_calls": {
                                                idx: {
                                                    "function": {
                                                        "parsed_arguments": True
                                                    }
                                                }
                                                for idx, _ in enumerate(
                                                    choice_snapshot.message.tool_calls
                                                    or []
                                                )
                                            },
                                        },
                                    ),
                                ),
                            ),
                            cast("dict[object, object]", choice.delta.to_dict()),
                        ),
                    ),
                )

                # ensure tools that have already been parsed are added back into the newly
                # constructed message snapshot
                for tool_index, prev_tool in enumerate(previous_tool_calls):
                    new_tool = (choice_snapshot.message.tool_calls or [])[tool_index]

                    if prev_tool.type == "function":
                        assert new_tool.type == "function"
                        new_tool.function.parsed_arguments = (
                            prev_tool.function.parsed_arguments
                        )
                    elif TYPE_CHECKING:  # type: ignore[unreachable]
                        assert_never(prev_tool)
            except IndexError:
                choice_snapshot = cast(
                    ParsedChoiceSnapshot,
                    construct_type(
                        type_=ParsedChoiceSnapshot,
                        value={
                            **choice.model_dump(exclude_unset=True, exclude={"delta"}),
                            "message": choice.delta.to_dict(),
                        },
                    ),
                )
                completion_snapshot.choices.append(choice_snapshot)

            if choice.finish_reason:
                choice_snapshot.finish_reason = choice.finish_reason

                if has_parseable_input(
                    response_format=self._response_format, input_tools=self._input_tools
                ):
                    if choice.finish_reason == "length":
                        # at the time of writing, `.usage` will always be `None` but
                        # we include it here in case that is changed in the future
                        raise LengthFinishReasonError(completion=completion_snapshot)

                    if choice.finish_reason == "content_filter":
                        raise ContentFilterFinishReasonError()

            if (
                choice_snapshot.message.content
                and not choice_snapshot.message.refusal
                and is_given(self._rich_response_format)
                # partial parsing fails on white-space
                and choice_snapshot.message.content.lstrip()
            ):
                choice_snapshot.message.parsed = from_json(
                    bytes(choice_snapshot.message.content, "utf-8"),
                    partial_mode=True,
                )

            for tool_call_chunk in choice.delta.tool_calls or []:
                tool_call_snapshot = (choice_snapshot.message.tool_calls or [])[
                    tool_call_chunk.index
                ]

                if tool_call_snapshot.type == "function":
                    input_tool = get_input_tool_by_name(
                        input_tools=self._input_tools,
                        name=tool_call_snapshot.function.name,
                    )

                    if (
                        input_tool
                        and input_tool.get("function", {}).get("strict")
                        and tool_call_snapshot.function.arguments
                    ):
                        tool_call_snapshot.function.parsed_arguments = from_json(
                            bytes(tool_call_snapshot.function.arguments, "utf-8"),
                            partial_mode=True,
                        )
                elif TYPE_CHECKING:  # type: ignore[unreachable]
                    assert_never(tool_call_snapshot)

            if choice.logprobs is not None:
                if choice_snapshot.logprobs is None:
                    choice_snapshot.logprobs = build(
                        ChoiceLogprobs,
                        content=choice.logprobs.content,
                        refusal=choice.logprobs.refusal,
                    )
                else:
                    if choice.logprobs.content:
                        if choice_snapshot.logprobs.content is None:
                            choice_snapshot.logprobs.content = []

                        choice_snapshot.logprobs.content.extend(choice.logprobs.content)

                    if choice.logprobs.refusal:
                        if choice_snapshot.logprobs.refusal is None:
                            choice_snapshot.logprobs.refusal = []

                        choice_snapshot.logprobs.refusal.extend(choice.logprobs.refusal)

        completion_snapshot.usage = chunk.usage
        completion_snapshot.system_fingerprint = chunk.system_fingerprint

        return completion_snapshot

    def _build_events(
        self,
        *,
        chunk: ChatCompletionChunk,
        completion_snapshot: ParsedChatCompletionSnapshot,
    ) -> list[ChatCompletionStreamEvent[ResponseFormatT]]:
        events_to_fire: list[ChatCompletionStreamEvent[ResponseFormatT]] = []

        events_to_fire.append(
            build(ChunkEvent, type="chunk", chunk=chunk, snapshot=completion_snapshot),
        )

        for choice in chunk.choices:
            choice_state = self._get_choice_state(choice)
            choice_snapshot = completion_snapshot.choices[choice.index]

            if (
                choice.delta.content is not None
                and choice_snapshot.message.content is not None
            ):
                events_to_fire.append(
                    build(
                        ContentDeltaEvent,
                        type="content.delta",
                        delta=choice.delta.content,
                        snapshot=choice_snapshot.message.content,
                        parsed=choice_snapshot.message.parsed,
                    )
                )

            if (
                choice.delta.refusal is not None
                and choice_snapshot.message.refusal is not None
            ):
                events_to_fire.append(
                    build(
                        RefusalDeltaEvent,
                        type="refusal.delta",
                        delta=choice.delta.refusal,
                        snapshot=choice_snapshot.message.refusal,
                    )
                )

            if choice.delta.tool_calls:
                tool_calls = choice_snapshot.message.tool_calls
                assert tool_calls is not None

                for tool_call_delta in choice.delta.tool_calls:
                    tool_call = tool_calls[tool_call_delta.index]

                    if tool_call.type == "function":
                        assert tool_call_delta.function is not None
                        events_to_fire.append(
                            build(
                                FunctionToolCallArgumentsDeltaEvent,
                                type="tool_calls.function.arguments.delta",
                                name=tool_call.function.name,
                                index=tool_call_delta.index,
                                arguments=tool_call.function.arguments,
                                parsed_arguments=tool_call.function.parsed_arguments,
                                arguments_delta=tool_call_delta.function.arguments
                                or "",
                            )
                        )
                    elif TYPE_CHECKING:  # type: ignore[unreachable]
                        assert_never(tool_call)

            if choice.logprobs is not None and choice_snapshot.logprobs is not None:
                if choice.logprobs.content and choice_snapshot.logprobs.content:
                    events_to_fire.append(
                        build(
                            LogprobsContentDeltaEvent,
                            type="logprobs.content.delta",
                            content=choice.logprobs.content,
                            snapshot=choice_snapshot.logprobs.content,
                        ),
                    )

                if choice.logprobs.refusal and choice_snapshot.logprobs.refusal:
                    events_to_fire.append(
                        build(
                            LogprobsRefusalDeltaEvent,
                            type="logprobs.refusal.delta",
                            refusal=choice.logprobs.refusal,
                            snapshot=choice_snapshot.logprobs.refusal,
                        ),
                    )

            events_to_fire.extend(
                choice_state.get_done_events(
                    choice_chunk=choice,
                    choice_snapshot=choice_snapshot,
                    response_format=self._response_format,
                )
            )

        return events_to_fire


class ChoiceEventState:
    def __init__(self, *, input_tools: list[ChatCompletionToolUnionParam]) -> None:
        self._input_tools = input_tools

        self._content_done = False
        self._refusal_done = False
        self._logprobs_content_done = False
        self._logprobs_refusal_done = False
        self._done_tool_calls: set[int] = set()
        self.__current_tool_call_index: int | None = None

    def get_done_events(
        self,
        *,
        choice_chunk: ChoiceChunk,
        choice_snapshot: ParsedChoiceSnapshot,
        response_format: type[ResponseFormatT] | ResponseFormatParam | NotGiven,
    ) -> list[ChatCompletionStreamEvent[ResponseFormatT]]:
        events_to_fire: list[ChatCompletionStreamEvent[ResponseFormatT]] = []

        if choice_snapshot.finish_reason:
            events_to_fire.extend(
                self._content_done_events(
                    choice_snapshot=choice_snapshot, response_format=response_format
                )
            )

            if (
                self.__current_tool_call_index is not None
                and self.__current_tool_call_index not in self._done_tool_calls
            ):
                self._add_tool_done_event(
                    events_to_fire=events_to_fire,
                    choice_snapshot=choice_snapshot,
                    tool_index=self.__current_tool_call_index,
                )

        for tool_call in choice_chunk.delta.tool_calls or []:
            if self.__current_tool_call_index != tool_call.index:
                events_to_fire.extend(
                    self._content_done_events(
                        choice_snapshot=choice_snapshot, response_format=response_format
                    )
                )

                if self.__current_tool_call_index is not None:
                    self._add_tool_done_event(
                        events_to_fire=events_to_fire,
                        choice_snapshot=choice_snapshot,
                        tool_index=self.__current_tool_call_index,
                    )

            self.__current_tool_call_index = tool_call.index

        return events_to_fire

    def _content_done_events(
        self,
        *,
        choice_snapshot: ParsedChoiceSnapshot,
        response_format: type[ResponseFormatT] | ResponseFormatParam | NotGiven,
    ) -> list[ChatCompletionStreamEvent[ResponseFormatT]]:
        events_to_fire: list[ChatCompletionStreamEvent[ResponseFormatT]] = []

        if choice_snapshot.message.content and not self._content_done:
            self._content_done = True

            parsed = maybe_parse_content(
                response_format=response_format,
                message=choice_snapshot.message,
            )

            # update the parsed content to now use the richer `response_format`
            # as opposed to the raw JSON-parsed object as the content is now
            # complete and can be fully validated.
            choice_snapshot.message.parsed = parsed

            events_to_fire.append(
                build(
                    # we do this dance so that when the `ContentDoneEvent` instance
                    # is printed at runtime the class name will include the solved
                    # type variable, e.g. `ContentDoneEvent[MyModelType]`
                    cast(  # pyright: ignore[reportUnnecessaryCast]
                        "type[ContentDoneEvent[ResponseFormatT]]",
                        cast(Any, ContentDoneEvent)[
                            solve_response_format_t(response_format)
                        ],
                    ),
                    type="content.done",
                    content=choice_snapshot.message.content,
                    parsed=parsed,
                ),
            )

        if choice_snapshot.message.refusal is not None and not self._refusal_done:
            self._refusal_done = True
            events_to_fire.append(
                build(
                    RefusalDoneEvent,
                    type="refusal.done",
                    refusal=choice_snapshot.message.refusal,
                ),
            )

        if (
            choice_snapshot.logprobs is not None
            and choice_snapshot.logprobs.content is not None
            and not self._logprobs_content_done
        ):
            self._logprobs_content_done = True
            events_to_fire.append(
                build(
                    LogprobsContentDoneEvent,
                    type="logprobs.content.done",
                    content=choice_snapshot.logprobs.content,
                ),
            )

        if (
            choice_snapshot.logprobs is not None
            and choice_snapshot.logprobs.refusal is not None
            and not self._logprobs_refusal_done
        ):
            self._logprobs_refusal_done = True
            events_to_fire.append(
                build(
                    LogprobsRefusalDoneEvent,
                    type="logprobs.refusal.done",
                    refusal=choice_snapshot.logprobs.refusal,
                ),
            )

        return events_to_fire

    def _add_tool_done_event(
        self,
        *,
        events_to_fire: list[ChatCompletionStreamEvent[ResponseFormatT]],
        choice_snapshot: ParsedChoiceSnapshot,
        tool_index: int,
    ) -> None:
        if tool_index in self._done_tool_calls:
            return

        self._done_tool_calls.add(tool_index)

        assert choice_snapshot.message.tool_calls is not None
        tool_call_snapshot = choice_snapshot.message.tool_calls[tool_index]

        if tool_call_snapshot.type == "function":
            parsed_arguments = parse_function_tool_arguments(
                input_tools=self._input_tools, function=tool_call_snapshot.function
            )

            # update the parsed content to potentially use a richer type
            # as opposed to the raw JSON-parsed object as the content is now
            # complete and can be fully validated.
            tool_call_snapshot.function.parsed_arguments = parsed_arguments

            events_to_fire.append(
                build(
                    FunctionToolCallArgumentsDoneEvent,
                    type="tool_calls.function.arguments.done",
                    index=tool_index,
                    name=tool_call_snapshot.function.name,
                    arguments=tool_call_snapshot.function.arguments,
                    parsed_arguments=parsed_arguments,
                )
            )
        elif TYPE_CHECKING:  # type: ignore[unreachable]
            assert_never(tool_call_snapshot)


def _convert_initial_chunk_into_snapshot(
    chunk: ChatCompletionChunk,
) -> ParsedChatCompletionSnapshot:
    data = chunk.to_dict()
    choices = cast("list[object]", data["choices"])

    for choice in chunk.choices:
        choices[choice.index] = {
            **choice.model_dump(exclude_unset=True, exclude={"delta"}),
            "message": choice.delta.to_dict(),
        }

    return cast(
        ParsedChatCompletionSnapshot,
        construct_type(
            type_=ParsedChatCompletionSnapshot,
            value={
                "system_fingerprint": None,
                **data,
                "object": "chat.completion",
            },
        ),
    )


def _is_valid_chat_completion_chunk_weak(sse_event: ChatCompletionChunk) -> bool:
    # Although the _raw_stream is always supposed to contain only objects adhering to ChatCompletionChunk schema,
    # this is broken by the Azure OpenAI in case of Asynchronous Filter enabled.
    # An easy filter is to check for the "object" property:
    # - should be "chat.completion.chunk" for a ChatCompletionChunk;
    # - is an empty string for Asynchronous Filter events.
    return sse_event.object == "chat.completion.chunk"  # type: ignore # pylance reports this as a useless check
