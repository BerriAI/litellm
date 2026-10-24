import asyncio
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import Final, Literal, cast

import anyio
import httpx
import openai
import pytest
import respx
from pydantic import BaseModel, JsonValue

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.openai.common_utils import OpenAIError
from tests.test_litellm.streaming_contract_fixtures import (
    JSON_OBJECT,
    TOOL_ARGUMENTS,
    TOOL_ID,
    Case,
    Event,
    Mode,
    Options,
    Provider,
    Recorder,
    Surface,
    Wire,
    nonstream_response,
)


@dataclass(frozen=True, slots=True)
class Session:
    case: Case
    wire: Wire
    recorder: Recorder
    client: openai.OpenAI | openai.AsyncOpenAI | HTTPHandler | AsyncHTTPHandler

    def kwargs(self, stream: bool = True) -> dict[str, object]:
        return {
            "model": self.case.model,
            "api_key": "streaming-matrix-placeholder",
            "stream": stream,
            "client": self.client,
            "num_retries": 0,
            "max_tokens": 100,
            **(self.case.stream_options if stream else {}),
            **(
                {"reasoning_effort": "none"}
                if self.case.provider == "openai" and self.case.tool and self.case.response_model == "gpt-6-astra"
                else {}
            ),
            **(
                {
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "description": "Look up a city",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"city": {"type": "string"}, "count": {"type": "integer"}},
                                    "required": ["city", "count"],
                                },
                            },
                        }
                    ]
                }
                if self.case.tool
                else {}
            ),
            **(
                {
                    "mock_response": (
                        litellm.ModelResponse(**nonstream_response(self.case))
                        if self.case.tool or self.case.mock_payload == "response"
                        else self.case.text
                    )
                }
                if self.case.provider == "mock"
                else {}
            ),
            **(
                {
                    self.case.admission_key: {
                        "user_api_key_budget_reservation": (
                            {"input_tokens": self.case.admission}
                            if self.case.reservation in ("count", "null_count")
                            else (None if self.case.reservation == "null" else {})
                        )
                    }
                }
                if self.case.admission is not None or self.case.reservation != "count"
                else {}
            ),
        }

    async def call(self, stream: bool = True, prompt: str = "hello") -> object:
        match self.case.mode, self.case.surface:
            case "sync", "chat":
                return litellm.completion(messages=[{"role": "user", "content": prompt}], **self.kwargs(stream))
            case "async", "chat":
                return await litellm.acompletion(messages=[{"role": "user", "content": prompt}], **self.kwargs(stream))
            case "sync", "text":
                return litellm.text_completion(prompt=prompt, **self.kwargs(stream))
            case "async", "text":
                return await litellm.atext_completion(prompt=prompt, **self.kwargs(stream))


@contextmanager
def registered_callbacks(recorder: Recorder) -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(litellm, "callbacks", [recorder])
        for callback_list in (
            "success_callback",
            "failure_callback",
            "input_callback",
            "_async_success_callback",
            "_async_failure_callback",
            "_async_input_callback",
        ):
            patch.setattr(litellm, callback_list, [])
        yield


@contextmanager
def synchronous_session(case: Case) -> Iterator[Session]:
    wire: Final = Wire(case)
    recorder: Final = Recorder()
    with (
        registered_callbacks(recorder),
        httpx.Client(transport=httpx.MockTransport(wire.respond), trust_env=False) as client,
    ):
        sdk_client: Final = (
            openai.OpenAI(api_key="placeholder", http_client=client, max_retries=0)
            if case.provider in ("openai", "openai_text", "mock")
            else HTTPHandler(client=client)
        )
        try:
            yield Session(case, wire, recorder, sdk_client)
        except OpenAIError as error:
            if case.provider == "openai_text" and "'MockValSer' object is not an instance of 'SchemaSerializer'" in str(
                error
            ):
                pytest.xfail("STREAM-005: cold native text usage fails in OpenAI/Pydantic serialization")
            raise


@asynccontextmanager
async def session(case: Case) -> AsyncIterator[Session]:
    if case.mode == "sync":
        with synchronous_session(case) as run:
            yield run
        return
    wire: Final = Wire(case)
    recorder: Final = Recorder()
    with registered_callbacks(recorder):
        async with httpx.AsyncClient(transport=httpx.MockTransport(wire.respond), trust_env=False) as async_http_client:
            if case.provider in ("openai", "openai_text", "mock"):
                async_client: Final = openai.AsyncOpenAI(
                    api_key="placeholder", http_client=async_http_client, max_retries=0
                )
                try:
                    yield Session(case, wire, recorder, async_client)
                except OpenAIError as error:
                    if (
                        case.provider == "openai_text"
                        and "'MockValSer' object is not an instance of 'SchemaSerializer'" in str(error)
                    ):
                        pytest.xfail("STREAM-005: cold native text usage fails in OpenAI/Pydantic serialization")
                    raise
                return
            handler: Final = AsyncHTTPHandler()
            await handler.client.aclose()
            handler.client = async_http_client
            yield Session(case, wire, recorder, handler)


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(litellm, "telemetry", False)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    with respx.mock(assert_all_called=False, assert_all_mocked=True):
        yield


def snapshot(response: object) -> dict[str, JsonValue]:
    assert isinstance(response, BaseModel), type(response)
    return JSON_OBJECT.validate_json(response.model_dump_json())


async def drain(response: object, mode: Mode) -> tuple[dict[str, JsonValue], ...]:
    if mode == "sync":
        return tuple(snapshot(chunk) for chunk in cast(Iterator[object], response))
    return tuple([snapshot(chunk) async for chunk in cast(AsyncIterator[object], response)])


async def first_content(response: object, mode: Mode, surface: Surface, remaining: int = 5) -> dict[str, JsonValue]:
    assert remaining > 0, "no content before lifecycle interruption"
    chunk: Final = snapshot(
        next(cast(Iterator[object], response)) if mode == "sync" else await anext(cast(AsyncIterator[object], response))
    )
    assert all(choice.get("finish_reason") is None for choice in objects(chunk["choices"]))
    if text((chunk,), surface):
        return chunk
    return await first_content(response, mode, surface, remaining - 1)


async def final_events(recorder: Recorder) -> tuple[Event, ...]:
    assert await asyncio.to_thread(recorder.arrived.wait, 5), "missing final SDK callback"
    await asyncio.sleep(0.05)
    return recorder.events


def objects(value: JsonValue) -> tuple[dict[str, JsonValue], ...]:
    assert isinstance(value, list), value
    assert all(isinstance(item, dict) for item in value), value
    return cast(tuple[dict[str, JsonValue], ...], tuple(value))


def field(value: JsonValue, key: str) -> JsonValue:
    assert isinstance(value, dict), value
    return value.get(key)


def text(chunks: tuple[dict[str, JsonValue], ...], surface: Surface) -> str:
    return "".join(
        str(part)
        for chunk in chunks
        for choice in objects(chunk["choices"])
        if (part := choice.get("text") if surface == "text" else field(choice.get("delta"), "content")) is not None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ("openai", "anthropic", "gemini", "openai_text"))
@pytest.mark.parametrize("mode", ("sync", "async"))
@pytest.mark.parametrize("surface", ("chat", "text"))
@pytest.mark.parametrize("options", ("omitted", "none", "empty", "hidden", "visible"))
@pytest.mark.parametrize("prompt_tokens", (0, 1, 37), ids=("p0", "p1", "p37"))
@pytest.mark.parametrize("output_tokens", (0, 1, 17), ids=("c0", "c1", "c17"))
async def test_success(
    provider: Provider, mode: Mode, surface: Surface, options: Options, prompt_tokens: int, output_tokens: int
) -> None:
    case: Final = Case(
        provider=provider,
        mode=mode,
        surface=surface,
        options=options,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
    )
    async with session(case) as run:
        response: Final = await run.call()
        chunks: Final = await drain(response, mode)
        assert text(chunks, surface) == case.text
        assert len(run.wire.requests) == 1, run.wire.requests
        if provider in ("openai", "openai_text"):
            body: Final = run.wire.request_bodies[0]
            assert body["model"] == case.response_model and body["stream"] is True
            assert body.get("stream_options") == (
                {"include_usage": True}
                if provider == "openai" and options in ("omitted", "none")
                else case.stream_options.get("stream_options")
            )
        reasons: Final = tuple(
            choice["finish_reason"]
            for chunk in chunks
            for choice in objects(chunk["choices"])
            if choice.get("finish_reason") is not None
        )
        assert reasons == ("stop",), chunks
        visible_usage: Final = tuple(chunk["usage"] for chunk in chunks if chunk.get("usage") is not None)
        visible: Final = options == "visible" or (
            provider == "openai" and surface == "chat" and options in ("omitted", "none")
        )
        if visible:
            assert len(visible_usage) == 1, chunks
            assert chunks[-1].get("usage") == visible_usage[0], "usage must be consumed after content and termination"
        elif surface == "chat":
            assert visible_usage == (), chunks
        else:
            assert all(
                tuple(field(item, key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")) == (0, 0, 0)
                for item in visible_usage
            ), chunks
        recounted: Final = (provider == "openai" and options in ("empty", "hidden")) or (
            provider == "openai_text" and options != "visible"
        )
        expected: Final = (8, 2, 10) if recounted else (prompt_tokens, output_tokens, prompt_tokens + output_tokens)
        events: Final = await final_events(run.recorder)
        assert len(events) == 1, events
        assert events[0].outcome == "success" and events[0].mode == mode
        assert isinstance(events[0].response, dict), events
        assert_success_output(chunks, case, events[0])
        callback_usage: Final = tuple(
            field(events[0].response.get("usage"), key)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        )
        observed_usages: Final = (callback_usage,) + (
            tuple(
                tuple(field(item, key) for key in ("prompt_tokens", "completion_tokens", "total_tokens"))
                for item in visible_usage
            )
            if visible
            else ()
        )
        known_recounted: Final = (
            8 if prompt_tokens == 0 else prompt_tokens,
            2 if output_tokens == 0 else output_tokens,
        )
        allowed_usages: Final = (
            (expected, (*known_recounted, sum(known_recounted)))
            if (prompt_tokens == 0 or output_tokens == 0) and not recounted
            else (expected,)
        )
        assert all(usage in allowed_usages for usage in observed_usages), observed_usages
        if any(usage != expected for usage in observed_usages):
            pytest.xfail("STREAM-002: provider authoritative zero usage is recounted")


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ("openai", "anthropic", "gemini"))
@pytest.mark.parametrize("mode", ("sync", "async"))
@pytest.mark.parametrize("surface", ("chat", "text"))
async def test_midstream_failure(provider: Provider, mode: Mode, surface: Surface) -> None:
    case: Final = Case(
        provider=provider,
        mode=mode,
        surface=surface,
        fragments=("first ", "second ", "third ", "last"),
        lifecycle="failure",
    )
    async with session(case) as run:
        response: Final = await run.call()
        first: Final = await first_content(response, mode, surface)
        assert text((first,), surface) == "first "
        with pytest.raises(litellm.exceptions.MidStreamFallbackError, match="matrix provider disconnected"):
            await drain(response, mode)
        events: Final = await final_events(run.recorder)
        assert len(events) == 1, events
        assert events[0].outcome == "failure", events
        assert "matrix provider disconnected" in str(events[0].exception), events
        if mode == "async":
            partial: Final = snapshot(events[0].partial_usage)
            expected_partial: Final = {"openai": (8, 3, 11), "anthropic": (11, 2, 13), "gemini": (8, 4, 12)}[provider]
            assert (
                tuple(partial[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens"))
                == expected_partial
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ("openai", "anthropic", "gemini"))
@pytest.mark.parametrize("mode", ("sync", "async"))
async def test_early_close(provider: Provider, mode: Mode) -> None:
    case: Final = Case(provider=provider, mode=mode, lifecycle="close")
    async with session(case) as run:
        response: Final = await run.call()
        first: Final = await first_content(response, mode, "chat")
        assert text((first,), "chat") == case.fragments[0]
        from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

        assert isinstance(response, CustomStreamWrapper)
        await response.aclose()
        await asyncio.sleep(0.05)
        assert run.recorder.events == (), "closing an unfinished stream fabricated a final callback"
        if not run.wire.closed and (provider == "anthropic" or (provider == "gemini" and mode == "sync")):
            pytest.xfail("STREAM-001: explicit close does not reach this provider transport")
        assert run.wire.closed, "explicit close did not reach simulated transport"


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ("openai", "anthropic", "gemini"))
async def test_async_cancel(provider: Provider) -> None:
    case: Final = Case(
        provider=provider, mode="async", lifecycle="cancel", fragments=("first ", "second ", "third ", "last")
    )
    async with session(case) as run:
        response: Final = await run.call()
        first: Final = await first_content(response, "async", "chat")
        assert text((first,), "chat") == case.fragments[0]
        consumer: Final = asyncio.create_task(drain(response, "async"))
        await asyncio.wait_for(run.wire.waiting.wait(), timeout=5)
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer
        from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

        assert isinstance(response, CustomStreamWrapper)
        await response.aclose()
        await asyncio.sleep(0.05)
        assert run.recorder.events == (), "cancellation fabricated a successful final response"
        if provider == "anthropic" and not run.wire.closed:
            pytest.xfail("STREAM-001: explicit close does not reach the Anthropic transport")
        assert run.wire.closed, "cancel followed by explicit close did not reach transport"


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ("openai", "gemini"))
async def test_close_inside_cancelled_scope(provider: Provider) -> None:
    case: Final = Case(provider=provider, mode="async", lifecycle="close", close_yield=True)
    async with session(case) as run:
        response: Final = await run.call()
        first: Final = await first_content(response, "async", "chat")
        assert text((first,), "chat") == case.fragments[0]
        from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

        assert isinstance(response, CustomStreamWrapper)
        with anyio.CancelScope() as scope:
            scope.cancel()
            await response.aclose()
        assert run.wire.closed, "active cancellation interrupted explicit transport close"
        await asyncio.sleep(0.05)
        assert run.recorder.events == (), "closing an unfinished stream fabricated a final callback"


def assert_success_output(
    chunks: tuple[dict[str, JsonValue], ...],
    case: Case,
    event: Event,
    tool_usage: tuple[int, int, int] = (11, 7, 18),
) -> None:
    assert event.outcome == "success" and event.mode == case.mode, event
    assert isinstance(event.response, dict), event
    assert len(objects(event.response["choices"])) == 1
    callback_choice: Final = objects(event.response["choices"])[0]
    assert callback_choice.get("index") == 0
    assert event.response.get("model") == case.response_model
    assert event.response.get("object") == "chat.completion"
    assert all(
        chunk.get("object") == ("text_completion" if case.surface == "text" else "chat.completion.chunk")
        for chunk in chunks
    )
    assert all(isinstance(chunk.get("created"), int) and cast(int, chunk["created"]) > 0 for chunk in chunks)
    assert all(isinstance(chunk.get("id"), str) and chunk["id"] for chunk in chunks)
    assert all(choice.get("index") == 0 for chunk in chunks for choice in objects(chunk["choices"]))
    assert field(callback_choice.get("message"), "role") == "assistant"
    assert all(chunk.get("model") == case.response_model for chunk in chunks)
    assert len({str(chunk.get("id")) for chunk in chunks}) == 1
    assert event.response.get("id") == chunks[0].get("id")
    if case.surface == "chat":
        roles: Final = tuple(
            field(choice.get("delta"), "role")
            for chunk in chunks
            for choice in objects(chunk["choices"])
            if field(choice.get("delta"), "role") is not None
        )
        assert all(role == "assistant" for role in roles)
        if case.text:
            assert roles
    callback_content: Final = field(callback_choice.get("message"), "content")
    if case.tool:
        assert (callback_content or "") == case.tool_preamble
        assert text(chunks, case.surface) == case.tool_preamble
    else:
        assert callback_content == case.text
    reasons: Final = tuple(
        choice.get("finish_reason")
        for chunk in chunks
        for choice in objects(chunk["choices"])
        if choice.get("finish_reason") is not None
    )
    assert reasons == ("tool_calls" if case.tool else "stop",), chunks
    terminal_index: Final = next(
        index
        for index, chunk in enumerate(chunks)
        if any(choice.get("finish_reason") is not None for choice in objects(chunk["choices"]))
    )
    assert text(chunks[terminal_index + 1 :], case.surface) == "", "content arrived after terminal completion"
    if not case.tool:
        assert text(chunks, case.surface) == case.text
        assert callback_choice.get("finish_reason") == reasons[0]
        return
    assert (
        tuple(field(event.response.get("usage"), key) for key in ("prompt_tokens", "completion_tokens", "total_tokens"))
        == tool_usage
    )
    calls: Final = tuple(
        call
        for chunk in chunks
        for choice in objects(chunk["choices"])
        for call in objects(field(choice.get("delta"), "tool_calls") or [])
    )
    assert all(call.get("index") == 0 for call in calls)
    assert all(call.get("type") in (None, "function") for call in calls)
    assert tuple(call["id"] for call in calls if call.get("id")) == (TOOL_ID,), calls
    assert tuple(field(call.get("function"), "name") for call in calls if field(call.get("function"), "name")) == (
        "lookup",
    )
    args: Final = "".join(str(field(call.get("function"), "arguments") or "") for call in calls)
    assert JSON_OBJECT.validate_json(args) == JSON_OBJECT.validate_json(case.text), args
    if case.provider != "gemini":
        assert args == case.text
    callback_calls: Final = objects(field(callback_choice.get("message"), "tool_calls"))
    assert len(callback_calls) == 1
    assert callback_calls[0].get("type") == "function"
    assert callback_calls[0]["id"] == TOOL_ID
    assert field(callback_calls[0].get("function"), "name") == "lookup"
    callback_args: Final = field(callback_calls[0].get("function"), "arguments")
    assert isinstance(callback_args, str)
    assert JSON_OBJECT.validate_json(callback_args) == JSON_OBJECT.validate_json(case.text)
    if case.provider != "gemini":
        assert callback_args == case.text
    if case.provider in ("gemini", "mock") and callback_choice.get("finish_reason") == "stop":
        pytest.xfail("STREAM-003: Gemini/structured mock tool callback has stop instead of tool_calls")
    assert callback_choice.get("finish_reason") == reasons[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ("openai", "anthropic", "gemini", "openai_text"))
@pytest.mark.parametrize("mode", ("sync", "async"))
async def test_sse_without_optional_space(provider: Provider, mode: Mode) -> None:
    case: Final = Case(provider=provider, mode=mode, sse_space=False, fragments=("café 猫 ", "🐈 e\u0301"))
    async with session(case) as run:
        chunks: Final = await drain(await run.call(), mode)
        events: Final = await final_events(run.recorder)
        assert len(events) == 1
        assert_success_output(chunks, case, events[0])
        assert isinstance(events[0].response, dict)
        assert tuple(
            field(events[0].response.get("usage"), key)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        ) == (11, 7, 18)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ("openai", "anthropic", "gemini"))
@pytest.mark.parametrize("mode", ("sync", "async"))
@pytest.mark.parametrize("split_bytes", (0, 1, 13))
@pytest.mark.parametrize("surface", ("chat", "text"))
@pytest.mark.parametrize(
    "fragments", (("",), ("hello ", "world"), ("café ", "猫", " 🐈", " e\u0301")), ids=("empty", "ascii", "unicode")
)
async def test_fragmented_text(
    provider: Provider, mode: Mode, split_bytes: int, fragments: tuple[str, ...], surface: Surface
) -> None:
    case: Final = Case(provider=provider, mode=mode, surface=surface, fragments=fragments, split_bytes=split_bytes)
    async with session(case) as run:
        chunks: Final = await drain(await run.call(), mode)
        events: Final = await final_events(run.recorder)
        assert len(events) == 1
        assert_success_output(chunks, case, events[0])


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ("openai", "anthropic", "gemini"))
@pytest.mark.parametrize("mode", ("sync", "async"))
@pytest.mark.parametrize("split_bytes", (0, 1, 13))
async def test_fragmented_tool(provider: Provider, mode: Mode, split_bytes: int) -> None:
    case: Final = Case(
        provider=provider,
        mode=mode,
        tool=True,
        split_bytes=split_bytes,
        fragments=(TOOL_ARGUMENTS[:9], TOOL_ARGUMENTS[9:17], TOOL_ARGUMENTS[17:]),
    )
    async with session(case) as run:
        chunks: Final = await drain(await run.call(), mode)
        events: Final = await final_events(run.recorder)
        assert len(events) == 1
        assert_success_output(chunks, case, events[0])


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("sync", "async"))
@pytest.mark.parametrize("surface", ("chat", "text"))
async def test_native_text_bridge(mode: Mode, surface: Surface) -> None:
    case: Final = Case(provider="openai_text", mode=mode, surface=surface)
    async with session(case) as run:
        chunks: Final = await drain(await run.call(), mode)
        events: Final = await final_events(run.recorder)
        assert len(events) == 1
        assert run.wire.requests == ("https://api.openai.com/v1/completions",)
        assert_success_output(chunks, case, events[0])


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ("openai", "anthropic", "gemini"))
@pytest.mark.parametrize("mode", ("sync", "async"))
@pytest.mark.parametrize("admission", (0, 999))
async def test_provider_usage_outranks_admission(provider: Provider, mode: Mode, admission: int) -> None:
    case: Final = Case(provider=provider, mode=mode, admission=admission)
    async with session(case) as run:
        chunks: Final = await drain(await run.call(), mode)
        events: Final = await final_events(run.recorder)
        assert len(events) == 1
        assert isinstance(events[0].response, dict)
        visible_usage: Final = tuple(chunk["usage"] for chunk in chunks if chunk.get("usage") is not None)
        assert len(visible_usage) == 1
        assert tuple(
            field(visible_usage[0], key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        ) == (11, 7, 18)
        assert tuple(
            field(events[0].response.get("usage"), key)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        ) == (11, 7, 18)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("sync", "async"))
@pytest.mark.parametrize("surface", ("chat", "text"))
@pytest.mark.parametrize("options", ("omitted", "none", "empty", "hidden", "visible"))
@pytest.mark.parametrize("admission_key", ("metadata", "litellm_metadata"))
@pytest.mark.parametrize(
    "reservation",
    ((None, "count"), (None, "null"), (None, "empty"), (None, "null_count"), (0, "count"), (1, "count"), (37, "count")),
    ids=("absent", "null", "empty", "null-count", "zero", "one", "larger"),
)
async def test_mock_reservations(
    mode: Mode,
    surface: Surface,
    options: Options,
    admission_key: Literal["metadata", "litellm_metadata"],
    reservation: tuple[int | None, Literal["count", "null", "empty", "null_count"]],
) -> None:
    admitted, state = reservation
    case: Final = Case(
        provider="mock",
        mode=mode,
        surface=surface,
        options=options,
        admission=admitted,
        admission_key=admission_key,
        reservation=state,
    )
    async with session(case) as run:
        chunks: Final = await drain(await run.call(), mode)
        events: Final = await final_events(run.recorder)
        assert len(events) == 1
        assert run.wire.requests == ()
        assert_success_output(chunks, case, events[0])
        assert isinstance(events[0].response, dict)
        expected: Final = (8, 2, 10) if admitted is None else (admitted, 20, admitted + 20)
        assert (
            tuple(
                field(events[0].response.get("usage"), key)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            )
            == expected
        )
        visible_usage: Final = tuple(chunk["usage"] for chunk in chunks if chunk.get("usage") is not None)
        if options == "visible":
            assert len(visible_usage) == 1
            assert (
                tuple(field(visible_usage[0], key) for key in ("prompt_tokens", "completion_tokens", "total_tokens"))
                == expected
            )
        elif surface == "chat":
            assert visible_usage == ()
        else:
            assert all(
                tuple(field(item, key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")) == (0, 0, 0)
                for item in visible_usage
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ("openai", "anthropic", "gemini", "mock", "openai_text"))
@pytest.mark.parametrize("mode", ("sync", "async"))
@pytest.mark.parametrize("surface", ("chat", "text"))
async def test_stream_nonstream_parity(provider: Provider, mode: Mode, surface: Surface) -> None:
    case: Final = Case(provider=provider, mode=mode, surface=surface, admission=11 if provider == "mock" else None)
    async with session(case) as streaming:
        chunks: Final = await drain(await streaming.call(), mode)
        events: Final = await final_events(streaming.recorder)
        assert len(events) == 1
        assert_success_output(chunks, case, events[0])
    async with session(case) as nonstream:
        response: Final = snapshot(await nonstream.call(stream=False))
        choice: Final = objects(response["choices"])[0]
        content: Final = choice.get("text") if surface == "text" else field(choice.get("message"), "content")
        assert content == text(chunks, surface) == case.text
        assert choice.get("finish_reason") == "stop"
        nonstream_events: Final = await final_events(nonstream.recorder)
        assert len(nonstream_events) == 1
        assert nonstream_events[0].outcome == "success"
        assert isinstance(events[0].response, dict)
        assert tuple(
            field(response.get("usage"), key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        ) == tuple(
            field(events[0].response.get("usage"), key)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("sync", "async"))
@pytest.mark.parametrize("surface", ("chat", "text"))
@pytest.mark.parametrize("payload", ("text", "response"))
@pytest.mark.parametrize("content", ("", "café 猫 🐈 e\u0301"), ids=("empty", "unicode"))
async def test_mock_content(mode: Mode, surface: Surface, payload: Literal["text", "response"], content: str) -> None:
    case: Final = Case(provider="mock", mode=mode, surface=surface, fragments=(content,), mock_payload=payload)
    async with session(case) as run:
        chunks: Final = await drain(await run.call(), mode)
        events: Final = await final_events(run.recorder)
        assert len(events) == 1
        assert_success_output(chunks, case, events[0])
        if payload == "text" and content == "" and run.wire.requests:
            pytest.xfail("STREAM-004: an empty mock_response string reaches the provider transport")
        assert run.wire.requests == (), "mock_response must not send a provider request"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("sync", "async"))
async def test_mock_tool(mode: Mode) -> None:
    case: Final = Case(provider="mock", mode=mode, tool=True, fragments=(TOOL_ARGUMENTS,))
    async with session(case) as run:
        chunks: Final = await drain(await run.call(), mode)
        events: Final = await final_events(run.recorder)
        assert len(events) == 1
        assert run.wire.requests == ()
        assert_success_output(chunks, case, events[0])


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ("openai", "anthropic", "gemini", "mock"))
@pytest.mark.parametrize("mode", ("sync", "async"))
async def test_tool_with_text(provider: Provider, mode: Mode) -> None:
    case: Final = Case(
        provider=provider,
        mode=mode,
        tool=True,
        tool_preamble="Let me check 猫. ",
        fragments=(TOOL_ARGUMENTS[:9], TOOL_ARGUMENTS[9:17], TOOL_ARGUMENTS[17:]),
    )
    async with session(case) as run:
        chunks: Final = await drain(await run.call(), mode)
        events: Final = await final_events(run.recorder)
        assert len(events) == 1
        assert_success_output(chunks, case, events[0])


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("sync", "async"))
@pytest.mark.parametrize("options", ("omitted", "none", "empty", "hidden", "visible"))
@pytest.mark.parametrize("model_name", ("gpt-4o-mini", "gpt-6-astra"))
@pytest.mark.parametrize("preamble", ("", "Let me check 猫. "), ids=("tool-only", "with-text"))
async def test_tool_usage_options(mode: Mode, options: Options, model_name: str, preamble: str) -> None:
    case: Final = Case(
        provider="openai",
        mode=mode,
        options=options,
        model_name=model_name,
        tool=True,
        tool_preamble=preamble,
        fragments=(TOOL_ARGUMENTS[:9], TOOL_ARGUMENTS[9:17], TOOL_ARGUMENTS[17:]),
    )
    fallback_completion: Final = 42 if preamble else (36 if model_name == "gpt-4o-mini" else 35)
    expected: Final = (
        (8, fallback_completion, 8 + fallback_completion) if options in ("empty", "hidden") else (11, 7, 18)
    )
    async with session(case) as run:
        chunks: Final = await drain(await run.call(), mode)
        events: Final = await final_events(run.recorder)
        assert len(events) == 1
        assert_success_output(chunks, case, events[0], tool_usage=expected)
        assert len(run.wire.requests) == 1
        assert run.wire.request_bodies[0].get("stream_options") == (
            {"include_usage": True} if options in ("omitted", "none") else case.stream_options.get("stream_options")
        )
        visible_usage: Final = tuple(chunk["usage"] for chunk in chunks if chunk.get("usage") is not None)
        if options in ("empty", "hidden"):
            assert visible_usage == ()
        else:
            assert len(visible_usage) == 1 and chunks[-1].get("usage") == visible_usage[0]
            assert (
                tuple(field(visible_usage[0], key) for key in ("prompt_tokens", "completion_tokens", "total_tokens"))
                == expected
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("sync", "async"))
@pytest.mark.parametrize("options", ("omitted", "none", "empty", "hidden", "visible"))
async def test_mock_tool_zero_reservation(mode: Mode, options: Options) -> None:
    case: Final = Case(
        provider="mock",
        mode=mode,
        options=options,
        tool=True,
        prompt_tokens=0,
        admission=0,
        fragments=(TOOL_ARGUMENTS,),
    )
    async with session(case) as run:
        chunks: Final = await drain(await run.call(), mode)
        events: Final = await final_events(run.recorder)
        assert len(events) == 1
        assert run.wire.requests == ()
        visible_usage: Final = tuple(chunk["usage"] for chunk in chunks if chunk.get("usage") is not None)
        if options == "visible":
            assert len(visible_usage) == 1 and chunks[-1].get("usage") == visible_usage[0]
            assert tuple(
                field(visible_usage[0], key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            ) == (0, 7, 7)
        else:
            assert visible_usage == ()
        assert_success_output(chunks, case, events[0], tool_usage=(0, 7, 7))


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ("openai", "gemini"))
@pytest.mark.parametrize("mode", ("sync", "async"))
@pytest.mark.parametrize("surface", ("chat", "text"))
async def test_absent_provider_usage(provider: Provider, mode: Mode, surface: Surface) -> None:
    case: Final = Case(provider=provider, mode=mode, surface=surface, prompt_tokens=None)
    async with session(case) as run:
        chunks: Final = await drain(await run.call(), mode)
        events: Final = await final_events(run.recorder)
        assert len(events) == 1
        assert_success_output(chunks, case, events[0])
        assert isinstance(events[0].response, dict)
        assert tuple(
            field(events[0].response.get("usage"), key)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        ) == (8, 2, 10)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ("openai", "anthropic", "gemini"))
@pytest.mark.parametrize("surface", ("chat", "text"))
async def test_unicode_partial_usage(provider: Provider, surface: Surface) -> None:
    case: Final = Case(
        provider=provider,
        mode="async",
        surface=surface,
        lifecycle="failure",
        fragments=("café 猫 🐈 ", "e\u0301 Zürich ", "third ", "last"),
    )
    async with session(case) as run:
        response: Final = await run.call()
        first: Final = await first_content(response, "async", surface)
        assert text((first,), surface) == case.fragments[0]
        with pytest.raises(litellm.exceptions.MidStreamFallbackError, match="matrix provider disconnected"):
            await drain(response, "async")
        events: Final = await final_events(run.recorder)
        assert len(events) == 1
        assert events[0].outcome == "failure" and events[0].mode == "async"
        partial: Final = snapshot(events[0].partial_usage)
        expected: Final = {"openai": (8, 14, 22), "anthropic": (11, 9, 20), "gemini": (8, 15, 23)}[provider]
        assert tuple(partial[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("sync", "async"))
@pytest.mark.parametrize("surface", ("chat", "text"))
@pytest.mark.parametrize("model_name,completion_tokens", (("gpt-4o-mini", 9), ("gpt-6-astra", 10)))
async def test_model_specific_usage_fallback(
    mode: Mode, surface: Surface, model_name: str, completion_tokens: int
) -> None:
    case: Final = Case(
        provider="openai",
        mode=mode,
        surface=surface,
        model_name=model_name,
        prompt_tokens=None,
        fragments=("café 猫 ", "🐈 e\u0301"),
    )
    async with session(case) as run:
        chunks: Final = await drain(await run.call(), mode)
        events: Final = await final_events(run.recorder)
        assert len(events) == 1
        assert_success_output(chunks, case, events[0])
        assert isinstance(events[0].response, dict)
        expected: Final = (8, completion_tokens, 8 + completion_tokens)
        assert (
            tuple(
                field(events[0].response.get("usage"), key)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            )
            == expected
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name,prompt_tokens", (("gpt-4o-mini", 16), ("gpt-6-astra", 17)))
@pytest.mark.parametrize("mode", ("sync", "async"))
@pytest.mark.parametrize("surface", ("chat", "text"))
async def test_unicode_prompt_usage(model_name: str, prompt_tokens: int, mode: Mode, surface: Surface) -> None:
    case: Final = Case(provider="openai", mode=mode, surface=surface, model_name=model_name, prompt_tokens=None)
    prompt: Final = "café 猫 🐈 e\u0301"
    async with session(case) as run:
        chunks: Final = await drain(await run.call(prompt=prompt), mode)
        events: Final = await final_events(run.recorder)
        assert len(events) == 1
        assert_success_output(chunks, case, events[0])
        assert run.wire.request_bodies[0]["messages"] == [{"role": "user", "content": prompt}]
        assert isinstance(events[0].response, dict)
        expected: Final = (prompt_tokens, 2, prompt_tokens + 2)
        for usage in (chunks[-1].get("usage"), events[0].response.get("usage")):
            assert (
                tuple(field(usage, key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")) == expected
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ("openai", "anthropic", "gemini", "openai_text"))
@pytest.mark.parametrize("mode", ("sync", "async"))
@pytest.mark.parametrize("surface", ("chat", "text"))
async def test_exhaustion_cleanup(provider: Provider, mode: Mode, surface: Surface) -> None:
    case: Final = Case(provider=provider, mode=mode, surface=surface)
    async with session(case) as run:
        chunks: Final = await drain(await run.call(), mode)
        events: Final = await final_events(run.recorder)
        assert len(events) == 1
        assert_success_output(chunks, case, events[0])
        if provider == "anthropic" and not run.wire.closed:
            pytest.xfail("STREAM-001: Anthropic exhaustion does not close the simulated transport")
        assert run.wire.closed, "exhaustion did not close the simulated transport before fixture teardown"


@pytest.mark.parametrize("provider", ("openai", "anthropic", "gemini", "mock", "openai_text"))
@pytest.mark.parametrize("surface", ("chat", "text"))
def test_sync_without_running_loop(provider: Provider, surface: Surface) -> None:
    case: Final = Case(provider=provider, mode="sync", surface=surface, admission=11 if provider == "mock" else None)
    with synchronous_session(case) as run:
        response: Final = (
            litellm.completion(messages=[{"role": "user", "content": "hello"}], **run.kwargs())
            if surface == "chat"
            else litellm.text_completion(prompt="hello", **run.kwargs())
        )
        chunks: Final = tuple(snapshot(chunk) for chunk in cast(Iterator[object], response))
        assert run.recorder.arrived.wait(5), "missing final SDK callback without an active event loop"
        time.sleep(0.05)
        events: Final = run.recorder.events
        assert len(events) == 1
        assert_success_output(chunks, case, events[0])
        assert isinstance(events[0].response, dict)
        expected: Final = (11, 20, 31) if provider == "mock" else (11, 7, 18)
        assert (
            tuple(
                field(events[0].response.get("usage"), key)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            )
            == expected
        )
