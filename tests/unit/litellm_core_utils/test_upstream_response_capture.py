import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from typing import Final

import httpx
import openai
import pytest

import litellm
from litellm.litellm_core_utils.upstream_response_capture import (
    CapturedUpstreamResponse,
    UpstreamResponseCapture,
    async_capture_explicit_response_headers,
    async_capture_response_headers,
    capture_explicit_response_headers,
    capture_response_headers,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler, MaskedHTTPStatusError
from litellm.types.utils import ModelResponse

HEADERS: Final = (("x-request-id", "upstream-probe"), ("x-probe", "first"), ("x-probe", "second"))
CONTENT: Final = b'{"choices":[]}'


class ProbeBody(httpx.SyncByteStream, httpx.AsyncByteStream):
    def __init__(self, failure: httpx.ReadError | None = None) -> None:
        self.failure: Final = failure
        self.started = False
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        self.started = True
        yield CONTENT
        if self.failure is not None:
            raise self.failure

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self:
            yield chunk

    def close(self) -> None:
        self.closed = True

    async def aclose(self) -> None:
        self.close()


@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("status", (200, 429))
def test_sync_handler_captures_before_body_or_status_failure(stream: bool, status: int) -> None:
    capture: Final = UpstreamResponseCapture()
    body: Final = ProbeBody()

    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, headers=HEADERS, stream=body, request=request)

    def verify_capture_before_body(response: httpx.Response) -> None:
        assert not body.started
        assert capture.responses == (CapturedUpstreamResponse("attempt", response.status_code, HEADERS),)

    with httpx.Client(
        transport=httpx.MockTransport(upstream),
        event_hooks={"response": [capture_response_headers, verify_capture_before_body]},
    ) as client:
        handler: Final = HTTPHandler(client=client)
        with capture.bind("attempt"):
            if status >= 400:
                with pytest.raises(MaskedHTTPStatusError) as failure:
                    handler.post("https://upstream.invalid/", stream=stream)
                assert failure.value.response.status_code == status
            else:
                response: Final = handler.post("https://upstream.invalid/", stream=stream)
                assert response.read() == CONTENT
                response.close()
    assert body.closed
    assert capture.responses == (CapturedUpstreamResponse("attempt", status, HEADERS),)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("status", (200, 429))
async def test_async_handler_captures_before_body_or_status_failure(stream: bool, status: int) -> None:
    capture: Final = UpstreamResponseCapture()
    body: Final = ProbeBody()

    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, headers=HEADERS, stream=body, request=request)

    async def verify_capture_before_body(response: httpx.Response) -> None:
        assert not body.started
        assert capture.responses == (CapturedUpstreamResponse("attempt", response.status_code, HEADERS),)

    handler: Final = AsyncHTTPHandler(
        transport=httpx.MockTransport(upstream),
        event_hooks={"response": [async_capture_response_headers, verify_capture_before_body]},
    )
    try:
        with capture.bind("attempt"):
            if status >= 400:
                with pytest.raises(MaskedHTTPStatusError) as failure:
                    await handler.post("https://upstream.invalid/", stream=stream)
                assert failure.value.response.status_code == status
            else:
                response: Final = await handler.post("https://upstream.invalid/", stream=stream)
                assert await response.aread() == CONTENT
                await response.aclose()
    finally:
        await handler.close()
    assert body.closed
    assert capture.responses == (CapturedUpstreamResponse("attempt", status, HEADERS),)


@pytest.mark.parametrize("stream", (False, True))
def test_sync_body_read_failure_preserves_capture_and_exception(stream: bool) -> None:
    capture: Final = UpstreamResponseCapture()
    failure: Final = httpx.ReadError("upstream body interrupted")
    body: Final = ProbeBody(failure)

    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers=HEADERS, stream=body, request=request)

    with httpx.Client(
        transport=httpx.MockTransport(upstream), event_hooks={"response": [capture_response_headers]}
    ) as client:
        with capture.bind("attempt"):
            if stream:
                with client.stream("POST", "https://upstream.invalid/") as response:
                    assert not body.started
                    with pytest.raises(httpx.ReadError) as stream_failure:
                        response.read()
                assert stream_failure.value is failure
            else:
                with pytest.raises(httpx.ReadError) as read_failure:
                    client.post("https://upstream.invalid/")
                assert read_failure.value is failure
    assert body.closed
    assert capture.responses == (CapturedUpstreamResponse("attempt", 200, HEADERS),)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", (False, True))
async def test_async_body_read_failure_preserves_capture_and_exception(stream: bool) -> None:
    capture: Final = UpstreamResponseCapture()
    failure: Final = httpx.ReadError("upstream body interrupted")
    body: Final = ProbeBody(failure)

    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers=HEADERS, stream=body, request=request)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(upstream), event_hooks={"response": [async_capture_response_headers]}
    ) as client:
        with capture.bind("attempt"):
            if stream:
                async with client.stream("POST", "https://upstream.invalid/") as response:
                    assert not body.started
                    with pytest.raises(httpx.ReadError) as stream_failure:
                        await response.aread()
                assert stream_failure.value is failure
            else:
                with pytest.raises(httpx.ReadError) as read_failure:
                    await client.post("https://upstream.invalid/")
                assert read_failure.value is failure
    assert body.closed
    assert capture.responses == (CapturedUpstreamResponse("attempt", 200, HEADERS),)


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit", (False, True))
async def test_shared_client_keeps_concurrent_captures_separate(explicit: bool) -> None:
    count: Final = 12
    started: Final = tuple(asyncio.Event() for _ in range(count))

    async def upstream(request: httpx.Request) -> httpx.Response:
        started[int(request.url.path.removeprefix("/"))].set()
        for event in started:
            await event.wait()
        return httpx.Response(200, headers={"x-request-id": request.url.path}, content=CONTENT)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(upstream),
        event_hooks={
            "response": [async_capture_explicit_response_headers if explicit else async_capture_response_headers]
        },
    ) as client:

        async def call(index: int) -> tuple[CapturedUpstreamResponse, ...]:
            capture: Final = UpstreamResponseCapture()
            marker: Final = str(index)
            with capture.bind("incorrect-ambient-attempt" if explicit else marker):
                response: Final = await client.get(
                    f"https://upstream.invalid/{marker}",
                    extensions=capture.request_extensions(marker) if explicit else None,
                )
            assert response.content == CONTENT
            assert len(capture.responses) == 1
            assert capture.responses[0].attempt_id == marker
            assert dict(capture.responses[0].headers)["x-request-id"] == f"/{marker}"
            return capture.responses

        captured: Final = await asyncio.gather(*(call(index) for index in range(count)))
    assert len(frozenset(record[0].attempt_id for record in captured)) == count


def test_nested_attempts_restore_binding_and_snapshots_do_not_alias_responses() -> None:
    capture: Final = UpstreamResponseCapture()
    response: Final = httpx.Response(429, headers=HEADERS)
    with capture.bind("outer"):
        capture_response_headers(response)
        before_fallback: Final = capture.responses
        with capture.bind("fallback"):
            capture_response_headers(httpx.Response(200, headers=HEADERS))
        capture_response_headers(response)
    capture_response_headers(response)
    response.headers.clear()
    assert before_fallback == (CapturedUpstreamResponse("outer", 429, HEADERS),)
    assert capture.responses == (
        CapturedUpstreamResponse("outer", 429, HEADERS),
        CapturedUpstreamResponse("fallback", 200, HEADERS),
        CapturedUpstreamResponse("outer", 429, HEADERS),
    )


def test_explicit_owner_survives_redirect_without_going_on_the_wire() -> None:
    capture: Final = UpstreamResponseCapture()
    extensions: Final = capture.request_extensions("redirect")

    def upstream(request: httpx.Request) -> httpx.Response:
        assert set(extensions).isdisjoint(request.headers)
        assert request.content == CONTENT
        if request.url.path == "/start":
            return httpx.Response(307, headers=(*HEADERS, ("location", "/finish")))
        assert request.url.path == "/finish"
        return httpx.Response(200, headers=HEADERS, content=CONTENT)

    with httpx.Client(
        transport=httpx.MockTransport(upstream),
        event_hooks={"response": [capture_explicit_response_headers]},
        follow_redirects=True,
    ) as client:
        response: Final = client.post("https://upstream.invalid/start", content=CONTENT, extensions=extensions)
    assert response.content == CONTENT
    assert capture.responses == (
        CapturedUpstreamResponse("redirect", 307, tuple(response.history[0].headers.multi_items())),
        CapturedUpstreamResponse("redirect", 200, tuple(response.headers.multi_items())),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", (False, True))
async def test_litellm_chat_capture_survives_response_reconstruction(stream: bool) -> None:
    capture: Final = UpstreamResponseCapture()
    payload: Final = {
        "id": "chatcmpl-header-probe",
        "object": "chat.completion.chunk" if stream else "chat.completion",
        "created": 1,
        "model": "header-probe",
        "choices": [
            {
                "index": 0,
                "delta" if stream else "message": {"role": "assistant", "content": "captured"},
                "finish_reason": "stop",
            }
        ],
    }

    def upstream(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        content: Final = f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n" if stream else json.dumps(payload)
        return httpx.Response(200, headers=HEADERS, content=content)

    async with openai.AsyncOpenAI(
        api_key="experiment-key",
        base_url="https://upstream.invalid/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(upstream), event_hooks={"response": [async_capture_response_headers]}
        ),
    ) as client:
        with capture.bind("chat"):
            response: Final = await litellm.acompletion(
                model="openai/header-probe",
                messages=[{"role": "user", "content": "probe"}],
                client=client,
                stream=stream,
                num_retries=0,
            )
            if stream:
                chunks: Final = tuple([chunk async for chunk in response])
                rebuilt: Final = litellm.stream_chunk_builder(chunks=list(chunks))
                assert rebuilt is not None
                assert rebuilt.choices[0].message.content == "captured"
            else:
                reconstructed: Final = ModelResponse.model_validate(response.model_dump())
                assert reconstructed.choices[0].message.content == "captured"
    assert len(capture.responses) == 1
    assert capture.responses[0].attempt_id == "chat"
    assert dict(capture.responses[0].headers)["x-request-id"] == dict(HEADERS)["x-request-id"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("stream", "native_stream"), ((False, True), (True, True), (True, False)))
async def test_litellm_responses_capture_without_chat_wrapper(stream: bool, native_stream: bool) -> None:
    capture: Final = UpstreamResponseCapture()
    litellm.register_model(
        {
            "header-probe": {
                "litellm_provider": "openai",
                "mode": "responses",
                "supports_native_streaming": native_stream,
            }
        }
    )
    payload: Final = {
        "id": "resp_header_probe",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "header-probe",
        "output": [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }

    def upstream(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/responses"
        completed: Final = {"type": "response.completed", "sequence_number": 0, "response": payload}
        content: Final = (
            f"event: response.completed\ndata: {json.dumps(completed)}\n\n"
            if stream and native_stream
            else json.dumps(payload)
        )
        return httpx.Response(200, headers=HEADERS, content=content)

    handler: Final = AsyncHTTPHandler(
        transport=httpx.MockTransport(upstream), event_hooks={"response": [async_capture_response_headers]}
    )
    try:
        with capture.bind("responses"):
            response: Final = await litellm.aresponses(
                model="openai/header-probe",
                input="probe",
                api_key="experiment-key",
                api_base="https://upstream.invalid/v1",
                client=handler,
                stream=stream,
            )
            if stream:
                events: Final = tuple([event async for event in response])
                assert events[-1].type == "response.completed"
                assert events[-1].response.status == payload["status"]
            else:
                assert response.status == payload["status"]
    finally:
        await handler.close()
    assert len(capture.responses) == 1
    assert capture.responses[0].attempt_id == "responses"
    assert dict(capture.responses[0].headers)["x-request-id"] == dict(HEADERS)["x-request-id"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", (False, True))
async def test_litellm_messages_capture_without_openai_response_objects(stream: bool) -> None:
    capture: Final = UpstreamResponseCapture()
    litellm.register_model(
        {
            "header-probe": {
                "litellm_provider": "anthropic",
                "mode": "chat",
                "input_cost_per_token": 0,
                "output_cost_per_token": 0,
            }
        }
    )
    payload: Final = {
        "id": "msg_header_probe",
        "type": "message",
        "role": "assistant",
        "model": "header-probe",
        "content": [{"type": "text", "text": "captured"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    wire_body: Final = (
        f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': payload})}\n\n"
        'event: message_stop\ndata: {"type":"message_stop"}\n\n'
        if stream
        else json.dumps(payload)
    )

    def upstream(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/messages"
        return httpx.Response(200, headers=HEADERS, content=wire_body)

    handler: Final = AsyncHTTPHandler(
        transport=httpx.MockTransport(upstream), event_hooks={"response": [async_capture_response_headers]}
    )
    try:
        with capture.bind("messages"):
            response: Final = await litellm.anthropic.messages.acreate(
                model="anthropic/header-probe",
                messages=[{"role": "user", "content": "probe"}],
                max_tokens=1,
                api_key="experiment-key",
                api_base="https://upstream.invalid",
                client=handler,
                stream=stream,
            )
            if stream:
                chunks: Final = tuple([chunk async for chunk in response])
                assert b"".join(chunks).decode() == wire_body
            else:
                assert response["content"] == payload["content"]
    finally:
        await handler.close()
    assert len(capture.responses) == 1
    assert capture.responses[0].attempt_id == "messages"
    assert dict(capture.responses[0].headers)["x-request-id"] == dict(HEADERS)["x-request-id"]


@pytest.mark.asyncio
async def test_cancelled_body_keeps_headers_and_closes_stream() -> None:
    capture: Final = UpstreamResponseCapture()
    started: Final = asyncio.Event()
    blocked: Final = asyncio.Event()
    closed: Final = asyncio.Event()

    class BlockedBody(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            started.set()
            await blocked.wait()
            yield CONTENT

        async def aclose(self) -> None:
            closed.set()

    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers=HEADERS, stream=BlockedBody(), request=request)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(upstream), event_hooks={"response": [async_capture_response_headers]}
    ) as client:

        async def consume() -> None:
            with capture.bind("cancelled"):
                async with client.stream("POST", "https://upstream.invalid/") as response:
                    await response.aread()

        task: Final = asyncio.create_task(consume())
        await started.wait()
        before_cancel: Final = capture.responses
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert closed.is_set()
    assert capture.responses == before_cancel == (CapturedUpstreamResponse("cancelled", 200, HEADERS),)
