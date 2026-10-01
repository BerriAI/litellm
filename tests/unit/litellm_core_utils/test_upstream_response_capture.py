import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from datetime import datetime, timezone
from typing import Final, Literal

import httpx
import openai
import pytest

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.upstream_response_capture import (
    UpstreamResponseCapture,
    async_send_with_capture,
    send_with_capture,
)
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    HTTPResponseLimitError,
    MaskedHTTPStatusError,
)
from litellm.types.utils import ModelResponse

HEADERS: Final = (("x-request-id", "upstream-probe"), ("x-probe", "first"), ("x-probe", "second"))
CONTENT: Final = b'{"choices":[]}'


class MissingReceivedHeaders(AssertionError):
    pass


def require_received_headers(capture: UpstreamResponseCapture, status: int) -> None:
    if not capture.responses:
        raise MissingReceivedHeaders(f"Received HTTP {status} with {HEADERS!r}; canonical capture is empty")
    assert len(capture.responses) == 1, capture.snapshot()
    assert capture.responses[0].status_code == status
    assert capture.responses[0].attempt_id == "attempt"
    assert tuple(pair for pair in capture.responses[0].headers if pair[0].startswith("x-")) == HEADERS


def make_logging(stream: bool = False, attempt: str = "attempt", call_type: str = "acompletion") -> Logging:
    return Logging(
        model="header-probe",
        messages=[],
        stream=stream,
        call_type=call_type,
        start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        litellm_call_id=attempt,
        function_id=attempt,
    )


class ProbeBody(httpx.SyncByteStream, httpx.AsyncByteStream):
    def __init__(self, capture: UpstreamResponseCapture, fail: bool = False) -> None:
        self.capture: Final = capture
        self.fail: Final = fail
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        assert self.capture.responses
        yield CONTENT
        if self.fail:
            raise httpx.ReadError("body interrupted")

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
    logging: Final = make_logging(stream)
    body: Final = ProbeBody(logging.upstream_response_capture)

    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, headers=HEADERS, stream=body)

    with httpx.Client(transport=httpx.MockTransport(upstream)) as client:
        handler: Final = HTTPHandler(client=client)
        if status >= 400:
            with pytest.raises(MaskedHTTPStatusError) as failure:
                handler.post("https://upstream.invalid/", stream=stream, logging_obj=logging)
            assert failure.value.response.status_code == status
        else:
            response: Final = handler.post("https://upstream.invalid/", stream=stream, logging_obj=logging)
            assert response.read() == CONTENT
            response.close()
    assert body.closed
    assert logging.upstream_response_capture.snapshot() == (
        {
            "attempt_id": "attempt",
            "status_code": status,
            "headers": HEADERS,
            "truncated": False,
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", (False, True))
@pytest.mark.parametrize("status", (200, 429))
async def test_async_handler_captures_before_body_or_status_failure(stream: bool, status: int) -> None:
    logging: Final = make_logging(stream)
    body: Final = ProbeBody(logging.upstream_response_capture)

    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, headers=HEADERS, stream=body)

    handler: Final = AsyncHTTPHandler(transport=httpx.MockTransport(upstream))
    try:
        if status >= 400:
            with pytest.raises(MaskedHTTPStatusError) as failure:
                await handler.post("https://upstream.invalid/", stream=stream, logging_obj=logging)
            assert failure.value.response.status_code == status
        else:
            response: Final = await handler.post("https://upstream.invalid/", stream=stream, logging_obj=logging)
            assert await response.aread() == CONTENT
            await response.aclose()
    finally:
        await handler.close()
    assert body.closed
    assert logging.upstream_response_capture.snapshot() == (
        {
            "attempt_id": "attempt",
            "status_code": status,
            "headers": HEADERS,
            "truncated": False,
        },
    )


@pytest.mark.asyncio
async def test_body_failure_keeps_headers_and_closes_response() -> None:
    logging: Final = make_logging()
    body: Final = ProbeBody(logging.upstream_response_capture, fail=True)

    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers=HEADERS, stream=body)

    handler: Final = AsyncHTTPHandler(transport=httpx.MockTransport(upstream))
    try:
        with pytest.raises(httpx.ReadError, match="body interrupted"):
            await handler.post("https://upstream.invalid/", logging_obj=logging)
    finally:
        await handler.close()
    assert body.closed
    assert logging.upstream_response_capture.responses[0].headers == HEADERS


@pytest.mark.asyncio
async def test_shared_client_keeps_concurrent_calls_and_unowned_calls_separate() -> None:
    count: Final = 12
    ready: Final = tuple(asyncio.Event() for _ in range(count))

    async def upstream(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/unowned":
            ready[int(request.url.path.removeprefix("/"))].set()
            for event in ready:
                await event.wait()
        return httpx.Response(200, headers={"x-request-id": request.url.path}, content=CONTENT)

    handler: Final = AsyncHTTPHandler(transport=httpx.MockTransport(upstream))

    async def call(index: int) -> str:
        logging: Final = make_logging(attempt=str(index))
        await handler.post(f"https://upstream.invalid/{index}", logging_obj=logging)
        await handler.post("https://upstream.invalid/unowned")
        captured: Final = logging.upstream_response_capture.snapshot()
        assert len(captured) == 1
        assert captured[0]["attempt_id"] == str(index)
        assert dict(captured[0]["headers"])["x-request-id"] == f"/{index}"
        return captured[0]["attempt_id"]

    try:
        captured: Final = await asyncio.gather(*(call(index) for index in range(count)))
        assert len(frozenset(captured)) == count
    finally:
        await handler.close()


def test_capture_redacts_bounds_and_snapshots_do_not_alias() -> None:
    capture: Final = UpstreamResponseCapture()
    response: Final = httpx.Response(
        200,
        headers=(
            ("x-request-id", "visible"),
            ("set-cookie", "secret"),
            ("x-api-key", "secret"),
            ("x-debug", "x" * 10000),
            ("x-probe", "first"),
            ("x-probe", "second"),
        ),
    )
    capture.record("first", response)
    snapshot: Final = capture.snapshot()
    response.headers.clear()
    for index in range(20):
        capture.record(str(index), httpx.Response(429, headers={"x-request-id": str(index)}))
    assert snapshot[0]["headers"] == (
        ("x-request-id", "visible"),
        ("set-cookie", "[REDACTED]"),
        ("x-api-key", "[REDACTED]"),
        ("x-debug", "x" * 512),
        ("x-probe", "first"),
        ("x-probe", "second"),
    )
    assert snapshot[0]["truncated"]
    assert len(capture.snapshot()) == 8
    assert capture.snapshot()[-1]["attempt_id"] == "19"
    assert all(item["truncated"] for item in capture.snapshot())


@pytest.mark.asyncio
@pytest.mark.parametrize("connect_failure", (False, True), ids=("size-limit", "redirect-connect-failure"))
async def test_bounded_get_keeps_received_headers_when_redirect_target_fails(connect_failure: bool) -> None:
    logging: Final = make_logging()
    body: Final = ProbeBody(logging.upstream_response_capture)

    def upstream(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "/next", "x-request-id": "redirect"})
        assert request.url.path == "/next"
        if connect_failure:
            raise httpx.ConnectError("upstream unavailable", request=request)
        return httpx.Response(200, headers=HEADERS, stream=body)

    handler: Final = AsyncHTTPHandler(transport=httpx.MockTransport(upstream))
    try:
        with pytest.raises(
            httpx.ConnectError if connect_failure else HTTPResponseLimitError,
            match="upstream unavailable" if connect_failure else "size limit",
        ):
            await handler.get("https://upstream.invalid/start", max_response_bytes=1, logging_obj=logging)
        captured: Final = logging.upstream_response_capture.snapshot()
        assert tuple(item["status_code"] for item in captured) == ((302,) if connect_failure else (302, 200))
        assert captured[0]["headers"] == (("location", "/next"), ("x-request-id", "redirect"))
        if not connect_failure:
            assert captured[1]["headers"] == HEADERS
            assert body.closed
        assert not handler.client.is_closed
    finally:
        await handler.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", (False, True))
async def test_litellm_chat_capture_survives_response_reconstruction(stream: bool) -> None:
    logging: Final = make_logging(stream)
    capture: Final = logging.upstream_response_capture
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
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(upstream)),
    ) as client:
        response: Final = await litellm.acompletion(
            model="openai/header-probe",
            messages=[{"role": "user", "content": "probe"}],
            client=client,
            stream=stream,
            litellm_logging_obj=logging,
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
    assert capture.responses[0].attempt_id == logging.litellm_call_id
    assert dict(capture.responses[0].headers)["x-request-id"] == dict(HEADERS)["x-request-id"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("stream", "native_stream"), ((False, True), (True, True), (True, False)))
async def test_litellm_responses_capture_without_chat_wrapper(stream: bool, native_stream: bool) -> None:
    logging: Final = make_logging(stream)
    capture: Final = logging.upstream_response_capture
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

    handler: Final = AsyncHTTPHandler(transport=httpx.MockTransport(upstream))
    try:
        response: Final = await litellm.aresponses(
            model="openai/header-probe",
            input="probe",
            api_key="experiment-key",
            api_base="https://upstream.invalid/v1",
            client=handler,
            stream=stream,
            litellm_logging_obj=logging,
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
    assert capture.responses[0].attempt_id == logging.litellm_call_id
    assert dict(capture.responses[0].headers)["x-request-id"] == dict(HEADERS)["x-request-id"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", (False, True))
async def test_litellm_messages_capture_without_openai_response_objects(stream: bool) -> None:
    logging: Final = make_logging(stream)
    capture: Final = logging.upstream_response_capture
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

    handler: Final = AsyncHTTPHandler(transport=httpx.MockTransport(upstream))
    try:
        response: Final = await litellm.anthropic.messages.acreate(
            model="anthropic/header-probe",
            messages=[{"role": "user", "content": "probe"}],
            max_tokens=1,
            api_key="experiment-key",
            api_base="https://upstream.invalid",
            client=handler,
            stream=stream,
            litellm_logging_obj=logging,
        )
        if stream:
            chunks: Final = tuple([chunk async for chunk in response])
            assert b"".join(chunks).decode() == wire_body
        else:
            assert response["content"] == payload["content"]
    finally:
        await handler.close()
    assert len(capture.responses) == 1
    assert capture.responses[0].attempt_id == logging.litellm_call_id
    assert dict(capture.responses[0].headers)["x-request-id"] == dict(HEADERS)["x-request-id"]


@pytest.mark.asyncio
async def test_cancelled_body_retains_headers_and_closes_stream() -> None:
    logging: Final = make_logging()
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
        return httpx.Response(200, headers=HEADERS, stream=BlockedBody())

    handler: Final = AsyncHTTPHandler(transport=httpx.MockTransport(upstream))
    try:
        task: Final = asyncio.create_task(handler.post("https://upstream.invalid/", logging_obj=logging))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed.is_set()
        assert logging.upstream_response_capture.responses[0].headers == HEADERS
    finally:
        await handler.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("sync", (False, True), ids=("async", "sync"))
@pytest.mark.parametrize(
    "failure",
    (
        pytest.param("body", id="body-read"),
        pytest.param("connect", id="before-headers"),
        pytest.param(
            "redirect",
            marks=pytest.mark.xfail(
                strict=True,
                raises=MissingReceivedHeaders,
                reason="PR #44053: received redirect headers lost when the next connection fails",
            ),
        ),
        pytest.param(
            "hook",
            marks=pytest.mark.xfail(
                strict=True,
                raises=MissingReceivedHeaders,
                reason="PR #44053: response hook reads the body before canonical capture",
            ),
        ),
    ),
)
async def test_received_headers_survive_send_failure(
    sync: bool, failure: Literal["body", "connect", "redirect", "hook"]
) -> None:
    capture: Final = UpstreamResponseCapture()

    class InterruptedBody(httpx.SyncByteStream, httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.closed = False

        def __iter__(self) -> Iterator[bytes]:
            yield CONTENT
            if failure != "redirect":
                raise httpx.ReadError("interrupted after headers")

        async def __aiter__(self) -> AsyncIterator[bytes]:
            for chunk in self:
                yield chunk

        def close(self) -> None:
            self.closed = True

        async def aclose(self) -> None:
            self.close()

    body: Final = InterruptedBody()

    def upstream(request: httpx.Request) -> httpx.Response:
        if failure == "connect" or request.url.path == "/unreachable":
            raise httpx.ConnectError("connection unavailable", request=request)
        return httpx.Response(
            307 if failure == "redirect" else 200,
            headers=(*HEADERS, ("location", "/unreachable")),
            stream=body,
        )

    def read_hook(response: httpx.Response) -> None:
        response.read()

    async def async_read_hook(response: httpx.Response) -> None:
        await response.aread()

    error: Final = httpx.ConnectError if failure in ("connect", "redirect") else httpx.ReadError
    if sync:
        with httpx.Client(
            transport=httpx.MockTransport(upstream),
            follow_redirects=True,
            event_hooks={"response": [read_hook] if failure == "hook" else []},
        ) as client:
            with pytest.raises(error):
                send_with_capture(
                    client, client.build_request("GET", "https://upstream.invalid/"), capture, "attempt", False
                )
            assert not client.is_closed
    else:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(upstream),
            follow_redirects=True,
            event_hooks={"response": [async_read_hook] if failure == "hook" else []},
        ) as async_client:
            with pytest.raises(error):
                await async_send_with_capture(
                    async_client,
                    async_client.build_request("GET", "https://upstream.invalid/"),
                    capture,
                    "attempt",
                    False,
                )
            assert not async_client.is_closed
    assert body.closed == (failure != "connect")
    if failure == "connect":
        assert capture.snapshot() == ()
        return
    require_received_headers(capture, 307 if failure == "redirect" else 200)


@pytest.mark.asyncio
@pytest.mark.parametrize("sync", (False, True), ids=("async", "sync"))
@pytest.mark.parametrize(
    ("provider", "endpoint"), (("openai", "embeddings"), ("azure", "embeddings"), ("openai", "images"))
)
@pytest.mark.xfail(
    strict=True,
    raises=MissingReceivedHeaders,
    reason="PR #44053: non-chat SDK calls do not pass canonical capture ownership",
)
async def test_non_chat_sdk_calls_capture_received_headers(
    sync: bool, provider: Literal["openai", "azure"], endpoint: Literal["embeddings", "images"]
) -> None:
    operation: Final = "embedding" if endpoint == "embeddings" else "image_generation"
    logging: Final = make_logging(call_type=operation if sync else f"a{operation}")

    def upstream(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert json.loads(request.content)["input" if endpoint == "embeddings" else "prompt"] == (
            ["probe"] if endpoint == "embeddings" else "probe"
        )
        assert request.url.path.endswith("/embeddings" if endpoint == "embeddings" else "/images/generations")
        payload: Final = (
            {
                "object": "list",
                "model": "header-probe",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.5]}],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            }
            if endpoint == "embeddings"
            else {"created": 1, "data": [{"url": "https://upstream.invalid/image.png"}]}
        )
        return httpx.Response(200, headers=HEADERS, json=payload)

    options: Final = {
        "model": f"{provider}/header-probe",
        "litellm_logging_obj": logging,
        "api_key": "synthetic",
        "api_base": "https://upstream.invalid/v1",
        "api_version": "2024-02-01",
        "num_retries": 0,
    }
    if sync:
        with httpx.Client(transport=httpx.MockTransport(upstream)) as borrowed:
            client: Final = (
                openai.AzureOpenAI(
                    api_key="synthetic",
                    base_url="https://upstream.invalid/v1",
                    api_version="2024-02-01",
                    http_client=borrowed,
                    max_retries=0,
                )
                if provider == "azure"
                else openai.OpenAI(
                    api_key="synthetic", base_url="https://upstream.invalid/v1", http_client=borrowed, max_retries=0
                )
            )
            result: Final = (
                litellm.embedding(input=["probe"], client=client, **options)
                if endpoint == "embeddings"
                else litellm.image_generation(prompt="probe", client=client, **options)
            )
            assert not borrowed.is_closed
    else:
        async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as async_borrowed:
            async_client: Final = (
                openai.AsyncAzureOpenAI(
                    api_key="synthetic",
                    base_url="https://upstream.invalid/v1",
                    api_version="2024-02-01",
                    http_client=async_borrowed,
                    max_retries=0,
                )
                if provider == "azure"
                else openai.AsyncOpenAI(
                    api_key="synthetic",
                    base_url="https://upstream.invalid/v1",
                    http_client=async_borrowed,
                    max_retries=0,
                )
            )
            async_result: Final = (
                await litellm.aembedding(input=["probe"], client=async_client, **options)
                if endpoint == "embeddings"
                else await litellm.aimage_generation(prompt="probe", client=async_client, **options)
            )
            assert not async_borrowed.is_closed
    response: Final = result if sync else async_result
    assert response.model_dump(exclude_none=True)["data"][0] == (
        {"object": "embedding", "index": 0, "embedding": [0.5]}
        if endpoint == "embeddings"
        else {"url": "https://upstream.invalid/image.png"}
    )
    require_received_headers(logging.upstream_response_capture, 200)
