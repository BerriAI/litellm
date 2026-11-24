import asyncio
import io
import json
from collections.abc import Awaitable, Iterator
from datetime import datetime
from queue import Empty, Queue
from typing import Final

import httpx
import pytest
import respx
from pydantic import BaseModel

import litellm
from litellm.images.streaming import AsyncImageStream, ImageStream
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import ImageResponse

MODEL: Final = "openai/gpt-image-1"
API_BASE: Final = "https://openai.test/v1"
PNG: Final = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
FINAL_IMAGE: Final = "RklOQUw="
USAGE: Final = {
    "total_tokens": 220,
    "input_tokens": 20,
    "output_tokens": 200,
    "input_tokens_details": {"text_tokens": 20, "image_tokens": 0},
}
ROUTES: Final = {"generation": ("generations", "image_generation"), "edit": ("edits", "image_edit")}


def _sse(event_type: str, **fields: object) -> bytes:
    payload: Final = {
        "type": event_type,
        "created_at": 1700000000,
        "size": "1024x1024",
        "quality": "low",
        "background": "opaque",
        "output_format": "png",
        **fields,
    }
    return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n".encode()


def _partial(prefix: str) -> bytes:
    return _sse(f"{prefix}.partial_image", b64_json="UA==", partial_image_index=0)


def _completed(prefix: str) -> bytes:
    return _sse(f"{prefix}.completed", b64_json=FINAL_IMAGE, usage=USAGE)


def _unstreamed() -> httpx.Response:
    return httpx.Response(200, json={"created": 1700000000, "data": [{"b64_json": FINAL_IMAGE}], "usage": USAGE})


def _provider(respx_mock: respx.MockRouter, kind: str, streamed_body: bytes | None) -> None:
    """Answers `stream=true` with `streamed_body` as an event stream, and anything else with a JSON image."""
    path, _ = ROUTES[kind]

    def answer(request: httpx.Request) -> httpx.Response:
        body: Final = request.content
        asks_to_stream: Final = b'"stream":true' in body.replace(b" ", b"") or b'name="stream"\r\n\r\ntrue' in body
        if asks_to_stream and streamed_body is not None:
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=streamed_body)
        return _unstreamed()

    respx_mock.post(f"{API_BASE}/images/{path}").mock(side_effect=answer)


class _Outcomes:
    def __init__(self) -> None:
        self.logged: Queue[tuple[str, object]] = Queue()

    async def next_outcome(self) -> tuple[str, object]:
        for _ in range(1000):
            try:
                return self.logged.get_nowait()
            except Empty:
                await asyncio.sleep(0.01)
        raise AssertionError("the call was never logged")

    async def has_no_more(self) -> bool:
        await asyncio.sleep(0.2)
        return self.logged.empty()


class _SyncCallLogger(CustomLogger):
    """Sync-only, since litellm sends a logger that overrides the async hooks nothing from a sync call."""

    def __init__(self, outcomes: _Outcomes) -> None:
        super().__init__()
        self.outcomes: Final = outcomes

    def log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        if kwargs.get("call_type") in ("image_generation", "image_edit"):
            self.outcomes.logged.put(("success", kwargs.get("response_cost")))

    def log_failure_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        if kwargs.get("call_type") in ("image_generation", "image_edit"):
            self.outcomes.logged.put(("failure", kwargs.get("exception")))


class _AsyncCallLogger(CustomLogger):
    def __init__(self, outcomes: _Outcomes) -> None:
        super().__init__()
        self.outcomes: Final = outcomes

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.outcomes.logged.put(("success", kwargs.get("response_cost")))

    async def async_log_failure_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.outcomes.logged.put(("failure", kwargs.get("exception")))


@pytest.fixture(autouse=True)
def httpx_transport_respx_can_script(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    yield
    litellm.in_memory_llm_clients_cache.flush_cache()


@pytest.fixture
def outcomes(monkeypatch: pytest.MonkeyPatch) -> _Outcomes:
    recorded: Final = _Outcomes()
    monkeypatch.setattr(litellm, "callbacks", [_SyncCallLogger(recorded), _AsyncCallLogger(recorded)])
    return recorded


def _call(kind: str, is_async: bool, **params: object) -> object:
    common: Final = {"model": MODEL, "api_key": "test", "api_base": API_BASE, **params}
    if kind == "generation":
        generate: Final = litellm.aimage_generation if is_async else litellm.image_generation
        return generate(prompt="A red circle", **common)
    edit: Final = litellm.aimage_edit if is_async else litellm.image_edit
    return edit(image=io.BytesIO(PNG), prompt="Make it blue", **common)


async def _settled(result: object) -> object:
    await asyncio.sleep(0)
    return await result if isinstance(result, Awaitable) else result


async def _events(stream: object) -> list[BaseModel]:
    if isinstance(stream, AsyncImageStream):
        return [event async for event in stream]
    assert isinstance(stream, ImageStream), type(stream)
    return list(stream)


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("kind", ["generation", "edit"])
async def test_stream_true_yields_the_provider_events_and_bills_like_the_unstreamed_call(
    respx_mock: respx.MockRouter, outcomes: _Outcomes, kind: str, is_async: bool
) -> None:
    prefix: Final = ROUTES[kind][1]
    _provider(respx_mock, kind, _partial(prefix) + _completed(prefix))

    await _settled(_call(kind, is_async))
    unstreamed_outcome: Final = await outcomes.next_outcome()
    events: Final = await _events(await _settled(_call(kind, is_async, stream=True, partial_images=1)))
    streamed_outcome: Final = await outcomes.next_outcome()

    assert [event.type for event in events] == [f"{prefix}.partial_image", f"{prefix}.completed"]
    assert events[-1].b64_json == FINAL_IMAGE
    assert unstreamed_outcome[0] == "success" and unstreamed_outcome[1] > 0
    assert streamed_outcome == unstreamed_outcome
    assert await outcomes.has_no_more()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["generation", "edit"])
async def test_a_stream_closed_after_its_first_event_is_still_billed_for_the_image(
    respx_mock: respx.MockRouter, outcomes: _Outcomes, kind: str
) -> None:
    prefix: Final = ROUTES[kind][1]
    _provider(respx_mock, kind, _partial(prefix) + _completed(prefix))

    await _settled(_call(kind, True))
    unstreamed_outcome: Final = await outcomes.next_outcome()
    stream: Final = await _settled(_call(kind, True, stream=True, partial_images=1))
    assert isinstance(stream, AsyncImageStream)
    first: Final = await stream.__anext__()
    await stream.aclose()

    assert first.type == f"{prefix}.partial_image"
    assert await outcomes.next_outcome() == unstreamed_outcome
    assert await outcomes.has_no_more()


class _HeldEventStream(httpx.AsyncByteStream):
    """An event stream whose bytes arrive only once `release` is set, so a read can be cancelled mid-flight."""

    def __init__(self, body: bytes) -> None:
        self.body: Final = body
        self.read_started: Final = asyncio.Event()
        self.release: Final = asyncio.Event()

    async def __aiter__(self):
        self.read_started.set()
        await self.release.wait()
        yield self.body


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["generation", "edit"])
async def test_a_stream_whose_first_read_is_cancelled_is_still_billed_for_the_image(
    respx_mock: respx.MockRouter, outcomes: _Outcomes, kind: str
) -> None:
    path, prefix = ROUTES[kind]
    _provider(respx_mock, kind, None)
    await _settled(_call(kind, True))
    unstreamed_outcome: Final = await outcomes.next_outcome()
    held: Final = _HeldEventStream(_partial(prefix) + _completed(prefix))
    respx_mock.post(f"{API_BASE}/images/{path}").mock(
        return_value=httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=held)
    )

    stream: Final = await _settled(_call(kind, True, stream=True, partial_images=1))
    assert isinstance(stream, AsyncImageStream)
    first_read: Final = asyncio.ensure_future(stream.__anext__())
    await held.read_started.wait()
    first_read.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_read
    held.release.set()
    await stream.aclose()

    assert await outcomes.next_outcome() == unstreamed_outcome
    assert await outcomes.has_no_more()


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("kind", ["generation", "edit"])
async def test_a_stream_that_ends_without_a_completed_event_raises_and_is_logged_as_a_failure(
    respx_mock: respx.MockRouter, outcomes: _Outcomes, kind: str, is_async: bool
) -> None:
    prefix: Final = ROUTES[kind][1]
    _provider(respx_mock, kind, _partial(prefix))

    with pytest.raises(litellm.BadGatewayError, match="ended before its completed event"):
        await _events(await _settled(_call(kind, is_async, stream=True)))
    status, exception = await outcomes.next_outcome()

    assert status == "failure"
    assert "ended before its completed event" in str(exception)
    assert await outcomes.has_no_more()


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
async def test_an_error_event_inside_an_edit_stream_raises_and_is_logged_as_a_failure(
    respx_mock: respx.MockRouter, outcomes: _Outcomes, is_async: bool
) -> None:
    error_frame: Final = b'event: error\ndata: {"type": "error", "error": {"message": "moderation_blocked"}}\n\n'
    _provider(respx_mock, "edit", _partial("image_edit") + error_frame)

    with pytest.raises(litellm.InternalServerError, match="moderation_blocked"):
        await _events(await _settled(_call("edit", is_async, stream=True)))
    status, exception = await outcomes.next_outcome()

    assert status == "failure"
    assert "moderation_blocked" in str(exception)
    assert await outcomes.has_no_more()


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("kind", ["generation", "edit"])
async def test_a_provider_that_answers_stream_true_with_json_still_returns_the_image_and_its_cost(
    respx_mock: respx.MockRouter, outcomes: _Outcomes, kind: str, is_async: bool
) -> None:
    _provider(respx_mock, kind, None)

    response: Final = await _settled(_call(kind, is_async, stream=True))
    status, cost = await outcomes.next_outcome()

    assert isinstance(response, ImageResponse)
    assert response.data[0].b64_json == FINAL_IMAGE
    assert status == "success" and cost > 0
    assert await outcomes.has_no_more()
