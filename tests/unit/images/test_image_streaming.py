from __future__ import annotations

import asyncio
import datetime
import json
import threading
from collections.abc import Generator
from typing import Any, Final

import httpx
import openai
import pytest
from fastapi import FastAPI
from openai.types.image_edit_stream_event import ImageEditCompletedEvent, ImageEditPartialImageEvent
from openai.types.image_gen_stream_event import ImageGenCompletedEvent, ImageGenPartialImageEvent

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.image_streaming import (
    AsyncImageSSEStream,
    SyncImageSSEStream,
    format_image_sse_event,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.utils import ImageResponse

_SAMPLE_USAGE: Final[dict[str, object]] = {
    "input_tokens": 12,
    "input_tokens_details": {"image_tokens": 0, "text_tokens": 12},
    "output_tokens": 1056,
    "total_tokens": 1068,
}


def _sse_body(events: list[tuple[str, dict[str, object]]]) -> bytes:
    frames: Final = [
        f"event: {event_name}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n" for event_name, payload in events
    ]
    return "".join(frames).encode("utf-8")


def _image_generation_sse_bytes() -> bytes:
    return _sse_body(
        [
            (
                "image_generation.partial_image",
                {
                    "type": "image_generation.partial_image",
                    "partial_image_index": 0,
                    "b64_json": "cGFydGlhbA==",
                    "background": "opaque",
                    "created_at": 1700000000,
                    "output_format": "png",
                    "quality": "high",
                    "size": "1024x1024",
                },
            ),
            (
                "image_generation.completed",
                {
                    "type": "image_generation.completed",
                    "b64_json": "ZmluYWw=",
                    "background": "opaque",
                    "created_at": 1700000001,
                    "output_format": "png",
                    "quality": "high",
                    "size": "1024x1024",
                    "usage": _SAMPLE_USAGE,
                },
            ),
        ]
    )


def _image_edit_sse_bytes() -> bytes:
    return _sse_body(
        [
            (
                "image_edit.partial_image",
                {
                    "type": "image_edit.partial_image",
                    "partial_image_index": 0,
                    "b64_json": "ZWRpdF9wYXJ0aWFs",
                    "background": "opaque",
                    "created_at": 1700000000,
                    "output_format": "png",
                    "quality": "high",
                    "size": "1024x1024",
                },
            ),
            (
                "image_edit.completed",
                {
                    "type": "image_edit.completed",
                    "b64_json": "ZWRpdF9maW5hbA==",
                    "background": "opaque",
                    "created_at": 1700000001,
                    "output_format": "png",
                    "quality": "high",
                    "size": "1024x1024",
                    "usage": _SAMPLE_USAGE,
                },
            ),
        ]
    )


def _non_stream_image_json(b64_json: str) -> bytes:
    return json.dumps(
        {
            "created": 1700000001,
            "data": [{"b64_json": b64_json}],
            "background": "opaque",
            "output_format": "png",
            "quality": "high",
            "size": "1024x1024",
            "usage": _SAMPLE_USAGE,
        }
    ).encode("utf-8")


class _RecordingLogger(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.successes: list[dict[str, Any]] = []
        self.async_successes: list[dict[str, Any]] = []
        self.failures: list[dict[str, Any]] = []
        self.async_failures: list[dict[str, Any]] = []
        self.sync_success_event: Final = threading.Event()
        self.async_success_event: Final = asyncio.Event()
        self.async_failure_event: Final = asyncio.Event()

    def log_success_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None:
        self.successes.append({"kwargs": kwargs, "response_obj": response_obj})
        self.sync_success_event.set()

    async def async_log_success_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None:
        self.async_successes.append({"kwargs": kwargs, "response_obj": response_obj})
        self.async_success_event.set()

    def log_failure_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None:
        self.failures.append({"kwargs": kwargs, "response_obj": response_obj})

    async def async_log_failure_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None:
        self.async_failures.append({"kwargs": kwargs, "response_obj": response_obj})
        self.async_failure_event.set()


@pytest.fixture
def recording_logger(monkeypatch: pytest.MonkeyPatch) -> Generator[_RecordingLogger, None, None]:
    litellm.logging_callback_manager._reset_all_callbacks()
    logger: Final = _RecordingLogger()
    monkeypatch.setattr(litellm, "callbacks", [logger])
    try:
        yield logger
    finally:
        litellm.logging_callback_manager._reset_all_callbacks()


@pytest.mark.asyncio
async def test_aimage_generation_streaming_and_cost_parity(recording_logger: _RecordingLogger) -> None:
    captured_requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body: Final = json.loads(request.content.decode("utf-8"))
        captured_requests.append(body)
        if body.get("stream") is True:
            return httpx.Response(
                status_code=200,
                headers={"content-type": "text/event-stream", "x-request-id": "req_gen_stream"},
                content=_image_generation_sse_bytes(),
            )
        return httpx.Response(
            status_code=200,
            headers={"content-type": "application/json", "x-request-id": "req_gen_sync"},
            content=_non_stream_image_json("ZmluYWw="),
        )

    transport: Final = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        openai_client: Final = openai.AsyncOpenAI(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            http_client=http_client,
        )

        non_stream_response: Final = await litellm.aimage_generation(
            prompt="a sunset over water",
            model="gpt-image-1.5",
            api_key="test-key",
            client=openai_client,
        )
        assert isinstance(non_stream_response, ImageResponse)
        await asyncio.wait_for(recording_logger.async_success_event.wait(), timeout=2.0)
        recording_logger.async_success_event.clear()
        non_stream_cost: Final = recording_logger.async_successes[-1]["kwargs"]["response_cost"]
        assert non_stream_cost > 0

        stream: Any = await litellm.aimage_generation(
            prompt="a sunset over water",
            model="gpt-image-1.5",
            stream=True,
            partial_images=1,
            api_key="test-key",
            client=openai_client,
        )
        assert isinstance(stream, openai.AsyncStream)
        events: Final = [event async for event in stream]
        await asyncio.wait_for(recording_logger.async_success_event.wait(), timeout=2.0)

    assert len(captured_requests) == 2
    assert captured_requests[1]["stream"] is True
    assert captured_requests[1]["partial_images"] == 1
    assert "extra_body" not in captured_requests[1]

    assert len(events) == 2
    assert isinstance(events[0], ImageGenPartialImageEvent)
    assert events[0].b64_json == "cGFydGlhbA=="
    assert isinstance(events[1], ImageGenCompletedEvent)
    assert events[1].b64_json == "ZmluYWw="

    assert len(recording_logger.async_successes) == 2
    stream_record: Final = recording_logger.async_successes[-1]
    assert isinstance(stream_record["response_obj"], ImageResponse)
    assert stream_record["response_obj"].data[0].b64_json == "ZmluYWw="
    assert stream_record["kwargs"]["response_cost"] == pytest.approx(non_stream_cost)


def test_sync_image_generation_streaming(recording_logger: _RecordingLogger) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body: Final = json.loads(request.content.decode("utf-8"))
        assert body["stream"] is True
        return httpx.Response(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            content=_image_generation_sse_bytes(),
        )

    transport: Final = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as http_client:
        openai_client: Final = openai.OpenAI(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            http_client=http_client,
        )
        stream: Any = litellm.image_generation(
            prompt="a mountain lake",
            model="gpt-image-1.5",
            stream=True,
            api_key="test-key",
            client=openai_client,
        )
        assert isinstance(stream, openai.Stream)
        events: Final = list(stream)

    assert recording_logger.sync_success_event.wait(timeout=2.0)
    assert len(events) == 2
    assert isinstance(events[0], ImageGenPartialImageEvent)
    assert isinstance(events[1], ImageGenCompletedEvent)
    assert len(recording_logger.successes) == 1
    assert recording_logger.successes[0]["kwargs"]["response_cost"] > 0


@pytest.mark.asyncio
async def test_aimage_edit_streaming_and_cost_parity(recording_logger: _RecordingLogger) -> None:
    captured_bodies: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        raw_body: Final = request.read()
        captured_bodies.append(raw_body)
        if b'name="stream"\r\n\r\ntrue' in raw_body:
            return httpx.Response(
                status_code=200,
                headers={"content-type": "text/event-stream", "x-request-id": "req_edit_stream"},
                content=_image_edit_sse_bytes(),
            )
        return httpx.Response(
            status_code=200,
            headers={"content-type": "application/json", "x-request-id": "req_edit_sync"},
            content=_non_stream_image_json("ZWRpdF9maW5hbA=="),
        )

    transport: Final = httpx.MockTransport(handler)
    async_handler: Final = AsyncHTTPHandler(transport=transport)

    try:
        non_stream_response: Final = await litellm.aimage_edit(
            image=b"\x89PNG\r\n\x1a\nfake",
            prompt="make the sky purple",
            model="gpt-image-1.5",
            api_key="test-key",
            client=async_handler,
        )
        assert isinstance(non_stream_response, ImageResponse)
        await asyncio.wait_for(recording_logger.async_success_event.wait(), timeout=2.0)
        recording_logger.async_success_event.clear()
        non_stream_cost: Final = recording_logger.async_successes[-1]["kwargs"]["response_cost"]
        assert non_stream_cost > 0

        stream: Any = await litellm.aimage_edit(
            image=b"\x89PNG\r\n\x1a\nfake",
            prompt="make the sky purple",
            model="gpt-image-1.5",
            stream=True,
            partial_images=1,
            api_key="test-key",
            client=async_handler,
        )
        assert isinstance(stream, openai.AsyncStream)
        events: Final = [event async for event in stream]
        await asyncio.wait_for(recording_logger.async_success_event.wait(), timeout=2.0)
    finally:
        await async_handler.client.aclose()

    assert len(captured_bodies) == 2
    assert b'name="stream"\r\n\r\ntrue' in captured_bodies[1]
    assert b'name="partial_images"\r\n\r\n1' in captured_bodies[1]

    assert len(events) == 2
    assert isinstance(events[0], ImageEditPartialImageEvent)
    assert events[0].b64_json == "ZWRpdF9wYXJ0aWFs"
    assert isinstance(events[1], ImageEditCompletedEvent)
    assert events[1].b64_json == "ZWRpdF9maW5hbA=="

    assert len(recording_logger.async_successes) == 2
    stream_record: Final = recording_logger.async_successes[-1]
    assert isinstance(stream_record["response_obj"], ImageResponse)
    assert stream_record["response_obj"].data[0].b64_json == "ZWRpdF9maW5hbA=="
    assert stream_record["kwargs"]["response_cost"] == pytest.approx(non_stream_cost)


def test_sync_image_edit_streaming(recording_logger: _RecordingLogger) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raw_body: Final = request.read()
        assert b'name="stream"\r\n\r\ntrue' in raw_body
        return httpx.Response(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            content=_image_edit_sse_bytes(),
        )

    transport: Final = httpx.MockTransport(handler)
    sync_handler: Final = HTTPHandler()
    sync_handler.client = httpx.Client(transport=transport)
    try:
        stream: Any = litellm.image_edit(
            image=b"\x89PNG\r\n\x1a\nfake",
            prompt="add clouds",
            model="gpt-image-1.5",
            stream=True,
            api_key="test-key",
            client=sync_handler,
        )
        assert isinstance(stream, openai.Stream)
        events: Final = list(stream)
    finally:
        sync_handler.client.close()

    assert recording_logger.sync_success_event.wait(timeout=2.0)
    assert len(events) == 2
    assert isinstance(events[0], ImageEditPartialImageEvent)
    assert isinstance(events[1], ImageEditCompletedEvent)
    assert len(recording_logger.successes) == 1
    assert recording_logger.successes[0]["kwargs"]["response_cost"] > 0


@pytest.mark.asyncio
async def test_image_edit_stream_error_and_close_cleanup(recording_logger: _RecordingLogger) -> None:
    error_sse: Final = _sse_body(
        [
            (
                "image_edit.partial_image",
                {
                    "type": "image_edit.partial_image",
                    "partial_image_index": 0,
                    "b64_json": "ZWRpdF9wYXJ0aWFs",
                    "background": "opaque",
                    "created_at": 1700000000,
                    "output_format": "png",
                    "quality": "high",
                    "size": "1024x1024",
                },
            ),
            (
                "error",
                {"error": {"message": "image content filter triggered"}},
            ),
        ]
    )

    transport: Final = httpx.MockTransport(
        lambda request: httpx.Response(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            content=error_sse,
        )
    )
    async_handler: Final = AsyncHTTPHandler(transport=transport)
    try:
        stream: Any = await litellm.aimage_edit(
            image=b"\x89PNG\r\n\x1a\nfake",
            prompt="bad prompt",
            model="gpt-image-1.5",
            stream=True,
            api_key="test-key",
            client=async_handler,
        )
        received: list[object] = []
        with pytest.raises(litellm.APIConnectionError, match="image content filter triggered"):
            async for event in stream:
                received.append(event)
        assert len(received) == 1
        assert stream.response.is_closed is True
        await asyncio.wait_for(recording_logger.async_failure_event.wait(), timeout=2.0)
    finally:
        await async_handler.client.aclose()

    assert len(recording_logger.async_successes) == 0
    assert len(recording_logger.async_failures) == 1


@pytest.mark.asyncio
async def test_proxy_image_endpoints_emit_named_sse_without_done(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    from litellm.proxy.image_endpoints.endpoints import router as images_router

    app: Final = FastAPI()
    app.include_router(images_router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(api_key="sk-test", user_id="u-test")

    observed_route_data: list[dict[str, Any]] = []

    async def fake_route_request(
        *,
        data: dict[str, Any],
        route_type: str,
        **kwargs: Any,
    ) -> Any:
        observed_route_data.append({"route_type": route_type, "data": dict(data)})
        if route_type == "aimage_generation":
            raw_stream: Final = AsyncImageSSEStream(
                httpx.Response(
                    status_code=200,
                    headers={"content-type": "text/event-stream"},
                    content=_image_generation_sse_bytes(),
                )
            )
        else:
            raw_stream = AsyncImageSSEStream(
                httpx.Response(
                    status_code=200,
                    headers={"content-type": "text/event-stream"},
                    content=_image_edit_sse_bytes(),
                )
            )

        async def _call() -> AsyncImageSSEStream:
            return raw_stream

        return _call()

    monkeypatch.setattr("litellm.proxy.image_endpoints.endpoints.route_request", fake_route_request)
    monkeypatch.setattr("litellm.proxy.common_request_processing.route_request", fake_route_request)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        gen_resp: Final = await client.post(
            "/v1/images/generations",
            json={"model": "gpt-image-1.5", "prompt": "draw a cat", "stream": True, "partial_images": 1},
        )
        assert gen_resp.status_code == 200
        assert gen_resp.headers["content-type"].startswith("text/event-stream")
        assert "event: image_generation.partial_image\ndata: " in gen_resp.text
        assert "event: image_generation.completed\ndata: " in gen_resp.text
        assert "[DONE]" not in gen_resp.text

        edit_resp: Final = await client.post(
            "/v1/images/edits",
            data={"model": "gpt-image-1.5", "prompt": "make it neon", "stream": "true", "partial_images": "1"},
            files={"image": ("input.png", b"\x89PNG\r\n\x1a\nfake", "image/png")},
        )
        assert edit_resp.status_code == 200
        assert edit_resp.headers["content-type"].startswith("text/event-stream")
        assert "event: image_edit.partial_image\ndata: " in edit_resp.text
        assert "event: image_edit.completed\ndata: " in edit_resp.text
        assert "[DONE]" not in edit_resp.text

    assert observed_route_data[0]["route_type"] == "aimage_generation"
    assert observed_route_data[0]["data"]["stream"] is True
    assert observed_route_data[1]["route_type"] == "aimage_edit"
    assert observed_route_data[1]["data"]["stream"] is True
    assert observed_route_data[1]["data"]["partial_images"] == 1


def test_format_and_sync_edit_sse_stream_parses_events() -> None:
    sse_response: Final = httpx.Response(
        status_code=200,
        headers={"content-type": "text/event-stream"},
        content=_image_edit_sse_bytes(),
    )
    events: Final = list(SyncImageSSEStream(sse_response))
    assert len(events) == 2
    formatted_partial: Final = format_image_sse_event(events[0])
    formatted_completed: Final = format_image_sse_event(events[1])
    assert formatted_partial is not None
    assert formatted_partial.startswith("event: image_edit.partial_image\ndata: ")
    assert formatted_completed is not None
    assert formatted_completed.startswith("event: image_edit.completed\ndata: ")
    assert format_image_sse_event({"not": "an event"}) is None
