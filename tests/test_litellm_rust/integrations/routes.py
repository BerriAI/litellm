import asyncio
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Final, Literal

import litellm
from tests.test_litellm_rust.callback_recorder import HookEvent, RecordingLogger
from tests.test_litellm_rust.contracts import (
    MESSAGES,
    MESSAGES_EVENTS,
    MESSAGES_MODEL,
    MESSAGES_RESPONSE,
    OCR_RESPONSE,
    call_aocr,
    call_ocr,
)
from tests.test_litellm_rust.recording_server import RecordingServer, ResponseSpec

RouteName = Literal["ocr-sync", "ocr-async", "messages", "messages-stream"]


@dataclass(frozen=True, slots=True)
class Route:
    name: RouteName
    call_type: str
    provider_model: str
    provider_response: Mapping[str, object]
    expected_text: str
    expected_cost: float
    fires_async_hooks: bool

    async def invoke(self, server: RecordingServer, **kwargs: object) -> object:
        match self.name:
            case "ocr-sync":
                return await asyncio.to_thread(call_ocr, server, **kwargs)
            case "ocr-async":
                return await call_aocr(server, **kwargs)
            case "messages":
                return await _call_messages(server, **kwargs)
            case "messages-stream":
                stream: Final = await self.open_stream(server, **kwargs)
                return [chunk async for chunk in stream]

    async def open_stream(self, server: RecordingServer, **kwargs: object) -> AsyncIterator[object]:
        if self.name != "messages-stream":
            raise ValueError(f"{self.name} is not a streaming route")
        stream: Final = await _call_messages(server, stream=True, **kwargs)
        if not isinstance(stream, AsyncIterator):
            raise TypeError(f"Expected async stream, got {type(stream).__name__}")
        return stream

    def response_text(self, response: object) -> str:
        match self.name:
            case "ocr-sync" | "ocr-async":
                pages: Final = getattr(response, "pages", None)
                if not isinstance(pages, list) or not pages:
                    raise TypeError(f"Expected OCR response pages, got {type(response).__name__}")
                markdown: Final = getattr(pages[0], "markdown", None)
                if not isinstance(markdown, str):
                    raise TypeError(f"Expected OCR markdown, got {type(markdown).__name__}")
                return markdown
            case "messages":
                if not isinstance(response, Mapping):
                    raise TypeError(f"Expected Messages mapping, got {type(response).__name__}")
                content: Final = response.get("content")
                if not isinstance(content, list) or not content or not isinstance(content[0], Mapping):
                    raise TypeError("Expected Messages text content")
                text: Final = content[0].get("text")
                if not isinstance(text, str):
                    raise TypeError(f"Expected Messages text, got {type(text).__name__}")
                return text
            case "messages-stream":
                raise ValueError("Streaming response text is assembled by the stream consumer")


async def _call_messages(server: RecordingServer, **kwargs: object) -> object:
    return await litellm.anthropic.messages.acreate(
        model=MESSAGES_MODEL,
        messages=MESSAGES,
        max_tokens=64,
        api_key="test-key",
        api_base=server.base_url,
        **kwargs,
    )


MESSAGES_COST: Final = 5 * 3e-06 + 4 * 1.5e-05
OCR_COST: Final = 0.004

OCR_SYNC: Final = Route("ocr-sync", "ocr", "mistral-ocr-latest", OCR_RESPONSE, "native OCR response", OCR_COST, False)
OCR_ASYNC: Final = Route("ocr-async", "aocr", "mistral-ocr-latest", OCR_RESPONSE, "native OCR response", OCR_COST, True)
MESSAGES_ROUTE: Final = Route(
    "messages",
    "anthropic_messages",
    "claude-sonnet-4-5-20250929",
    MESSAGES_RESPONSE,
    "Hello from native Messages",
    MESSAGES_COST,
    True,
)
MESSAGES_STREAM: Final = Route(
    "messages-stream",
    "anthropic_messages",
    "claude-sonnet-4-5-20250929",
    MESSAGES_RESPONSE,
    "Hello from native Messages",
    MESSAGES_COST,
    True,
)
ALL_ROUTES: Final = (OCR_SYNC, OCR_ASYNC, MESSAGES_ROUTE, MESSAGES_STREAM)
ASYNC_ROUTES: Final = tuple(route for route in ALL_ROUTES if route.fires_async_hooks)
NON_STREAM_ASYNC_ROUTES: Final = (OCR_ASYNC, MESSAGES_ROUTE)


def route_id(route: Route) -> str:
    return route.name


def provider_response(route: Route) -> ResponseSpec:
    return ResponseSpec(
        body=route.provider_response,
        events=MESSAGES_EVENTS if route.name == "messages-stream" else (),
    )


async def wait_for_callback(
    route: Route,
    recorder: RecordingLogger,
    outcome: Literal["success", "failure"] = "success",
    count: int = 1,
) -> tuple[HookEvent, ...]:
    event: Final = f"{'async_' if route.fires_async_hooks else ''}log_{outcome}_event"
    if route.fires_async_hooks:
        return await recorder.wait_for_async(event, count=count)
    return recorder.wait_for(event, count=count)
