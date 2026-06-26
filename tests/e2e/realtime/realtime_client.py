"""Client for realtime e2e tests over the proxy's /v1/realtime websocket.

The proxy normalizes every provider's realtime stream into the OpenAI GA event
schema toward the client, so one GA-speaking websocket client validates every
provider and only the model alias changes. Every other e2e suite is HTTP-only;
this is the one suite that opens a websocket, using websockets.sync so it stays
synchronous like the rest of tests/e2e. Sent and received events are pydantic
models, matching the suite's no-raw-dicts rule.
"""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, TypeVar
from urllib.parse import urlencode

import pytest
from pydantic import BaseModel, ConfigDict
from websockets.sync.client import connect
from websockets.sync.connection import Connection

from e2e_config import PROXY_BASE_URL
from e2e_gateway import Gateway, build_gateway

_M = TypeVar("_M", bound=BaseModel)


def _ws_base_url() -> str:
    for scheme, ws_scheme in (("https://", "wss://"), ("http://", "ws://")):
        if PROXY_BASE_URL.startswith(scheme):
            return ws_scheme + PROXY_BASE_URL[len(scheme) :]
    return PROXY_BASE_URL


def realtime_ws_url(model: str) -> str:
    return f"{_ws_base_url()}/v1/realtime?{urlencode({'model': model})}"


@dataclass(frozen=True, slots=True)
class RealtimeProvider:
    id: str
    model: str


PROVIDERS = (
    RealtimeProvider("openai", "openai-realtime"),
    RealtimeProvider("azure", "azure-realtime"),
    RealtimeProvider("gemini", "gemini-realtime"),
    RealtimeProvider("vertex_ai", "vertex-realtime"),
    # RealtimeProvider("bedrock", "bedrock-realtime"), # TODO: Enable this when Bedrock is passing
    RealtimeProvider("xai", "xai-realtime"),
)


def skip_if_unconfigured(
    provider: RealtimeProvider, configured: frozenset[str]
) -> None:
    if provider.model not in configured:
        pytest.skip(f"{provider.model} not configured on proxy")


# ---- sent events -------------------------------------------------------


class JsonSchemaProperty(BaseModel):
    type: str


class JsonSchema(BaseModel):
    type: str = "object"
    properties: dict[str, JsonSchemaProperty]
    required: list[str]


class FunctionTool(BaseModel):
    type: str = "function"
    name: str
    description: str
    parameters: JsonSchema


class SessionConfig(BaseModel):
    modalities: list[str] = ["text"]
    instructions: str | None = None
    tools: list[FunctionTool] | None = None
    tool_choice: str | None = None


class SessionUpdate(BaseModel):
    type: str = "session.update"
    session: SessionConfig


class InputTextContent(BaseModel):
    type: str = "input_text"
    text: str


class MessageItem(BaseModel):
    type: str = "message"
    role: str = "user"
    content: list[InputTextContent]


class FunctionCallOutputItem(BaseModel):
    type: str = "function_call_output"
    call_id: str
    output: str


class ConversationItemCreate(BaseModel):
    type: str = "conversation.item.create"
    item: MessageItem | FunctionCallOutputItem


class ResponseCreate(BaseModel):
    type: str = "response.create"


def user_message(text: str) -> ConversationItemCreate:
    return ConversationItemCreate(
        item=MessageItem(content=[InputTextContent(text=text)])
    )


# ---- received events ---------------------------------------------------


class ServerEnvelope(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str = ""


class DeltaEvent(BaseModel):
    type: str
    delta: str = ""


class FunctionCallArgumentsDone(BaseModel):
    type: str
    call_id: str
    arguments: str


class ContentPart(BaseModel):
    model_config = ConfigDict(extra="allow")
    text: str | None = None
    transcript: str | None = None


class OutputItem(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str | None = None
    name: str | None = None
    call_id: str | None = None
    content: list[ContentPart] | None = None


class OutputItemDone(BaseModel):
    type: str
    item: OutputItem


class ResponsePayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    usage: dict[str, Any] | None = None
    output: list[OutputItem] | None = None


class ResponseDone(BaseModel):
    type: str
    response: ResponsePayload


@dataclass(frozen=True, slots=True)
class ReceivedEvent:
    type: str
    payload: str


def events_of_type(
    events: tuple[ReceivedEvent, ...], event_type: str
) -> tuple[ReceivedEvent, ...]:
    return tuple(e for e in events if e.type == event_type)


def parse_last(
    events: tuple[ReceivedEvent, ...], event_type: str, model: type[_M]
) -> _M | None:
    matches = events_of_type(events, event_type)
    return model.model_validate_json(matches[-1].payload) if matches else None


_TEXT_DELTA_TYPES = (
    # beta protocol (OpenAI-Beta: realtime=v1)
    "response.text.delta",
    "response.audio_transcript.delta",
    # GA protocol (default toward the proxy)
    "response.output_text.delta",
    "response.output_audio_transcript.delta",
)


def _text_from_response_done(events: tuple[ReceivedEvent, ...]) -> str:
    done = parse_last(events, "response.done", ResponseDone)
    if done is None or not done.response.output:
        return ""
    parts: list[str] = []
    for item in done.response.output:
        if item.type != "message":
            continue
        for content in item.content or []:
            if content.text:
                parts.append(content.text)
            elif content.transcript:
                parts.append(content.transcript)
    return "".join(parts)


def transcript(events: tuple[ReceivedEvent, ...]) -> str:
    for delta_type in _TEXT_DELTA_TYPES:
        text = "".join(
            DeltaEvent.model_validate_json(e.payload).delta
            for e in events_of_type(events, delta_type)
        )
        if text:
            return text
    return _text_from_response_done(events)


def function_call_item(events: tuple[ReceivedEvent, ...]) -> OutputItem | None:
    for event in events_of_type(events, "response.output_item.done"):
        item = OutputItemDone.model_validate_json(event.payload).item
        if item.type == "function_call":
            return item
    return None


# ---- session + client --------------------------------------------------


def _as_text(message: str | bytes) -> str:
    return message.decode("utf-8") if isinstance(message, bytes) else message


@dataclass(frozen=True, slots=True)
class RealtimeSession:
    connection: Connection

    def send(self, event: BaseModel) -> None:
        self.connection.send(event.model_dump_json(by_alias=True, exclude_none=True))

    def collect_until(
        self, stop_type: str, *, timeout: float
    ) -> tuple[ReceivedEvent, ...]:
        deadline = time.monotonic() + timeout
        collected: list[ReceivedEvent] = []
        while time.monotonic() < deadline:
            try:
                text = _as_text(
                    self.connection.recv(timeout=deadline - time.monotonic())
                )
            except TimeoutError:
                break
            event = ReceivedEvent(
                type=ServerEnvelope.model_validate_json(text).type, payload=text
            )
            collected.append(event)
            if event.type == stop_type:
                return tuple(collected)
        raise TimeoutError(
            f"no {stop_type!r} within {timeout}s; got {[e.type for e in collected]}"
        )


@dataclass(frozen=True, slots=True)
class RealtimeClient:
    gateway: Gateway

    def configured_models(self) -> frozenset[str]:
        return frozenset(
            entry.model_name
            for entry in self.gateway.model_info()
            if entry.model_info.mode == "realtime"
        )

    @contextmanager
    def connect(
        self, *, key: str, model: str, timeout: float = 15.0
    ) -> Generator[RealtimeSession, None, None]:
        with connect(
            realtime_ws_url(model),
            additional_headers={"Authorization": f"Bearer {key}"},
            open_timeout=timeout,
        ) as connection:
            yield RealtimeSession(connection=connection)


def build_client() -> RealtimeClient:
    return RealtimeClient(gateway=build_gateway())
