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
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, TypeVar
from urllib.parse import urlencode

from pydantic import BaseModel, ConfigDict
from websockets.sync.client import connect
from websockets.sync.connection import Connection

from e2e_config import PROXY_BASE_URL, unique_marker
from e2e_gateway import Gateway, build_gateway
from models import LiteLLMParamsBody

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
    """A realtime provider the suite exercises. `litellm_params` is the deployment
    the suite registers through /model/new (the gateway resolves the os.environ/*
    credential refs), so the suite is self-contained and never depends on a static
    gateway model_list. Every provider here is provisioned and asserted: per
    tests/e2e/CLAUDE.md the suite never skips a provider, so a provider whose
    credentials or upstream realtime model are missing on the gateway is a hard
    failure, not a skip."""

    id: str
    alias: str
    litellm_params: LiteLLMParamsBody


PROVIDERS = (
    RealtimeProvider(
        "openai",
        "openai-realtime",
        LiteLLMParamsBody(
            model="openai/gpt-realtime-2",
            api_key="os.environ/OPENAI_API_KEY",
        ),
    ),
    RealtimeProvider(
        "azure",
        "azure-realtime",
        LiteLLMParamsBody(
            model="azure/gpt-realtime",
            api_key="os.environ/AZURE_API_KEY",
            api_version="2025-08-28",
            realtime_protocol="GA",
        ),
    ),
    RealtimeProvider(
        "gemini",
        "gemini-realtime",
        LiteLLMParamsBody(
            model="gemini/gemini-3.1-flash-live-preview",
            api_key="os.environ/GEMINI_API_KEY",
        ),
    ),
    RealtimeProvider(
        "vertex_ai",
        "vertex-realtime",
        LiteLLMParamsBody(
            model="vertex_ai/gemini-live-2.5-flash-preview-native-audio-09-2025",
            vertex_location="us-central1",
            vertex_credentials="os.environ/VERTEXAI_CREDENTIALS",
        ),
    ),
    # RealtimeProvider("bedrock", "bedrock-realtime", ...) # TODO: Enable when Bedrock is passing
    # RealtimeProvider(
    #     "xai",
    #     "xai-realtime",
    #     LiteLLMParamsBody(
    #         model="xai/grok-4-1-fast",
    #         api_key="os.environ/XAI_API_KEY",
    #     ),
    # ),  # TODO: Enable once xai Grok Voice realtime is passing end-to-end here
)


def realtime_model(provider: RealtimeProvider, provisioned: Mapping[str, str]) -> str:
    """Return the provisioned deployment name for this provider. Every provider in
    PROVIDERS is provisioned at session start, so a missing entry is a harness bug,
    never an environment skip - the suite hard-fails instead (see tests/e2e/CLAUDE.md)."""
    model = provisioned.get(provider.id)
    assert model is not None, (
        f"{provider.id} was not provisioned; the realtime_models fixture is broken"
    )
    return model


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

    def provision(self, provider: RealtimeProvider) -> tuple[str, str]:
        """Register this provider's realtime deployment through /model/new and return
        (model_name, model_id). The name is marker-unique so it never collides with a
        same-named deployment already on the shared proxy, and mode=realtime makes it
        show up as a realtime model on /model/info. add_deployment runs synchronously,
        so the deployment is connectable as soon as this returns."""
        model_name = f"{provider.alias}-{unique_marker()}"
        model_id = self.gateway.create_model(
            model_name, provider.litellm_params, mode="realtime"
        )
        return model_name, model_id

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
