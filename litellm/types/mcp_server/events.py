from collections.abc import Mapping
from enum import IntEnum
from typing import Final, Literal, TypeAlias

from mcp.types import RequestParams
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    TypeAdapter,
    model_serializer,
)
from pydantic.alias_generators import to_camel

EVENTS_EXTENSION: Final = "io.modelcontextprotocol/events"
EVENTS_PROTOCOL_VERSION: Final = "2026-07-28"
SUBSCRIPTION_ID_META_KEY: Final = "io.modelcontextprotocol/subscriptionId"

DeliveryMode: TypeAlias = Literal["poll", "push", "webhook"]
DELIVERY_MODES: Final[tuple[DeliveryMode, ...]] = ("poll", "push", "webhook")
JsonObject: TypeAlias = Mapping[str, JsonValue]
_JSON_OBJECT: Final = TypeAdapter[dict[str, JsonValue]](dict[str, JsonValue])


class EventsErrorCode(IntEnum):
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    METHOD_NOT_FOUND = -32601
    HEADER_MISMATCH = -32020
    NOT_FOUND = -32023
    FORBIDDEN = -32024
    RESOURCE_EXHAUSTED = -32025
    UNSUPPORTED = -32026
    CALLBACK_ENDPOINT_ERROR = -32027


CallbackFailure: TypeAlias = Literal[
    "connection_refused", "timeout", "tls_error", "http_4xx", "http_5xx", "challenge_failed"
]


class _Wire(BaseModel):
    model_config = ConfigDict(frozen=True, alias_generator=to_camel, populate_by_name=True, extra="ignore")


class _WireParams(RequestParams):
    model_config = ConfigDict(frozen=True, alias_generator=to_camel, populate_by_name=True, extra="ignore")


class EventType(_Wire):
    name: str = Field(min_length=1)
    description: str | None = None
    delivery: tuple[DeliveryMode, ...] = Field(min_length=1)
    input_schema: JsonObject = Field(default_factory=lambda: {"type": "object"})
    payload_schema: JsonObject = Field(default_factory=lambda: {"type": "object"})
    meta: JsonObject | None = Field(default=None, alias="_meta")


class ListEventsParams(_WireParams):
    cursor: str | None = None


class ListEventsResult(_Wire):
    events: tuple[EventType, ...] = ()
    next_cursor: str | None = None
    meta: JsonObject | None = Field(default=None, alias="_meta")


class EventOccurrence(_Wire):
    event_id: str
    name: str
    timestamp: str
    data: JsonObject
    cursor: str | None = None
    meta: JsonObject | None = Field(default=None, alias="_meta")


class _SubscriptionParams(_WireParams):
    name: str = Field(min_length=1)
    arguments: JsonObject = Field(default_factory=dict)
    cursor: str | None = None
    max_age_ms: int | None = Field(default=None, ge=0)


class PollEventsParams(_SubscriptionParams):
    max_events: int | None = Field(default=None, ge=1)


class PollEventsResult(_Wire):
    events: tuple[EventOccurrence, ...]
    cursor: str | None = None
    truncated: bool
    has_more: bool
    next_poll_ms: int = Field(ge=0)


class StreamEventsParams(_SubscriptionParams):
    heartbeat_ms: int | None = Field(default=None, ge=1)


class WebhookDelivery(_Wire):
    mode: Literal["webhook"] = "webhook"
    url: str
    secret: str


class SubscribeEventsParams(_SubscriptionParams):
    delivery: WebhookDelivery
    ttl_ms: int | None = Field(default=None, ge=0)


class DeliveryStatus(_Wire):
    active: bool
    last_delivery_at: str | None = None
    last_error: CallbackFailure | None = None
    failed_since: str | None = None
    throttled: bool | None = None
    retry_after_ms: int | None = None


class SubscribeEventsResult(_Wire):
    id: str
    refresh_before: str | None
    cursor: str | None = None
    truncated: bool = False
    delivery_status: DeliveryStatus | None = None

    @model_serializer(mode="wrap")
    def _keep_refresh_before(self, handler: SerializerFunctionWrapHandler, info: SerializationInfo) -> JsonObject:
        serialized: Final = _JSON_OBJECT.validate_python(handler(self))
        field: Final = "refreshBefore" if info.by_alias else "refresh_before"
        return {**serialized, field: self.refresh_before}


class StreamEventsResult(_Wire):
    meta: JsonObject | None = Field(default=None, alias="_meta")


class UnsubscribeDelivery(_Wire):
    url: str


class UnsubscribeEventsParams(_WireParams):
    name: str = Field(min_length=1)
    arguments: JsonObject = Field(default_factory=dict)
    delivery: UnsubscribeDelivery


class EventsError(_Wire):
    code: int
    message: str
    data: JsonObject | None = None


class ActiveNotification(_Wire):
    cursor: str | None = None
    truncated: bool = False
    heartbeat_ms: int


class HeartbeatNotification(_Wire):
    cursor: str | None = None


class ErrorNotification(_Wire):
    error: EventsError


class GapEnvelope(_Wire):
    type: Literal["gap"] = "gap"
    cursor: str | None = None


class TerminatedEnvelope(_Wire):
    type: Literal["terminated"] = "terminated"
    error: EventsError


class VerificationEnvelope(_Wire):
    type: Literal["verification"] = "verification"
    challenge: str
