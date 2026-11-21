from dataclasses import dataclass
from typing import Final

from mcp.shared.exceptions import MCPError

from litellm.types.mcp_server.events import CallbackFailure, EventsErrorCode, JsonObject


@dataclass(frozen=True, slots=True)
class EventsFailure:
    code: EventsErrorCode
    message: str
    data: JsonObject | None = None

    def to_mcp(self) -> MCPError:
        return MCPError(code=int(self.code), message=self.message, data=dict(self.data) if self.data else None)


def method_not_found() -> EventsFailure:
    return EventsFailure(EventsErrorCode.METHOD_NOT_FOUND, "Method not found")


def event_not_found() -> EventsFailure:
    return EventsFailure(EventsErrorCode.NOT_FOUND, "NotFound", {"kind": "event"})


def subscription_not_found() -> EventsFailure:
    return EventsFailure(EventsErrorCode.NOT_FOUND, "NotFound", {"kind": "subscription"})


def forbidden(reason: str) -> EventsFailure:
    return EventsFailure(EventsErrorCode.FORBIDDEN, "Forbidden", {"reason": reason})


def exhausted(limit: str, maximum: int) -> EventsFailure:
    return EventsFailure(EventsErrorCode.RESOURCE_EXHAUSTED, "ResourceExhausted", {"limit": limit, "max": maximum})


def unsupported(feature: str, value: str | None = None) -> EventsFailure:
    data: Final[JsonObject] = {"feature": feature} if value is None else {"feature": feature, "value": value}
    return EventsFailure(EventsErrorCode.UNSUPPORTED, "Unsupported", data)


def invalid_params(message: str) -> EventsFailure:
    return EventsFailure(EventsErrorCode.INVALID_PARAMS, message)


def internal_error(message: str) -> EventsFailure:
    return EventsFailure(EventsErrorCode.INTERNAL_ERROR, message)


def callback_error(reason: CallbackFailure) -> EventsFailure:
    return EventsFailure(EventsErrorCode.CALLBACK_ENDPOINT_ERROR, "CallbackEndpointError", {"reason": reason})


def header_mismatch(header: str) -> EventsFailure:
    return EventsFailure(EventsErrorCode.HEADER_MISMATCH, "HeaderMismatch", {"header": header})
