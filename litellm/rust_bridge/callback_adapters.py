import json
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Final, Protocol

from pydantic import BaseModel, ConfigDict, JsonValue
from typing_extensions import ReadOnly, TypedDict

from .callbacks import CallbackDecision, CallbackUnchanged, SessionCallbackHandle


class PreCallArguments(TypedDict):
    complete_input_dict: ReadOnly[Mapping[str, JsonValue]]
    api_base: ReadOnly[str]
    headers: ReadOnly[Mapping[str, str]]


class ProviderLogging(Protocol):
    @property
    def model_call_details(self) -> MutableMapping[str, object]: ...  # mutable-ok: legacy logger stores provider events

    def pre_call(self, *, input: object, api_key: str | None, additional_args: PreCallArguments) -> None: ...

    def post_call(self, *, original_response: str, input: object, api_key: str | None) -> None: ...


class ProviderEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    call_id: str
    trace_id: str | None = None
    attempt: int
    started_at: float


class ProviderPreCall(ProviderEvent):
    api_key: str | None = None
    request: Mapping[str, JsonValue]
    api_base: str
    headers: Mapping[str, str]


class ProviderPostCall(ProviderEvent):
    api_key: str | None = None
    response: JsonValue
    status_code: int
    headers: Mapping[str, str]
    ended_at: float


class ProviderError(ProviderEvent):
    message: str
    stage: str
    committed: bool
    status_code: int | None
    will_retry: bool
    ended_at: float


class StreamEvent(ProviderEvent):
    event: JsonValue
    sequence: int


class StreamClose(ProviderEvent):
    outcome: str
    ended_at: float


class SessionEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    call_id: str
    trace_id: str | None = None
    event: JsonValue | None = None
    response_id: str | None = None
    sequence: int | None = None
    message: str | None = None


def _unchanged() -> CallbackUnchanged:
    return {"action": "unchanged"}  # mutable-ok: callback protocol requires a concrete decision payload


@dataclass(frozen=True, slots=True)
class ProviderLoggingAdapter:
    logging_obj: ProviderLogging
    input: object
    api_key: str | None

    def pre_call(self, payload: object, /) -> CallbackDecision:
        event: Final = ProviderPreCall.model_validate(payload)
        additional_args: Final[PreCallArguments] = {
            "complete_input_dict": event.request,
            "api_base": event.api_base,
            "headers": event.headers,
        }
        self.logging_obj.pre_call(
            input=self.input, api_key=event.api_key or self.api_key, additional_args=additional_args
        )
        return _unchanged()

    def post_call(self, payload: object, /) -> CallbackDecision:
        event: Final = ProviderPostCall.model_validate(payload)
        response: Final = event.response if isinstance(event.response, str) else json.dumps(event.response)
        self.logging_obj.post_call(original_response=response, input=self.input, api_key=event.api_key or self.api_key)
        return _unchanged()

    def error(self, payload: object, /) -> None:
        event: Final = ProviderError.model_validate(payload)
        self.logging_obj.model_call_details["provider_error"] = event.model_dump()

    def stream_event(self, payload: object, /) -> CallbackDecision:
        event: Final = StreamEvent.model_validate(payload)
        self.logging_obj.model_call_details["provider_stream_event"] = event.model_dump()
        return _unchanged()

    def stream_close(self, payload: object, /) -> None:
        event: Final = StreamClose.model_validate(payload)
        self.logging_obj.model_call_details["provider_stream_close"] = event.model_dump()


@dataclass(frozen=True, slots=True)
class SessionCallbackAdapter:
    callback: SessionCallbackHandle

    def before_connect(self, payload: object, /) -> CallbackDecision:
        return self.callback.before_connect(SessionEvent.model_validate(payload).model_dump())

    def connected(self, payload: object, /) -> None:
        self.callback.connected(SessionEvent.model_validate(payload).model_dump())

    def before_send(self, payload: object, /) -> CallbackDecision:
        return self.callback.before_send(SessionEvent.model_validate(payload).model_dump())

    def after_receive(self, payload: object, /) -> CallbackDecision:
        return self.callback.after_receive(SessionEvent.model_validate(payload).model_dump())

    def response_complete(self, payload: object, /) -> None:
        self.callback.response_complete(SessionEvent.model_validate(payload).model_dump())

    def response_error(self, payload: object, /) -> None:
        self.callback.response_error(SessionEvent.model_validate(payload).model_dump())

    def error(self, payload: object, /) -> None:
        self.callback.error(SessionEvent.model_validate(payload).model_dump())

    def close(self, payload: object, /) -> None:
        self.callback.close(SessionEvent.model_validate(payload).model_dump())
