from typing import Literal, Protocol, TypeAlias

from typing_extensions import ReadOnly, TypedDict


class CallbackUnchanged(TypedDict):
    action: ReadOnly[Literal["unchanged"]]


class CallbackReplace(TypedDict):
    action: ReadOnly[Literal["replace"]]
    payload: ReadOnly[object]


class CallbackReject(TypedDict):
    action: ReadOnly[Literal["reject"]]
    message: ReadOnly[str]
    status_code: ReadOnly[int | None]


CallbackDecision: TypeAlias = CallbackUnchanged | CallbackReplace | CallbackReject


class ProviderAttemptCallbackHandle(Protocol):
    """Observe the provider operation inside one native call.

    Successful operations receive ``pre_call`` and ``post_call`` once. Failed
    operations receive ``pre_call`` and ``error`` once. Outer SDK success and
    failure callbacks remain owned by Python after endpoint dispatch completes.
    """

    def pre_call(self, payload: object, /) -> CallbackDecision: ...

    def post_call(self, payload: object, /) -> CallbackDecision: ...

    def error(self, payload: object, /) -> None: ...


class OneShotCallbackHandle(ProviderAttemptCallbackHandle, Protocol):
    pass


class StreamingCallbackHandle(ProviderAttemptCallbackHandle, Protocol):
    def stream_event(self, payload: object, /) -> CallbackDecision: ...

    def stream_close(self, payload: object, /) -> None: ...


class SessionCallbackHandle(Protocol):
    def before_connect(self, payload: object, /) -> CallbackDecision: ...

    def connected(self, payload: object, /) -> None: ...

    def before_send(self, payload: object, /) -> CallbackDecision: ...

    def after_receive(self, payload: object, /) -> CallbackDecision: ...

    def response_complete(self, payload: object, /) -> None: ...

    def response_error(self, payload: object, /) -> None: ...

    def error(self, payload: object, /) -> None: ...

    def close(self, payload: object, /) -> None: ...
