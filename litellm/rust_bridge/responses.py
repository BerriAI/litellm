from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Final, Protocol, TypeVar, cast  # noqa: TID251  # runtime typing constructs

from .bindings import UNCHANGED, Unchanged
from .callbacks import OneShotCallbackHandle
from .configuration import rust_enabled
from .runtime import BridgeErrorContext, EndpointDispatch

ResultT = TypeVar("ResultT")


class RustResponses(Protocol):
    def __call__(
        self, request: Mapping[str, object], callback_adapter: OneShotCallbackHandle | None
    ) -> Mapping[str, object]: ...


class RustAresponses(Protocol):
    def __call__(
        self, request: Mapping[str, object], callback_adapter: OneShotCallbackHandle | None
    ) -> Awaitable[Mapping[str, object]]: ...


@dataclass(frozen=True, slots=True)
class NativeResponsesRequest:
    body: Mapping[str, object]
    callback_adapter: OneShotCallbackHandle | None = None


_RESPONSES: Final = cast(  # cast-ok: generic classmethod loses the route Protocol parameters
    EndpointDispatch[RustResponses, RustAresponses],
    EndpointDispatch.native(
        route="responses",
        sync="responses",
        asynchronous="aresponses",
        enabled=rust_enabled,
    ),
)  # cast-ok: generic classmethod cannot preserve the route Protocol parameters


def set_rust_responses(
    *,
    sync: RustResponses | None | Unchanged = UNCHANGED,
    asynchronous: RustAresponses | None | Unchanged = UNCHANGED,
) -> None:
    if not isinstance(sync, Unchanged):
        if sync is None:
            _RESPONSES.sync.reset()
        else:
            _RESPONSES.sync.override(sync)
    if not isinstance(asynchronous, Unchanged):
        if asynchronous is None:
            _RESPONSES.asynchronous.reset()
        else:
            _RESPONSES.asynchronous.override(asynchronous)


def dispatch_responses(
    *,
    prepare: Callable[[], NativeResponsesRequest],
    fallback: Callable[[], ResultT],
    adapt: Callable[[Mapping[str, object]], ResultT],
    model: str,
    provider: str,
    request_override: bool | None = None,
    eligible: bool = True,
) -> ResultT:
    return _RESPONSES.invoke(
        prepare=prepare,
        call=_call_responses,
        fallback=fallback,
        adapt=adapt,
        context=BridgeErrorContext(provider=provider, model=model),
        request_override=request_override,
        eligible=eligible,
    )


async def adispatch_responses(
    *,
    prepare: Callable[[], NativeResponsesRequest],
    fallback: Callable[[], Awaitable[ResultT]],
    adapt: Callable[[Mapping[str, object]], ResultT],
    model: str,
    provider: str,
    request_override: bool | None = None,
    eligible: bool = True,
) -> ResultT:
    return await _RESPONSES.ainvoke(
        prepare=prepare,
        call=_call_aresponses,
        fallback=fallback,
        adapt=adapt,
        context=BridgeErrorContext(provider=provider, model=model),
        request_override=request_override,
        eligible=eligible,
    )


def _call_responses(native: RustResponses, request: NativeResponsesRequest) -> Mapping[str, object]:
    return native(request.body, callback_adapter=request.callback_adapter)


def _call_aresponses(
    native: RustAresponses,
    request: NativeResponsesRequest,
) -> Awaitable[Mapping[str, object]]:
    return native(request.body, callback_adapter=request.callback_adapter)
