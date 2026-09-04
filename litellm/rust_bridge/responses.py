from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Final, Protocol, TypeVar, cast  # noqa: TID251  # runtime typing constructs

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


_RESPONSES: Final = cast(  # cast-ok: generic classmethod loses the route Protocol parameters
    EndpointDispatch[RustResponses, RustAresponses],
    EndpointDispatch.native(
        route="responses",
        sync="responses",
        asynchronous="aresponses",
        enabled=rust_enabled,
    ),
)  # cast-ok: generic classmethod cannot preserve the route Protocol parameters


def responses(
    *,
    prepare: Callable[[], Mapping[str, object]],
    fallback: Callable[[], ResultT],
    adapt: Callable[[Mapping[str, object]], ResultT],
    model: str,
    provider: str,
    request_override: bool | None = None,
    callback_adapter: OneShotCallbackHandle | None = None,
    eligible: bool = True,
) -> ResultT:
    return _RESPONSES.invoke(
        call=lambda rust_responses: rust_responses(prepare(), callback_adapter=callback_adapter),
        fallback=fallback,
        adapt=adapt,
        context=BridgeErrorContext(provider=provider, model=model),
        request_override=request_override,
        eligible=eligible,
    )


async def aresponses(
    *,
    prepare: Callable[[], Mapping[str, object]],
    fallback: Callable[[], Awaitable[ResultT]],
    adapt: Callable[[Mapping[str, object]], ResultT],
    model: str,
    provider: str,
    request_override: bool | None = None,
    callback_adapter: OneShotCallbackHandle | None = None,
    eligible: bool = True,
) -> ResultT:
    return await _RESPONSES.ainvoke(
        call=lambda rust_aresponses: rust_aresponses(prepare(), callback_adapter=callback_adapter),
        fallback=fallback,
        adapt=adapt,
        context=BridgeErrorContext(provider=provider, model=model),
        request_override=request_override,
        eligible=eligible,
    )
