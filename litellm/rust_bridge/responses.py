from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Final, Protocol

from litellm.rust_bridge.configuration import rust_enabled
from litellm.rust_bridge.runtime import BridgeErrorContext, RustEndpoint, async_none, identity


class RustResponses(Protocol):
    def __call__(self, request: Mapping[str, object]) -> Mapping[str, object]: ...


class RustAresponses(Protocol):
    def __call__(self, request: Mapping[str, object]) -> Awaitable[Mapping[str, object]]: ...


_RESPONSES: Final[RustEndpoint[RustResponses, RustAresponses]] = RustEndpoint.native(
    route="responses",
    sync="responses",
    asynchronous="aresponses",
    enabled=rust_enabled,
)


def responses(
    *,
    prepare: Callable[[], Mapping[str, object]],
    model: str,
    provider: str,
    request_override: bool | None = None,
) -> Mapping[str, object] | None:
    return _RESPONSES.invoke(
        call=lambda rust_responses: rust_responses(prepare()),
        fallback=lambda: None,
        adapt=identity,
        context=BridgeErrorContext(provider=provider, model=model),
        request_override=request_override,
    )


async def aresponses(
    *,
    prepare: Callable[[], Mapping[str, object]],
    model: str,
    provider: str,
    request_override: bool | None = None,
) -> Mapping[str, object] | None:
    return await _RESPONSES.ainvoke(
        call=lambda rust_aresponses: rust_aresponses(prepare()),
        fallback=async_none,
        adapt=identity,
        context=BridgeErrorContext(provider=provider, model=model),
        request_override=request_override,
    )
