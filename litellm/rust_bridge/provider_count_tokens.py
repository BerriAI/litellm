from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Final, Protocol, cast  # noqa: TID251  # runtime typing constructs

from litellm.rust_bridge.bindings import UNCHANGED, Unchanged
from litellm.rust_bridge.callbacks import OneShotCallbackHandle
from litellm.rust_bridge.configuration import rust_enabled
from litellm.rust_bridge.runtime import AsyncEndpointDispatch, BridgeErrorContext, identity
from litellm.types.utils import TokenCountResponse


class RustProviderCountTokens(Protocol):
    def __call__(
        self,
        request: Mapping[str, object],
        callback_adapter: OneShotCallbackHandle | None,
    ) -> Awaitable[TokenCountResponse]: ...


@dataclass(frozen=True, slots=True)
class NativeProviderCountTokensRequest:
    body: Mapping[str, object]
    callback_adapter: OneShotCallbackHandle | None = None


_PROVIDER_COUNT_TOKENS: Final = cast(
    AsyncEndpointDispatch[RustProviderCountTokens],
    AsyncEndpointDispatch.native(
        route="provider_count_tokens",
        asynchronous="acount_tokens",
        enabled=rust_enabled,
    ),
)  # cast-ok: generic classmethod cannot preserve the route Protocol parameter


def set_rust_provider_count_tokens(
    *,
    asynchronous: RustProviderCountTokens | None | Unchanged = UNCHANGED,
) -> None:
    if isinstance(asynchronous, Unchanged):
        return
    if asynchronous is None:
        _PROVIDER_COUNT_TOKENS.reset()
        return
    _PROVIDER_COUNT_TOKENS.override(asynchronous)


async def adispatch_provider_count_tokens(
    *,
    prepare: Callable[[], NativeProviderCountTokensRequest],
    fallback: Callable[[], Awaitable[TokenCountResponse | None]],
    model: str,
    provider: str,
    request_override: bool | None = None,
) -> TokenCountResponse | None:
    return await _PROVIDER_COUNT_TOKENS.ainvoke(
        prepare=prepare,
        call=_call_provider_count_tokens,
        fallback=fallback,
        adapt=identity,
        context=BridgeErrorContext(provider=provider, model=model),
        request_override=request_override,
    )


def _call_provider_count_tokens(
    native: RustProviderCountTokens,
    request: NativeProviderCountTokensRequest,
) -> Awaitable[TokenCountResponse]:
    return native(request.body, callback_adapter=request.callback_adapter)
