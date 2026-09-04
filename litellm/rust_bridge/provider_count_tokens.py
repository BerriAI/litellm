from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
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


_PROVIDER_COUNT_TOKENS: Final = cast(
    AsyncEndpointDispatch[RustProviderCountTokens],
    AsyncEndpointDispatch.native(
        route="provider_count_tokens",
        asynchronous="acount_tokens",
        enabled=rust_enabled,
    ),
)  # cast-ok: generic classmethod cannot preserve the route Protocol parameter


def set_rust_provider_count_tokens(
    acount_tokens: RustProviderCountTokens | None | Unchanged = UNCHANGED,
) -> None:
    if isinstance(acount_tokens, Unchanged):
        return
    if acount_tokens is None:
        _PROVIDER_COUNT_TOKENS.reset()
        return
    _PROVIDER_COUNT_TOKENS.override(acount_tokens)


async def acount_tokens(
    *,
    prepare: Callable[[], Mapping[str, object]],
    fallback: Callable[[], Awaitable[TokenCountResponse | None]],
    model: str,
    provider: str,
    request_override: bool | None = None,
    callback_adapter: OneShotCallbackHandle | None = None,
) -> TokenCountResponse | None:
    return await _PROVIDER_COUNT_TOKENS.ainvoke(
        call=lambda native: native(prepare(), callback_adapter=callback_adapter),
        fallback=fallback,
        adapt=identity,
        context=BridgeErrorContext(provider=provider, model=model),
        request_override=request_override,
    )
