from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from litellm.router_backends.rust_support import unsupported_reason
from litellm.rust_bridge.bindings import NativeBinding


@dataclass(frozen=True, slots=True)
class RustRouterDeclined:
    reason: str


def _native_router_type(value: object) -> type | None:
    return value if isinstance(value, type) else None


NATIVE_ROUTER: Final = NativeBinding("Router", validate=_native_router_type)


class RustRouter:
    """The Rust-backed router. Members it does not emulate yet raise `NotImplementedError` naming them."""

    def __getattr__(self, name: str) -> object:
        raise NotImplementedError(f"the Rust router backend does not emulate Router.{name} yet")


def build_rust_router(arguments: Mapping[str, object]) -> RustRouter | RustRouterDeclined:
    reason: Final = unsupported_reason(arguments)
    if reason is not None:
        return RustRouterDeclined(reason)
    if NATIVE_ROUTER.load() is None:
        return RustRouterDeclined("the native router is not available in this build")
    return RustRouterDeclined("the native router is not wired up yet")
