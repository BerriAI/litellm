from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class NativeBridgeExceptions:
    declined: type[BaseException]
    upstream: type[BaseException]


@dataclass(frozen=True, slots=True)
class RustUpstreamFailure:
    status_code: int
    message: str


def native_bridge_exceptions(native_bridge: object | None) -> NativeBridgeExceptions | None:
    if native_bridge is None:
        return None
    declined: Final = getattr(native_bridge, "RustBridgeDeclined", None)
    upstream: Final = getattr(native_bridge, "RustUpstreamError", None)
    if not isinstance(declined, type) or not isinstance(upstream, type):
        return None
    return NativeBridgeExceptions(declined=declined, upstream=upstream)


def rust_upstream_failure(
    error: BaseException,
    native_bridge: object | None,
) -> RustUpstreamFailure | None:
    exceptions: Final = native_bridge_exceptions(native_bridge)
    if exceptions is None or not isinstance(error, exceptions.upstream):
        return None
    status: Final = error.args[0] if error.args else 0
    message: Final = error.args[1] if len(error.args) > 1 else ""
    return RustUpstreamFailure(status_code=int(status) or 500, message=str(message))
