"""Turns a `RustDenial` into the public error the proxy raised for it before the decision moved to Rust."""

from __future__ import annotations

from litellm.proxy._types import ProxyErrorTypes, ProxyException
from litellm.rust_bridge._native import RustDenial


def denial_error(denial: RustDenial) -> Exception:
    match denial:
        case RustDenial.Proxy(status=status, message=message, error_type=error_type, param=param):
            return ProxyException(message=message, type=ProxyErrorTypes(error_type), param=param, code=status)
        case unknown:
            raise RuntimeError(f"rust auth returned a denial this host does not know: {unknown!r}")
