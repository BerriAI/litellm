"""Thin Python wrapper for the native Rust auth bridge.

The Rust core owns SHA-256 token hashing. This module loads the native
function and provides a fallback when the bridge is unavailable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final, Protocol

from litellm._logging import verbose_logger
from litellm.rust_bridge.loader import get_native_bridge

_TRUTHY_ENV_VALUES: Final = frozenset({"1", "true", "yes", "on"})


class RustHashToken(Protocol):
    def __call__(self, token: str) -> str:
        raise NotImplementedError


@dataclass(slots=True)
class _RustAuthState:
    hash_token: RustHashToken | None = None


_STATE: Final[_RustAuthState] = _RustAuthState()


def set_rust_hash_token(hash_token: RustHashToken | None) -> None:
    """Inject the native callable, so tests can supply a double."""
    _STATE.hash_token = hash_token


def _env_enables_rust_auth() -> bool:
    return (
        os.getenv("LITELLM_RUST_AUTH", "").strip().lower()
        in _TRUTHY_ENV_VALUES
    )


def load_rust_hash_token() -> RustHashToken | None:
    """Return the native hash_token function, or None when unavailable."""
    if _STATE.hash_token is not None:
        return _STATE.hash_token

    native_bridge = get_native_bridge()
    if native_bridge is None:
        return None

    func = getattr(native_bridge, "hash_token", None)
    if func is not None:
        _STATE.hash_token = func
    return func


def try_rust_hash_token(token: str) -> str | None:
    """Attempt to hash a token using the Rust bridge.

    Returns the 64-char hex SHA-256 hash on success, or None if the Rust
    path is unavailable.
    """
    if not _env_enables_rust_auth():
        return None

    rust_func = load_rust_hash_token()
    if rust_func is None:
        return None

    try:
        return rust_func(token)
    except Exception:
        verbose_logger.debug(
            "Rust hash_token failed, falling back to Python", exc_info=True
        )
        return None
