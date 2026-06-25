"""Loader for the packaged LiteLLM Rust extension."""

from __future__ import annotations

from types import ModuleType

_BRIDGE_SENTINEL = object()
_cached_bridge: ModuleType | None | object = _BRIDGE_SENTINEL


def get_native_bridge() -> ModuleType | None:
    """Return the packaged Rust extension, or ``None`` when unavailable."""
    global _cached_bridge
    if _cached_bridge is not _BRIDGE_SENTINEL:
        return _cached_bridge if isinstance(_cached_bridge, ModuleType) else None

    try:
        from litellm.rust_bridge import _native
    except ImportError:
        _cached_bridge = None
        return None
    _cached_bridge = _native
    return _native


def native_bridge_available() -> bool:
    """Whether the packaged Rust extension is importable."""
    return get_native_bridge() is not None
