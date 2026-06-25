"""Loader for the packaged LiteLLM Rust extension."""

from __future__ import annotations

from types import ModuleType


def get_native_bridge() -> ModuleType | None:
    """Return the packaged Rust extension, or ``None`` when unavailable."""
    try:
        from litellm.rust_bridge import _native
    except ImportError:
        return None
    return _native


def native_bridge_available() -> bool:
    """Whether the packaged Rust extension is importable."""
    return get_native_bridge() is not None
