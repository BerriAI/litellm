"""
Optional Rust-backed OCR path.

Enable with ``litellm.use_litellm_rust()``; ``litellm.ocr()`` / ``litellm.aocr()``
then route supported providers through the compiled ``litellm_python_bridge``
extension, which performs the whole OCR call (URL, headers, HTTP, parse) in Rust.

The bridge exposes two entry points that mirror the Python API:

* ``ocr(...)`` blocks on the async core with the GIL released (sync SDK callers).
* ``aocr(...)`` returns a Python awaitable driven by a Tokio runtime, so the
  proxy can ``await`` it without tying up a thread-pool worker per request.

No module-level ``litellm`` imports keep this a leaf so ``litellm/ocr/main.py``
can import it statically without forming an import cycle.
"""

from __future__ import annotations

from typing import Any, Awaitable, Final, Protocol, Union, cast


class RustBridge(Protocol):
    """The compiled ``litellm_python_bridge`` surface used by the OCR path."""

    def ocr(
        self,
        provider: str,
        model: str,
        document: dict[str, Any],
        api_key: str | None = None,
        api_base: str | None = None,
        extra_headers: dict[str, str] | None = None,
        timeout_seconds: float | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def aocr(
        self,
        provider: str,
        model: str,
        document: dict[str, Any],
        api_key: str | None = None,
        api_base: str | None = None,
        extra_headers: dict[str, str] | None = None,
        timeout_seconds: float | None = None,
        params: dict[str, Any] | None = None,
    ) -> Awaitable[dict[str, Any]]: ...


# Providers whose full OCR call the Rust core can handle today. Grows one entry
# per provider PR; everything else stays on the Python path.
RUST_SUPPORTED_PROVIDERS: Final[frozenset[str]] = frozenset({"mistral"})


class _Unset:
    """Sentinel so ``bridge=None`` can clear a prior injection while omission preserves it."""


_UNSET: Final[_Unset] = _Unset()

_rust_ocr_enabled = False
_rust_bridge: RustBridge | None = None


def use_litellm_rust(
    enabled: bool = True, *, bridge: Union[RustBridge, None, _Unset] = _UNSET
) -> None:
    """Route supported OCR calls through the Rust ``litellm_python_bridge`` extension.

    ``bridge`` injects the bridge object (mainly for tests); when omitted the
    compiled extension is loaded on demand and any previously injected bridge is
    preserved. Pass ``bridge=None`` explicitly to clear a prior injection.
    """
    global _rust_ocr_enabled, _rust_bridge
    _rust_ocr_enabled = enabled
    if not isinstance(bridge, _Unset):
        _rust_bridge = bridge


def rust_ocr_enabled() -> bool:
    """Whether the Rust OCR path has been turned on via ``use_litellm_rust()``."""
    return _rust_ocr_enabled


def rust_supports(provider: str) -> bool:
    """Whether the Rust core can handle this provider's OCR call end to end."""
    return provider in RUST_SUPPORTED_PROVIDERS


def load_rust_bridge() -> RustBridge | None:
    """Return the Rust bridge, or ``None`` when no bridge is available.

    Prefers an injected bridge, otherwise loads the compiled
    ``litellm_python_bridge`` extension; a missing extension yields ``None`` so
    the caller can fall back to the Python path instead of hard-failing.
    """
    if _rust_bridge is not None:
        return _rust_bridge
    try:
        import litellm_python_bridge
    except ImportError:
        return None
    # The compiled module exposes ocr / aocr / gil_stats at module level.
    return cast(RustBridge, litellm_python_bridge)
