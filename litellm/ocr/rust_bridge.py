"""
Optional Rust-backed OCR path.

Enable with ``litellm.use_litellm_rust()``; the sync ``litellm.ocr()`` entrypoint
then routes supported Mistral calls through the compiled ``litellm.rust_bridge._native``
extension, which performs the whole OCR call (URL, headers, HTTP, parse) in Rust.

No module-level ``litellm`` imports keep this a leaf so ``litellm/ocr/main.py``
can import it statically without forming an import cycle.
"""

from __future__ import annotations

from typing import Final, Protocol, cast


class RustOcr(Protocol):
    """Signature of the compiled Rust OCR entrypoint."""

    def __call__(
        self,
        model: str,
        document: dict[str, object],
        api_key: str | None,
        api_base: str | None,
        optional_params: dict[str, object],
        timeout_seconds: float | None,
    ) -> dict[str, object]: ...


class _Unset:
    """Sentinel type so ``ocr=None`` can clear a prior injection while omission preserves it."""


_UNSET: Final[_Unset] = _Unset()

_rust_ocr_enabled = False
_rust_ocr_impl: RustOcr | None = None


def use_litellm_rust(
    enabled: bool = True, *, ocr: RustOcr | None | _Unset = _UNSET
) -> None:
    """Route supported OCR calls through the packaged Rust extension.

    ``ocr`` injects the bridge callable; when omitted the compiled extension is
    loaded on demand and any previously injected bridge is preserved. Pass
    ``ocr=None`` explicitly to clear a prior injection.
    """
    global _rust_ocr_enabled, _rust_ocr_impl
    _rust_ocr_enabled = enabled
    if not isinstance(ocr, _Unset):
        _rust_ocr_impl = ocr


def rust_ocr_enabled() -> bool:
    """Whether the Rust OCR path has been turned on via ``use_litellm_rust()``."""
    return _rust_ocr_enabled


def load_rust_ocr() -> RustOcr | None:
    """Return the Rust OCR callable, or ``None`` when no bridge is available.

    Prefers an injected implementation, otherwise loads the compiled
    ``litellm.rust_bridge._native`` extension; a missing extension yields ``None`` so
    the caller can fall back to the Python path instead of hard-failing.
    """
    if _rust_ocr_impl is not None:
        return _rust_ocr_impl
    from litellm.rust_bridge import get_native_bridge

    native_bridge = get_native_bridge()
    if native_bridge is None:
        return None
    return cast(RustOcr, native_bridge.ocr)
