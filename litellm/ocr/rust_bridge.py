"""
Rust-backed OCR path.

Supported OCR providers route through the compiled ``litellm.rust_bridge._native``
extension by default.

No module-level ``litellm`` imports keep this a leaf so ``litellm/ocr/main.py``
can import it statically without forming an import cycle.
"""

from __future__ import annotations

from typing import Awaitable, Final, Protocol, cast


class RustOcr(Protocol):
    """Signature of the compiled Rust OCR entrypoint."""

    def __call__(
        self,
        model: str,
        document: dict[str, object],
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str,
        extra_headers: dict[str, object] | None,
        optional_params: dict[str, object],
        timeout_seconds: float | None,
    ) -> dict[str, object]:
        raise NotImplementedError


class RustAocr(Protocol):
    """Signature of the compiled ``litellm_python_bridge.aocr`` entrypoint."""

    def __call__(
        self,
        model: str,
        document: dict[str, object],
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str,
        extra_headers: dict[str, object] | None,
        optional_params: dict[str, object],
        timeout_seconds: float | None,
    ) -> Awaitable[dict[str, object]]:
        raise NotImplementedError


class _Unset:
    """Sentinel type so ``ocr=None`` can clear a prior injection while omission preserves it."""


_UNSET: Final[_Unset] = _Unset()


_rust_ocr_impl: RustOcr | None = None
_rust_aocr_impl: RustAocr | None = None
_rust_ocr_input_error_override: type[BaseException] | None | _Unset = _UNSET


def _set_rust_ocr_input_error_type(
    error_type: type[BaseException] | None | _Unset = _UNSET,
) -> None:
    global _rust_ocr_input_error_override
    if not isinstance(error_type, _Unset):
        _rust_ocr_input_error_override = error_type


def _set_rust_ocr_bridge(
    ocr: RustOcr | None | _Unset = _UNSET,
    aocr: RustAocr | None | _Unset = _UNSET,
) -> None:
    """Configure OCR bridge injection for tests.

    ``ocr`` and ``aocr`` inject bridge callables. When omitted, any previously
    injected bridge is preserved. Pass ``None`` explicitly to clear a prior
    injection.
    """
    global _rust_ocr_impl, _rust_aocr_impl
    if not isinstance(ocr, _Unset):
        _rust_ocr_impl = ocr
    if not isinstance(aocr, _Unset):
        _rust_aocr_impl = aocr


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


def load_rust_aocr() -> RustAocr | None:
    """Return the async Rust OCR callable, or ``None`` when unavailable."""
    if _rust_aocr_impl is not None:
        return _rust_aocr_impl
    from litellm.rust_bridge import get_native_bridge

    native_bridge = get_native_bridge()
    if native_bridge is None:
        return None
    return cast(RustAocr, getattr(native_bridge, "aocr", None))


def rust_ocr_input_error_type() -> type[BaseException] | None:
    if not isinstance(_rust_ocr_input_error_override, _Unset):
        return _rust_ocr_input_error_override
    try:
        from litellm.rust_bridge._native import RustOcrInputError
    except ImportError:
        return None
    if isinstance(RustOcrInputError, type) and issubclass(
        RustOcrInputError, BaseException
    ):
        return RustOcrInputError
    return None
