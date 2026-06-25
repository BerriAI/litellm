"""
Optional Rust-backed OCR path.

Enable with ``litellm.use_litellm_rust()``; the sync ``litellm.ocr()`` entrypoint
then routes supported Mistral calls through the compiled ``litellm_python_bridge``
extension, which performs the whole OCR call (URL, headers, HTTP, parse) in Rust.

No module-level ``litellm`` imports keep this a leaf so ``litellm/ocr/main.py``
can import it statically without forming an import cycle.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Final, Protocol, cast


class RustOcr(Protocol):
    """Signature of the compiled ``litellm_python_bridge.ocr`` entrypoint."""

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


RustImageEdit = Callable[
    [
        str,
        list[dict[str, object]],
        str | None,
        dict[str, object] | None,
        str | None,
        str | None,
        str,
        dict[str, object] | None,
        dict[str, object],
        float | None,
    ],
    dict[str, object],
]
RustAimageEdit = Callable[
    [
        str,
        list[dict[str, object]],
        str | None,
        dict[str, object] | None,
        str | None,
        str | None,
        str,
        dict[str, object] | None,
        dict[str, object],
        float | None,
    ],
    Awaitable[dict[str, object]],
]


class _Unset:
    """Sentinel type so ``ocr=None`` can clear a prior injection while omission preserves it."""


_UNSET: Final[_Unset] = _Unset()

_rust_ocr_enabled = False
_rust_ocr_impl: RustOcr | None = None
_rust_aocr_impl: RustAocr | None = None
_rust_image_edit_impl: RustImageEdit | None = None
_rust_aimage_edit_impl: RustAimageEdit | None = None


def use_litellm_rust(
    enabled: bool = True,
    *,
    ocr: RustOcr | None | _Unset = _UNSET,
    aocr: RustAocr | None | _Unset = _UNSET,
    image_edit: RustImageEdit | None | _Unset = _UNSET,
    aimage_edit: RustAimageEdit | None | _Unset = _UNSET,
) -> None:
    """Route supported calls through the Rust ``litellm_python_bridge`` extension.

    Bridge callables can be injected for tests; when omitted the compiled
    extension is loaded on demand and any previously injected bridge is
    preserved. Pass ``None`` explicitly to clear a prior injection.
    """
    global _rust_ocr_enabled, _rust_ocr_impl, _rust_aocr_impl
    global _rust_image_edit_impl, _rust_aimage_edit_impl
    _rust_ocr_enabled = enabled
    if not isinstance(ocr, _Unset):
        _rust_ocr_impl = ocr
    if not isinstance(aocr, _Unset):
        _rust_aocr_impl = aocr
    if not isinstance(image_edit, _Unset):
        _rust_image_edit_impl = image_edit
    if not isinstance(aimage_edit, _Unset):
        _rust_aimage_edit_impl = aimage_edit


def rust_ocr_enabled() -> bool:
    """Whether the Rust OCR path has been turned on via ``use_litellm_rust()``."""
    return _rust_ocr_enabled


def rust_image_edit_enabled() -> bool:
    """Whether the Rust image-edit path has been turned on."""
    return _rust_ocr_enabled


def load_rust_ocr() -> RustOcr | None:
    """Return the Rust OCR callable, or ``None`` when no bridge is available.

    Prefers an injected implementation, otherwise loads the compiled
    ``litellm_python_bridge`` extension; a missing extension yields ``None`` so
    the caller can fall back to the Python path instead of hard-failing.
    """
    if _rust_ocr_impl is not None:
        return _rust_ocr_impl
    try:
        import litellm_python_bridge
    except ImportError:
        return None
    return cast(RustOcr, litellm_python_bridge.ocr)


def load_rust_aocr() -> RustAocr | None:
    """Return the async Rust OCR callable, or ``None`` when unavailable."""
    if _rust_aocr_impl is not None:
        return _rust_aocr_impl
    try:
        import litellm_python_bridge
    except ImportError:
        return None
    return cast(RustAocr, getattr(litellm_python_bridge, "aocr", None))


def load_rust_image_edit() -> RustImageEdit | None:
    """Return the Rust image-edit callable, or ``None`` when unavailable."""
    if _rust_image_edit_impl is not None:
        return _rust_image_edit_impl
    try:
        import litellm_python_bridge
    except ImportError:
        return None
    return cast(RustImageEdit, getattr(litellm_python_bridge, "image_edit", None))


def load_rust_aimage_edit() -> RustAimageEdit | None:
    """Return the async Rust image-edit callable, or ``None`` when unavailable."""
    if _rust_aimage_edit_impl is not None:
        return _rust_aimage_edit_impl
    try:
        import litellm_python_bridge
    except ImportError:
        return None
    return cast(RustAimageEdit, getattr(litellm_python_bridge, "aimage_edit", None))
