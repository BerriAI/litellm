"""
Optional Rust-backed OCR path.

Enable with ``litellm.use_litellm_rust()``; the sync ``litellm.ocr()`` entrypoint
then routes supported Mistral calls through the compiled ``litellm_python_bridge``
extension, which performs the whole OCR call (URL, headers, HTTP, parse) in Rust.
"""

from __future__ import annotations

_RUST_OCR_ENABLED = False


def use_litellm_rust(enabled: bool = True) -> None:
    """Route supported OCR calls through the Rust ``litellm_python_bridge`` extension."""
    global _RUST_OCR_ENABLED
    _RUST_OCR_ENABLED = enabled


def rust_ocr_enabled() -> bool:
    """Whether the Rust OCR path has been turned on via ``use_litellm_rust()``."""
    return _RUST_OCR_ENABLED


def rust_ocr(
    model: str,
    document: dict,
    api_key: str | None,
    api_base: str | None,
    optional_params: dict,
    timeout_seconds: float | None = None,
) -> dict:
    """Call the Rust bridge and return the raw OCR response dict.

    Kept free of ``litellm`` imports so this module stays a leaf — the caller
    (``litellm/ocr/main.py``) wraps the dict into an ``OCRResponse``. This avoids
    the import edge CodeQL repeatedly flags (and auto-"fixes") as a cyclic import.
    """
    import litellm_python_bridge

    return litellm_python_bridge.ocr(
        model, document, api_key, api_base, optional_params, timeout_seconds
    )
