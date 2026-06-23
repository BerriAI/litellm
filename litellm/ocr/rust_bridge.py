"""
Optional Rust-backed OCR path.

Enable with ``litellm.use_litellm_rust()``; the sync ``litellm.ocr()`` entrypoint
then routes supported Mistral calls through the compiled ``litellm_python_bridge``
extension, which performs the whole OCR call (URL, headers, HTTP, parse) in Rust.
"""

from typing import Any, Dict, Optional

from litellm.llms.base_llm.ocr.transformation import OCRResponse

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
    document: Dict[str, Any],
    api_key: Optional[str],
    api_base: Optional[str],
    optional_params: Dict[str, Any],
) -> OCRResponse:
    """Run a Mistral OCR call end-to-end in Rust and wrap the result as ``OCRResponse``."""
    import litellm_python_bridge

    return OCRResponse(
        **litellm_python_bridge.ocr(model, document, api_key, api_base, optional_params)
    )
