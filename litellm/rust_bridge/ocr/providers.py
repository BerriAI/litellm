from typing import Any, Optional

from litellm.rust_bridge.loader import call_rust_function, rust_core_enabled


def call_ocr(payload: dict[str, Any]) -> Optional[dict[str, Any]]:
    provider = payload.get("provider")
    if not isinstance(provider, str):
        return None

    if not _rust_ocr_provider_enabled(provider):
        return None

    result = call_rust_function("ocr", payload)
    if result is None:
        return None
    if not isinstance(result, dict):
        raise ValueError("Rust OCR bridge returned invalid response")
    return result


def _rust_ocr_provider_enabled(provider: str) -> bool:
    return (
        rust_core_enabled("ocr")
        or rust_core_enabled(f"ocr:{provider}")
        or rust_core_enabled(f"{provider}_ocr")
    )
