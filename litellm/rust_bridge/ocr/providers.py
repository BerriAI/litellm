from enum import Enum
from typing import Any, Dict, Optional

from litellm.rust_bridge.loader import call_rust_function, rust_core_enabled


class RustOcrProvider(str, Enum):
    MISTRAL = "mistral"


RUST_OCR_PROVIDERS = frozenset({RustOcrProvider.MISTRAL.value})


def call_ocr(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    provider = payload.get("provider")
    if not isinstance(provider, str) or provider not in RUST_OCR_PROVIDERS:
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
