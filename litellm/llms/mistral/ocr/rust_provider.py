from enum import Enum
from typing import Any

from litellm.secret_managers.main import get_secret_str


class MistralRustOcrProvider(str, Enum):
    MISTRAL = "mistral"


def get_mistral_rust_ocr_provider(
    custom_llm_provider: str | None,
) -> str | None:
    provider_value = getattr(custom_llm_provider, "value", custom_llm_provider)
    if provider_value != MistralRustOcrProvider.MISTRAL.value:
        return None
    return MistralRustOcrProvider.MISTRAL.value


def get_mistral_rust_ocr_url(api_base: str | None) -> str:
    if api_base is None:
        api_base = "https://api.mistral.ai/v1"

    api_base = api_base.rstrip("/")
    if api_base.endswith("/v1"):
        return f"{api_base}/ocr"
    return f"{api_base}/v1/ocr"


def get_mistral_rust_ocr_headers(
    headers: dict[str, Any] | None,
    api_key: str | None,
) -> dict[str, Any]:
    if api_key is None:
        api_key = get_secret_str("MISTRAL_API_KEY")

    if api_key is None:
        raise ValueError(
            "Missing Mistral API Key - A call is being made to Mistral but no key "
            "is set either in the environment variables or via params"
        )

    return {
        "Authorization": f"Bearer {api_key}",
        **(headers or {}),
    }
