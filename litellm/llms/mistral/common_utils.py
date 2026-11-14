from collections.abc import Mapping
from typing import Final

import httpx

from litellm.litellm_core_utils.url_utils import ensure_api_route
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str

MISTRAL_API_BASE: Final = "https://api.mistral.ai"
MISTRAL_API_KEY_ENV_VAR: Final = "MISTRAL_API_KEY"


class MistralError(BaseLLMException):
    pass


def get_mistral_api_base(api_base: str | None) -> str:
    """Return the Mistral origin without a trailing ``/v1``, so callers can append ``/v1/<route>``."""
    resolved: Final = (api_base or get_secret_str("MISTRAL_API_BASE") or MISTRAL_API_BASE).rstrip("/")
    return resolved.removesuffix("/v1")


def get_mistral_chat_credentials(api_base: str | None, api_key: str | None) -> tuple[str, str | None]:
    resolved_base: Final = api_base or get_secret_str("MISTRAL_AZURE_API_BASE") or f"{MISTRAL_API_BASE}/v1"
    resolved_key: Final = api_key or get_secret_str("MISTRAL_AZURE_API_KEY") or get_secret_str(MISTRAL_API_KEY_ENV_VAR)
    return ensure_api_route(resolved_base, "v1"), resolved_key


def get_mistral_auth_headers(
    headers: Mapping[str, str], api_key: str | None
) -> dict[str, str]:  # mutable-ok: BaseConfig.validate_environment contract returns dict
    resolved_key: Final = api_key or get_secret_str(MISTRAL_API_KEY_ENV_VAR)
    if resolved_key is None:
        raise ValueError(
            "Missing Mistral API Key - A call is being made to Mistral but no key is set either in the environment variables or via params"
        )
    return dict(headers, Authorization=f"Bearer {resolved_key}")


def mistral_error(error_message: str, status_code: int, headers: Mapping[str, str] | httpx.Headers) -> MistralError:
    return MistralError(
        status_code=status_code,
        message=error_message,
        headers=headers if isinstance(headers, httpx.Headers) else httpx.Headers(dict(headers)),
    )
