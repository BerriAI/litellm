from typing import Optional

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str

DEFAULT_COMETAPI_API_BASE = "https://api.cometapi.com/v1"


def get_cometapi_api_key(api_key: Optional[str] = None) -> Optional[str]:
    return (
        api_key or get_secret_str("COMETAPI_KEY") or get_secret_str("COMETAPI_API_KEY")
    )


def get_cometapi_api_base(api_base: Optional[str] = None) -> str:
    return (
        api_base
        or get_secret_str("COMETAPI_BASE_URL")
        or get_secret_str("COMETAPI_API_BASE")
        or DEFAULT_COMETAPI_API_BASE
    )


def get_cometapi_complete_url(api_base: Optional[str], endpoint: str) -> str:
    base_url = get_cometapi_api_base(api_base).rstrip("/")
    normalized_endpoint = endpoint.strip("/")

    if base_url.endswith(f"/{normalized_endpoint}"):
        return base_url

    if base_url.endswith("/v1"):
        return f"{base_url}/{normalized_endpoint}"

    if "/v1" not in base_url.split("//", 1)[-1]:
        return f"{base_url}/v1/{normalized_endpoint}"

    return f"{base_url}/{normalized_endpoint}"


def require_cometapi_api_key(api_key: Optional[str] = None) -> str:
    final_api_key = get_cometapi_api_key(api_key)
    if not final_api_key:
        raise ValueError("COMETAPI_KEY or COMETAPI_API_KEY is not set")
    return final_api_key


class CometAPIException(BaseLLMException):
    """CometAPI exception handling class"""

    pass
