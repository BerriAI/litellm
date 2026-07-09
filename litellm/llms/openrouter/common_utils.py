from typing import Any, Dict, Optional, Union

import httpx

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret, get_secret_str

DEFAULT_OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"


class OpenRouterException(BaseLLMException):
    pass


def get_openrouter_api_base(api_base: Optional[str] = None) -> str:
    return (
        api_base or litellm.api_base or get_secret_str("OPENROUTER_API_BASE") or DEFAULT_OPENROUTER_API_BASE
    ).rstrip("/")


def get_openrouter_endpoint(api_base: Optional[str], endpoint_path: str) -> str:
    base_url = get_openrouter_api_base(api_base)
    normalized_path = endpoint_path.strip("/")
    if not normalized_path:
        return base_url
    if base_url.endswith(f"/{normalized_path}"):
        return base_url
    return f"{base_url}/{normalized_path}"


def get_openrouter_api_key(api_key: Optional[str] = None) -> str:
    resolved_api_key = (
        api_key
        or litellm.api_key
        or litellm.openrouter_key
        or get_secret_str("OPENROUTER_API_KEY")
        or get_secret_str("OR_API_KEY")
    )
    if not resolved_api_key:
        raise ValueError(
            "OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable or pass api_key parameter."
        )
    return resolved_api_key


def get_openrouter_headers(
    api_key: Optional[str] = None,
    headers: Optional[Dict[str, Any]] = None,
    content_type: Optional[str] = "application/json",
) -> Dict[str, Any]:
    openrouter_headers: Dict[str, Any] = {
        "Authorization": f"Bearer {get_openrouter_api_key(api_key)}",
        "HTTP-Referer": get_secret("OR_SITE_URL") or "https://litellm.ai",
        "X-Title": get_secret("OR_APP_NAME") or "liteLLM",
    }
    if content_type is not None:
        openrouter_headers["Content-Type"] = content_type
    openrouter_headers.update(headers or {})
    return openrouter_headers


def get_openrouter_error_message(response_json: Union[Dict[str, Any], Any], default: str) -> str:
    if isinstance(response_json, dict):
        error = response_json.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if message:
                return str(message)
        if isinstance(error, str):
            return error
        message = response_json.get("message")
        if message:
            return str(message)
    return default


def raise_openrouter_error(raw_response: httpx.Response) -> None:
    if 200 <= raw_response.status_code < 300:
        return

    response_json: Union[Dict[str, Any], Any]
    try:
        response_json = raw_response.json()
    except ValueError:
        response_json = {}

    raise OpenRouterException(
        message=get_openrouter_error_message(
            response_json=response_json,
            default=raw_response.text,
        ),
        status_code=raw_response.status_code,
        headers=raw_response.headers,
    )
