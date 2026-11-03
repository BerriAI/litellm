import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Literal
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import JsonValue, TypeAdapter, ValidationError

import litellm
from litellm.exceptions import ContentPolicyViolationError, ContextWindowExceededError
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str

BIGMODEL_API_BASE: Final = "https://open.bigmodel.cn"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_BUSINESS_CODE_IN_ERROR: Final = re.compile(r"""['"]code['"]\s*:\s*['"]?(\d{4})\b""")
_CODING_CHAT_PATH: Final = "/api/coding/paas/v4"
_API_PATHS: Final = (
    "/api/coding/paas/v4/chat/completions",
    _CODING_CHAT_PATH,
    "/api/paas/v4/chat/completions",
    "/api/anthropic/v1/messages",
    "/api/anthropic/v1",
    "/api/v1/responses",
    "/api/paas/v4",
    "/api/anthropic",
    "/api/v1",
    "/api",
)
_ENDPOINT_PATHS: Final = MappingProxyType(
    {
        "chat": "/api/paas/v4",
        "responses": "/api/v1/responses",
        "messages": "/api/anthropic/v1/messages",
    }
)


def get_bigmodel_api_key(api_key: str | None) -> str | None:
    return api_key or get_secret_str("BIGMODEL_API_KEY")


def get_bigmodel_url(api_base: str | None, endpoint: Literal["chat", "responses", "messages"]) -> str:
    resolved_base: Final = api_base or get_secret_str("BIGMODEL_API_BASE") or BIGMODEL_API_BASE
    parsed: Final = urlsplit(resolved_base)
    path: Final = parsed.path.rstrip("/")
    suffix: Final = next((candidate for candidate in _API_PATHS if path.endswith(candidate)), "")
    endpoint_path: Final = (
        _CODING_CHAT_PATH if endpoint == "chat" and suffix.startswith(_CODING_CHAT_PATH) else _ENDPOINT_PATHS[endpoint]
    )
    return urlunsplit(
        (parsed.scheme, parsed.netloc, f"{path.removesuffix(suffix)}{endpoint_path}", parsed.query, parsed.fragment)
    )


def bigmodel_json_body(response: httpx.Response) -> Mapping[str, JsonValue]:
    try:
        return _JSON_OBJECT.validate_json(response.content)
    except ValidationError:
        raise BaseLLMException(message=response.text, status_code=response.status_code, headers=response.headers)


def is_unsupported_disabled_thinking(model: str, thinking: object) -> bool:
    return thinking == {"type": "disabled"} and litellm.utils.supports_factory(
        model=model, custom_llm_provider="bigmodel", key="thinking_always_on"
    )


def bigmodel_business_error(
    code: JsonValue, message: str, model: str, response: httpx.Response | None
) -> ContextWindowExceededError | ContentPolicyViolationError | None:
    match code:
        case 1261 | "1261":
            return ContextWindowExceededError(message=message, model=model, llm_provider="bigmodel", response=response)
        case 1301 | "1301":
            return ContentPolicyViolationError(message=message, model=model, llm_provider="bigmodel", response=response)
        case _:
            return None


def raise_bigmodel_business_error(model: object, error_str: str, original_exception: object) -> None:
    found: Final = _BUSINESS_CODE_IN_ERROR.search(error_str)
    response: Final = getattr(original_exception, "response", None)
    error: Final = bigmodel_business_error(
        found.group(1) if found else None,
        error_str,
        str(model),
        response if isinstance(response, httpx.Response) else None,
    )
    if error is not None:
        raise error
