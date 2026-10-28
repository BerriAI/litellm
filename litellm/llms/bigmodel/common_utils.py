from types import MappingProxyType
from typing import Final, Literal
from urllib.parse import urlsplit, urlunsplit

from litellm.secret_managers.main import get_secret_str

BIGMODEL_API_BASE: Final = "https://open.bigmodel.cn"
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
