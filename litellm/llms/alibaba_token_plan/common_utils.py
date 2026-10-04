from collections.abc import Mapping
from typing import Final
from urllib.parse import urlsplit, urlunsplit

import litellm
from litellm.secret_managers.main import get_secret_str

DEFAULT_API_BASE: Final = "https://token-plan.ap-southeast-1.maas.aliyuncs.com"
DEFAULT_CHAT_API_BASE: Final = f"{DEFAULT_API_BASE}/compatible-mode/v1"
DEFAULT_MESSAGES_API_BASE: Final = f"{DEFAULT_API_BASE}/apps/anthropic"
CHAT_ENDPOINT: Final = "compatible-mode/v1/chat/completions"
IMAGE_ENDPOINT: Final = "api/v1/services/aigc/multimodal-generation/generation"
SPEECH_ENDPOINT: Final = "api/v1/services/audio/tts/SpeechSynthesizer"
REALTIME_ENDPOINT: Final = "api-ws/v1/realtime"
VIDEO_ENDPOINT: Final = "api/v1/services/aigc/video-generation/video-synthesis"

_OFFICIAL_HOSTNAME: Final = urlsplit(DEFAULT_API_BASE).hostname


def get_api_key(api_key: str | None) -> str | None:
    return api_key or get_secret_str("ALIBABA_TOKEN_PLAN_API_KEY") or litellm.api_key


def require_api_key(api_key: str | None, model: str = "") -> str:
    resolved_key: Final = get_api_key(api_key)
    if not resolved_key:
        raise litellm.AuthenticationError(
            message="ALIBABA_TOKEN_PLAN_API_KEY is not set; provide a Token Plan api_key",
            llm_provider="alibaba_token_plan",
            model=model,
        )
    return resolved_key


def _normalized_api_base(api_base: str) -> str:
    parsed: Final = urlsplit(api_base)
    hostname: Final = (parsed.hostname or "").encode("idna").decode("ascii").rstrip(".")
    official: Final = hostname == _OFFICIAL_HOSTNAME
    if official and parsed.scheme not in ("https", "wss"):
        raise ValueError("Alibaba Token Plan endpoints require HTTPS or WSS")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), parsed.query, parsed.fragment))


def _configured_api_base(api_base: str | None) -> str | None:
    return api_base or get_secret_str("ALIBABA_TOKEN_PLAN_API_BASE")


def _resolve_api_url(api_base: str, endpoint: str) -> str:
    normalized: Final = _normalized_api_base(api_base)
    parsed: Final = urlsplit(normalized)
    common_paths: Final = (
        "/compatible-mode/v1/chat/completions",
        "/compatible-mode/v1",
        "/apps/anthropic/v1/messages",
        "/apps/anthropic",
    )
    root_path: Final = next(
        (parsed.path.removesuffix(suffix) for suffix in common_paths if parsed.path.endswith(suffix)),
        None if parsed.path else "",
    )
    if root_path is None:
        return normalized
    path: Final = f"{root_path}/{endpoint.lstrip('/')}".rstrip("/")
    return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment))


def get_api_base(api_base: str | None) -> str:
    """Resolve the OpenAI-compatible base URL used by chat and Responses."""

    configured_base: Final = _configured_api_base(api_base) or DEFAULT_CHAT_API_BASE
    normalized: Final = _resolve_api_url(configured_base, "compatible-mode/v1")
    parsed: Final = urlsplit(normalized)
    if parsed.query or parsed.fragment:
        raise ValueError("Alibaba Token Plan chat api_base must not include a query or fragment")
    if parsed.path.endswith("/chat/completions"):
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.removesuffix("/chat/completions"), "", ""))
    return normalized


def get_messages_api_base(api_base: str | None) -> str:
    """Resolve the base URL for the native Anthropic Messages endpoint."""

    configured_base: Final = _configured_api_base(api_base)
    if configured_base is None:
        return DEFAULT_MESSAGES_API_BASE
    normalized: Final = _resolve_api_url(configured_base, "apps/anthropic")
    parsed: Final = urlsplit(normalized)
    if parsed.query or parsed.fragment:
        raise ValueError("Alibaba Token Plan Messages api_base must not include a query or fragment")
    return normalized


def get_native_api_url(api_base: str | None, endpoint: str) -> str:
    """Resolve a native operation from a shared base or an explicit operation URL."""

    configured_base: Final = _configured_api_base(api_base)
    if configured_base is None:
        return f"{DEFAULT_API_BASE}/{endpoint.lstrip('/')}"
    return _resolve_api_url(configured_base, endpoint)


def validate_headers(
    headers: Mapping[str, str], api_key: str | None
) -> dict[str, str]:  # mutable-ok: provider interface requires a mutable return value
    resolved_key: Final = require_api_key(api_key)
    return {
        **{key: value for key, value in headers.items() if key.lower() not in ("authorization", "content-type")},
        "Authorization": f"Bearer {resolved_key}",
        "Content-Type": "application/json",
    }
