"""
Common utilities for Volcengine LLM provider
"""

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str


class VolcEngineError(BaseLLMException):
    """
    Custom exception class for Volcengine provider errors.
    """

    def __init__(
        self, status_code: int, message: str, headers: httpx.Headers | None = None
    ):
        self.status_code = status_code
        self.message = message
        self.headers = headers or httpx.Headers()
        super().__init__(status_code=status_code, message=message, headers=dict(self.headers))


def get_volcengine_base_url(api_base: str | None = None) -> str:
    """
    Get the base URL for Volcengine API calls.

    Args:
        api_base: Optional custom API base URL

    Returns:
        The base URL to use for API calls
    """
    if api_base:
        return api_base
    return "https://ark.cn-beijing.volces.com"


def get_volcengine_headers(api_key: str, extra_headers: dict | None = None) -> dict:
    """
    Get headers for Volcengine API calls.

    Args:
        api_key: The API key for authentication
        extra_headers: Optional additional headers

    Returns:
        Dictionary of headers
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    if extra_headers:
        headers.update(extra_headers)

    return headers


def get_volcengine_speech_api_key(api_key: str | None) -> str:
    resolved_key = api_key or get_secret_str("VOLCENGINE_SPEECH_KEY")
    if not resolved_key or not resolved_key.strip():
        raise VolcEngineError(
            status_code=401,
            message=(
                "Volcengine Speech key is required. Set VOLCENGINE_SPEECH_KEY "
                "as a Speech API Key."
            ),
        )
    resolved_key = resolved_key.strip()
    if ":" in resolved_key:
        raise VolcEngineError(
            status_code=401,
            message=(
                "Volcengine Speech key must be a single Speech API Key. "
                "Legacy '<appId>:<accessKey>' credentials are not supported."
            ),
        )
    return resolved_key


def get_volcengine_configured_ws_api_base(
    litellm_params: dict | None, default_api_base: str
) -> str:
    """Resolve Volcengine speech WebSocket endpoints from trusted config data."""
    if litellm_params is None:
        return default_api_base

    metadata = litellm_params.get("metadata")
    if isinstance(metadata, dict):
        metadata_api_base = metadata.get("api_base")
        if _is_volcengine_ws_api_base(metadata_api_base):
            return metadata_api_base

    configured_api_base = litellm_params.get("api_base")
    if _is_volcengine_ws_api_base(configured_api_base):
        return configured_api_base

    return default_api_base


def _is_volcengine_ws_api_base(value: object) -> bool:
    return isinstance(value, str) and value.startswith(("ws://", "wss://"))
