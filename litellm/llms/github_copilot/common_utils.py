"""
Constants for Copilot integration
"""

import os
from collections.abc import MutableMapping
from typing import Final
from urllib.parse import urlsplit

import httpx

from litellm._logging import verbose_logger
from litellm.llms.base_llm.chat.transformation import BaseLLMException

COPILOT_VERSION: Final = "0.26.7"
EDITOR_PLUGIN_VERSION: Final = f"copilot-chat/{COPILOT_VERSION}"
USER_AGENT: Final = f"GitHubCopilotChat/{COPILOT_VERSION}"
API_VERSION: Final = "2025-04-01"
DEFAULT_GITHUB_COPILOT_API_BASE: Final = "https://api.githubcopilot.com"
DEFAULT_COPILOT_INTEGRATION_ID: Final = "vscode-chat"
DEFAULT_COPILOT_EDITOR_VERSION: Final = "vscode/1.115.0"
DEFAULT_COPILOT_EDITOR_PLUGIN_VERSION: Final = EDITOR_PLUGIN_VERSION
DEFAULT_COPILOT_USER_AGENT: Final = USER_AGENT

_COPILOT_AUTH_HEADER_CONFIG = (
    ("accept", "GITHUB_COPILOT_ACCEPT", "application/json"),
    ("content-type", "GITHUB_COPILOT_CONTENT_TYPE", "application/json"),
    ("copilot-integration-id", "GITHUB_COPILOT_INTEGRATION_ID", DEFAULT_COPILOT_INTEGRATION_ID),
    ("editor-version", "GITHUB_COPILOT_EDITOR_VERSION", DEFAULT_COPILOT_EDITOR_VERSION),
    ("editor-plugin-version", "GITHUB_COPILOT_EDITOR_PLUGIN_VERSION", DEFAULT_COPILOT_EDITOR_PLUGIN_VERSION),
    ("user-agent", "GITHUB_COPILOT_USER_AGENT", DEFAULT_COPILOT_USER_AGENT),
)
_COPILOT_REQUEST_HEADER_CONFIG = _COPILOT_AUTH_HEADER_CONFIG + (
    ("openai-intent", "GITHUB_COPILOT_OPENAI_INTENT", None),
    ("x-github-api-version", "GITHUB_COPILOT_API_VERSION", None),
    (
        "x-vscode-user-agent-library-version",
        "GITHUB_COPILOT_USER_AGENT_LIBRARY_VERSION",
        None,
    ),
)


class GithubCopilotError(BaseLLMException):
    def __init__(
        self,
        status_code,
        message,
        request: httpx.Request | None = None,
        response: httpx.Response | None = None,
        headers: httpx.Headers | dict | None = None,
        body: dict | None = None,
    ):
        super().__init__(
            status_code=status_code,
            message=message,
            request=request,
            response=response,
            headers=headers,
            body=body,
        )


class GetDeviceCodeError(GithubCopilotError):
    pass


class GetAccessTokenError(GithubCopilotError):
    pass


class GetAPIKeyError(GithubCopilotError):
    pass


def _get_copilot_header_value(environment_variable: str, default: str | None) -> str | None:
    value = os.getenv(environment_variable)
    if value is None:
        return default
    return value or None


def _get_configured_copilot_headers(
    config: tuple[tuple[str, str, str | None], ...],
) -> dict[str, str]:
    return {
        header: value
        for header, environment_variable, default in config
        if (value := _get_copilot_header_value(environment_variable, default)) is not None
    }


def get_copilot_auth_headers(
    access_token: str | None = None,
) -> dict[str, str]:  # mutable-ok: returned straight into httpx handlers whose headers params require dict
    headers: Final = _get_configured_copilot_headers(_COPILOT_AUTH_HEADER_CONFIG)
    return {**headers, "authorization": f"token {access_token}"} if access_token else headers


def get_copilot_default_headers(
    api_key: str,
) -> dict[str, str]:  # mutable-ok: callers merge caller headers into the returned dict
    return {
        **_get_configured_copilot_headers(_COPILOT_REQUEST_HEADER_CONFIG),
        "Authorization": f"Bearer {api_key}",
    }


def _is_valid_api_base(api_base: str) -> bool:
    parsed_api_base: Final = urlsplit(api_base)
    return (
        parsed_api_base.scheme.lower() in ("http", "https")
        and parsed_api_base.hostname is not None
        and parsed_api_base.username is None
        and parsed_api_base.password is None
        and not parsed_api_base.query
        and not parsed_api_base.fragment
    )


def get_copilot_api_base(api_base: str | None = None) -> str:
    candidates: Final = (
        ("deployment api_base", api_base),
        ("GITHUB_COPILOT_API_BASE", os.getenv("GITHUB_COPILOT_API_BASE")),
    )
    for source, candidate in candidates:
        if not candidate:
            continue
        if _is_valid_api_base(candidate):
            return candidate
        verbose_logger.warning(
            "Ignoring %s because it must be an HTTP(S) URL without credentials, query, or fragment",
            source,
        )
    return DEFAULT_GITHUB_COPILOT_API_BASE


def pin_session_authorization(
    headers: MutableMapping[str, str],  # mutable-ok: the helper fixes caller-owned headers in place
    session_token: str,
) -> None:
    """The stored per-user session token owns Authorization outright: drop every
    caller-supplied bearer header (any casing) so it can never displace the
    session's on the wire."""
    for key in tuple(k for k in headers if k.lower() == "authorization"):
        del headers[key]
    headers["Authorization"] = f"Bearer {session_token}"
