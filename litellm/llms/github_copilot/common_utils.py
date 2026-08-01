"""
Constants for Copilot integration
"""

import os
from typing import Optional, Union

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException

DEFAULT_GITHUB_COPILOT_API_BASE = "https://api.githubcopilot.com"
DEFAULT_COPILOT_INTEGRATION_ID = "vscode-chat"
DEFAULT_COPILOT_EDITOR_VERSION = "vscode/1.115.0"
DEFAULT_COPILOT_EDITOR_PLUGIN_VERSION = "copilot-chat/0.44.0"
DEFAULT_COPILOT_USER_AGENT = "GitHubCopilotChat/0.44.0"

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
        request: Optional[httpx.Request] = None,
        response: Optional[httpx.Response] = None,
        headers: Optional[Union[httpx.Headers, dict]] = None,
        body: Optional[dict] = None,
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


def get_copilot_auth_headers() -> dict[str, str]:
    return _get_configured_copilot_headers(_COPILOT_AUTH_HEADER_CONFIG)


def get_copilot_default_headers(api_key: str) -> dict[str, str]:
    return {
        **_get_configured_copilot_headers(_COPILOT_REQUEST_HEADER_CONFIG),
        "Authorization": f"Bearer {api_key}",
    }
