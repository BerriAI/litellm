"""
Constants for Copilot integration
"""

from typing import Final
from uuid import uuid4

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException

# Constants
COPILOT_VERSION: Final = "0.26.7"
EDITOR_PLUGIN_VERSION: Final = f"copilot-chat/{COPILOT_VERSION}"
USER_AGENT: Final = f"GitHubCopilotChat/{COPILOT_VERSION}"
API_VERSION: Final = "2025-04-01"
DEFAULT_GITHUB_COPILOT_API_BASE: Final = "https://api.githubcopilot.com"


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


class APIKeyExpiredError(GithubCopilotError):
    pass


class RefreshAPIKeyError(GithubCopilotError):
    pass


class GetAPIKeyError(GithubCopilotError):
    pass


def get_copilot_default_headers(api_key: str) -> dict:
    """
    Get default headers for GitHub Copilot Responses API.

    Based on copilot-api's header configuration.
    """
    return {
        "Authorization": f"Bearer {api_key}",
        "content-type": "application/json",
        "copilot-integration-id": "vscode-chat",
        "editor-version": "vscode/1.95.0",  # Fixed version for stability
        "editor-plugin-version": EDITOR_PLUGIN_VERSION,
        "user-agent": USER_AGENT,
        "openai-intent": "conversation-panel",
        "x-github-api-version": API_VERSION,
        "x-request-id": str(uuid4()),
        "x-vscode-user-agent-library-version": "electron-fetch",
    }
