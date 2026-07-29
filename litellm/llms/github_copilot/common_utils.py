"""
Constants for Copilot integration
"""

from typing import Optional, Union
from uuid import uuid4

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException

# Constants
COPILOT_VERSION = "0.26.7"
EDITOR_PLUGIN_VERSION = f"copilot-chat/{COPILOT_VERSION}"
USER_AGENT = f"GitHubCopilotChat/{COPILOT_VERSION}"
API_VERSION = "2025-04-01"
DEFAULT_GITHUB_COPILOT_API_BASE = "https://api.githubcopilot.com"


def get_copilot_initiator(input_param: object) -> str:
    """Classify a Copilot request as user- or agent-initiated.

    Responses API history contains assistant/tool/function-call/reasoning items
    as well as role-less items. Any such prior item means the request continues
    an agent turn; a string or user-only input is a new user turn.
    """
    if isinstance(input_param, str):
        return "user"
    if isinstance(input_param, list):
        for item in input_param:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            if not role or (isinstance(role, str) and role.lower() in {"assistant", "tool"}):
                return "agent"
            if item.get("type") in {"function_call", "function_call_output", "reasoning"}:
                return "agent"
    return "user"


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
