"""
Constants for Copilot integration
"""

import os
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Final, Literal
from uuid import uuid4

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException

# Constants
COPILOT_VERSION: Final = "0.26.7"
EDITOR_PLUGIN_VERSION: Final = f"copilot-chat/{COPILOT_VERSION}"
USER_AGENT: Final = f"GitHubCopilotChat/{COPILOT_VERSION}"
API_VERSION: Final = "2025-04-01"
DEFAULT_GITHUB_COPILOT_API_BASE: Final = "https://api.githubcopilot.com"
DEFAULT_GITHUB_CLIENT_ID: Final = "Iv1.b507a08c87ecfe98"
VSCODE_CHAT_INTEGRATION_ID: Final = "vscode-chat"
COPILOT_CLI_INTEGRATION_ID: Final = "copilot-developer-cli"


def custom_github_app_hint() -> str | None:
    """Explain the GitHub App permission a custom Copilot sign-in app needs, or None for the default app."""
    if os.getenv("GITHUB_COPILOT_CLIENT_ID", DEFAULT_GITHUB_CLIENT_ID) in ("", DEFAULT_GITHUB_CLIENT_ID):
        return None
    return (
        "LiteLLM signs in to GitHub Copilot with a custom GitHub App (GITHUB_COPILOT_CLIENT_ID). Check that the "
        "app's 'Copilot Requests' permission is set to 'Read-only' and that the app's installation accepted it: "
        "open https://github.com/settings/apps (or your organization's Settings > Developer settings > GitHub Apps), "
        "select the app, then Permissions & events > Organization permissions > Copilot Requests."
    )


class GithubCopilotError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message: str,
        request: httpx.Request | None = None,
        response: httpx.Response | None = None,
        headers: httpx.Headers | dict[str, str] | None = None,
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


def copilot_integration_id() -> str:
    """Copilot serves a custom GitHub App's user tokens only as the Copilot CLI integration."""
    client_id: Final = os.getenv("GITHUB_COPILOT_CLIENT_ID", DEFAULT_GITHUB_CLIENT_ID)
    if client_id in ("", DEFAULT_GITHUB_CLIENT_ID):
        return VSCODE_CHAT_INTEGRATION_ID
    return COPILOT_CLI_INTEGRATION_ID


def get_copilot_default_headers(
    api_key: str,
) -> dict[str, str]:  # mutable-ok: callers merge caller headers into the returned dict
    """
    Get default headers for GitHub Copilot Responses API.

    Based on copilot-api's header configuration.
    """
    return {
        "Authorization": f"Bearer {api_key}",
        "content-type": "application/json",
        "copilot-integration-id": copilot_integration_id(),
        "editor-version": "vscode/1.95.0",  # Fixed version for stability
        "editor-plugin-version": EDITOR_PLUGIN_VERSION,
        "user-agent": USER_AGENT,
        "openai-intent": "conversation-panel",
        "x-github-api-version": API_VERSION,
        "x-request-id": str(uuid4()),
        "x-vscode-user-agent-library-version": "electron-fetch",
    }


_AGENT_ROLES: Final = frozenset({"assistant", "tool"})


def _continues_agent_turn(item: Mapping[str, object]) -> bool:
    if not isinstance(item, Mapping):  # pyright: ignore[reportUnnecessaryIsInstance]  # items arrive from untyped request bodies
        return False
    role: Final = item.get("role")
    return not role or (isinstance(role, str) and role.lower() in _AGENT_ROLES)


def get_copilot_initiator(input_param: str | Sequence[Mapping[str, object]]) -> Literal["user", "agent"]:
    """Copilot bills a request as a premium user request unless X-Initiator says an agent turn is continuing.
    Assistant, tool and role-less history items (function calls, their outputs, reasoning) mark an agent turn."""
    if isinstance(input_param, str):
        return "user"
    return "agent" if any(_continues_agent_turn(item) for item in input_param) else "user"


def pin_session_authorization(
    headers: MutableMapping[str, str],  # mutable-ok: the helper fixes caller-owned headers in place
    session_token: str,
    integration_id: str | None = None,
) -> None:
    """The stored per-user session token owns Authorization outright: drop every
    caller-supplied bearer header (any casing) so it can never displace the
    session's on the wire."""
    for key in tuple(k for k in headers if k.lower() == "authorization"):
        del headers[key]
    headers["Authorization"] = f"Bearer {session_token}"
    if integration_id is not None:
        for key in tuple(k for k in headers if k.lower() == "copilot-integration-id"):
            del headers[key]  # rebind-ok: pin the authenticated identity in caller-owned headers
        headers["copilot-integration-id"] = integration_id  # rebind-ok: pin the authenticated identity
