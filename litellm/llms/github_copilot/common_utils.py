"""
Constants for Copilot integration
"""

import threading
from collections import OrderedDict
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


# Header name for conversation session correlation.
# CONFIRMED NO-OP: X-Initiator alone controls premium billing. Neither CopilotApi
# (ericc-ch/copilot-api) nor OpenCode (sst/opencode-copilot-auth) send this header —
# both rely solely on X-Initiator. Verified in Phase 3: direct API tests with
# and without metadata["copilot_conversation_id"] both accepted by the Copilot
# API; billing is governed by X-Initiator header, not session correlation.
# Header retained for optional caller opt-in.
COPILOT_CONVERSATION_ID_HEADER = (
    "x-conversation-id"  # CONFIRMED NO-OP for billing; X-Initiator sufficient
)


def get_copilot_default_headers(
    api_key: str,
    conversation_key: Optional[str] = None,
) -> dict:
    """
    Get default headers for GitHub Copilot API requests.

    Based on copilot-api's header configuration.

    Args:
        api_key: GitHub Copilot API key.
        conversation_key: Optional stable conversation identifier (caller-supplied).
            When provided, a consistent conversation_id is included in headers to
            enable session-scoped billing (only the first turn is charged as
            a premium request). Pass metadata["copilot_conversation_id"] from
            the LiteLLM call's metadata dict.
            When None (default), omits conversation_id and preserves
            existing per-request behavior (backward compatible).
    """
    headers = {
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

    if conversation_key is not None:
        conversation_id = get_or_create_conversation_id(conversation_key)
        # Header confirmed NO-OP for billing — see COPILOT_CONVERSATION_ID_HEADER.
        headers[COPILOT_CONVERSATION_ID_HEADER] = conversation_id

    return headers


# ---------------------------------------------------------------------------
# Conversation ID store — stable per-conversation identifier for billing
# ---------------------------------------------------------------------------

_MAX_CONVERSATION_STORE = 10_000
_conversation_store: OrderedDict = OrderedDict()
_conversation_store_lock = threading.Lock()


def get_or_create_conversation_id(conversation_key: str) -> str:
    """
    Return the stable conversation_id for a given key, creating one if new.

    The conversation_key must be globally unique per user-conversation.
    Recommended: pass metadata["copilot_conversation_id"] from the caller.

    The store is process-local and ephemeral — restarts start fresh,
    which is acceptable since GitHub treats missing conversation_id as
    a new conversation anyway.

    Args:
        conversation_key: Stable, unique identifier for the conversation.

    Returns:
        UUID string to use as the conversation_id header value.
    """
    with _conversation_store_lock:
        if conversation_key in _conversation_store:
            _conversation_store.move_to_end(conversation_key)
            return _conversation_store[conversation_key]
        new_id = str(uuid4())
        _conversation_store[conversation_key] = new_id
        if len(_conversation_store) > _MAX_CONVERSATION_STORE:
            _conversation_store.popitem(last=False)
        return new_id


# ---------------------------------------------------------------------------
# Shared X-Initiator helper — unified across Chat and Responses API
# ---------------------------------------------------------------------------


def determine_x_initiator(messages_or_input: Union[list, str]) -> str:
    """
    Determine X-Initiator header value for GitHub Copilot requests.

    Unified helper covering both Chat API messages and Responses API input items.
    Uses the Responses API logic (superset) since it handles role-less items
    that the Chat API does not encounter.

    Returns "agent" if:
    - Input is a list containing any item with role "assistant" or "tool"
    - Input is a list containing any item with no "role" key
      (Responses API: function_call, function_call_output, mcp_call,
       mcp_call_result, reasoning items)

    Returns "user" if:
    - Input is a string (single-turn user prompt)
    - Input is a list with only user/system role items

    Args:
        messages_or_input: Chat API messages list or Responses API input param.

    Returns:
        "agent" or "user"
    """
    if isinstance(messages_or_input, str):
        return "user"

    if isinstance(messages_or_input, list):
        for item in messages_or_input:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            # No role = agent-initiated (Responses API role-less item types)
            if role is None:
                return "agent"
            # assistant, tool, or legacy function role = agent continuation
            if role in ("assistant", "tool", "function"):
                return "agent"

    return "user"
