"""
Constants for Copilot integration
"""

from collections.abc import Mapping, MutableMapping, Sequence
from typing import Final
from uuid import uuid4

import httpx
from pydantic import BaseModel, TypeAdapter

from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.utils import PromptTokensDetailsWrapper, Usage

# Constants
COPILOT_VERSION: Final = "0.26.7"
EDITOR_PLUGIN_VERSION: Final = f"copilot-chat/{COPILOT_VERSION}"
USER_AGENT: Final = f"GitHubCopilotChat/{COPILOT_VERSION}"
API_VERSION: Final = "2025-04-01"
DEFAULT_GITHUB_COPILOT_API_BASE: Final = "https://api.githubcopilot.com"

_TOKEN_DETAILS: Final = TypeAdapter(tuple[object, ...])


def _copilot_cache_count(entries: Sequence[object], token_type: str) -> int | None:
    mappings: Final = tuple(as_str_mapping(entry) for entry in entries)
    counts: Final = tuple(
        entry.get("token_count") for entry in mappings if entry is not None and entry.get("token_type") == token_type
    )
    valid_counts: Final = tuple(
        count for count in counts if isinstance(count, int) and not isinstance(count, bool) and count >= 0
    )
    return sum(valid_counts) if valid_counts else None


def normalize_copilot_usage(usage: Mapping[str, object], copilot_usage: object = None) -> Mapping[str, object]:
    outer: Final = as_str_mapping(copilot_usage)
    metadata: Final = outer if outer is not None else as_str_mapping(usage.get("copilot_usage"))
    token_details: Final = metadata.get("token_details") if metadata is not None else None
    if not isinstance(token_details, list):
        return usage
    entries: Final = _TOKEN_DETAILS.validate_python(token_details)
    details: Final = PromptTokensDetailsWrapper.model_validate(
        as_str_mapping(usage.get("prompt_tokens_details")) or {}
    ).model_dump(exclude_unset=True)
    candidates: Final = (
        ("cache_read_input_tokens", "cached_tokens", _copilot_cache_count(entries, "cache_read")),
        ("cache_creation_input_tokens", "cache_write_tokens", _copilot_cache_count(entries, "cache_write")),
    )
    counts: Final = {
        field: details.get(detail_field) if details.get(detail_field) is not None else count
        for field, detail_field, count in candidates
        if usage.get(field) is None and (details.get(detail_field) is not None or count is not None)
    }
    return {**usage, **counts}


def normalize_copilot_chunk(chunk: BaseModel) -> BaseModel:
    usage: Final[object] = getattr(chunk, "usage", None)
    if not isinstance(usage, BaseModel):
        return chunk
    copilot_usage: Final[object] = (chunk.model_extra or {}).get("copilot_usage")
    if copilot_usage is None and "copilot_usage" not in (usage.model_extra or {}):
        return chunk
    normalized: Final = normalize_copilot_usage(usage.model_dump(exclude_unset=True), copilot_usage)
    return chunk.model_copy(update={"usage": Usage.model_validate(normalized)})


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
        "copilot-integration-id": "vscode-chat",
        "editor-version": "vscode/1.95.0",  # Fixed version for stability
        "editor-plugin-version": EDITOR_PLUGIN_VERSION,
        "user-agent": USER_AGENT,
        "openai-intent": "conversation-panel",
        "x-github-api-version": API_VERSION,
        "x-request-id": str(uuid4()),
        "x-vscode-user-agent-library-version": "electron-fetch",
    }


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
