"""
GitHub Copilot Anthropic Messages transformation.

Routes ``/v1/messages`` to Copilot's native Anthropic-compatible endpoint
instead of translating to chat/completions, so native content blocks (notably
PDF ``document`` blocks) pass through untouched. Auth reuses the shared
Authenticator + the standard Copilot headers; beta-header handling is inherited
from ``AnthropicMessagesConfig``.
"""

from typing import Any, Optional

from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)

from ..authenticator import Authenticator
from ..common_utils import (
    DEFAULT_GITHUB_COPILOT_API_BASE,
    get_copilot_default_headers,
)


class GithubCopilotMessagesConfig(AnthropicMessagesConfig):
    """Anthropic Messages config for GitHub Copilot's native /v1/messages."""

    def __init__(self) -> None:
        super().__init__()
        self.authenticator = Authenticator()

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "github_copilot"

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        base_url = api_base or DEFAULT_GITHUB_COPILOT_API_BASE
        if base_url.endswith("/v1/messages"):
            return base_url
        return f"{base_url.rstrip('/')}/v1/messages"

    def validate_anthropic_messages_environment(
        self,
        headers: dict,
        model: str,
        messages: list[Any],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> tuple[dict, Optional[str]]:
        # Apply the standard Copilot headers (bearer token + editor identity)
        # unless the caller already supplied auth.
        has_auth = "authorization" in {k.lower() for k in headers}
        if not has_auth:
            token = api_key or self.authenticator.get_api_key()
            for key, value in get_copilot_default_headers(token).items():
                headers.setdefault(key, value)
        # Delegate to the parent for anthropic-version and the feature-driven
        # anthropic-beta injection (e.g. context_management ->
        # 'context-management-2025-06-27'), which the Copilot backend requires.
        # The parent doesn't recognise our capitalized "Authorization", so drop
        # the api_key to stop it adding a redundant x-api-key alongside it.
        headers, api_base = super().validate_anthropic_messages_environment(
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key=None if not has_auth else api_key,
            api_base=api_base,
        )
        return headers, api_base
