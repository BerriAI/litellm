from typing import Any, Optional

from litellm.exceptions import AuthenticationError
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)

from ..authenticator import Authenticator
from ..common_utils import (
    DEFAULT_GITHUB_COPILOT_API_BASE,
    GetAPIKeyError,
    get_copilot_default_headers,
)

_MESSAGES_PROXY_API_VERSION = "2026-06-01"


class GithubCopilotAnthropicMessagesConfig(AnthropicMessagesConfig):
    """
    GitHub Copilot implementation of Anthropic messages API.
    Routes requests to Copilot's /v1/messages endpoint with appropriate authentication and headers.
    """

    def __init__(self) -> None:
        super().__init__()
        self.authenticator = Authenticator()

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "github_copilot"

    def handles_web_search_natively(self) -> bool:
        """
        Copilot's /v1/messages endpoint does not execute ``web_search`` tools, so
        the interception handler must short-circuit web-search-only requests
        instead of routing them here.
        """
        return False

    def should_filter_anthropic_beta_headers(self) -> bool:
        """
        Copilot's /v1/messages is a native Anthropic Messages passthrough, so
        ``anthropic-beta`` values injected by ``_update_headers_with_anthropic_beta``
        (context_management, structured outputs, ...) must reach the upstream
        verbatim. The default provider-scoped filter would drop them because
        github_copilot has no entry in ``anthropic_beta_headers_config.json``.
        """
        return False

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
        """
        Validate environment for GitHub Copilot and add Copilot-specific headers.

        The caller-supplied ``api_base`` is intentionally ignored. Routing this
        request anywhere other than the authenticated Copilot endpoint would
        leak the Copilot bearer token to a caller-controlled URL.
        """
        # Always use the Copilot endpoint resolved from the authenticated
        # session, never the caller-supplied api_base. rstrip so a
        # tenant-specific base with a trailing slash does not yield a
        # double-slash URL once "/v1/messages" is appended downstream.
        dynamic_api_base = (self.authenticator.get_api_base() or DEFAULT_GITHUB_COPILOT_API_BASE).rstrip("/")
        try:
            dynamic_api_key = self.authenticator.get_api_key()
        except GetAPIKeyError as e:
            raise AuthenticationError(
                model=model,
                llm_provider="github_copilot",
                message=str(e),
            )

        # Merge Copilot headers with provided headers
        copilot_headers = get_copilot_default_headers(dynamic_api_key)
        for key, value in copilot_headers.items():
            if key not in headers:
                headers[key] = value

        headers["openai-intent"] = "messages-proxy"
        headers["x-interaction-type"] = "messages-proxy"
        headers["x-github-api-version"] = _MESSAGES_PROXY_API_VERSION

        if "anthropic-version" not in headers:
            headers["anthropic-version"] = "2023-06-01"

        headers = self._update_headers_with_anthropic_beta(
            headers, optional_params, custom_llm_provider="github_copilot"
        )

        return headers, dynamic_api_base

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        Return the complete URL for GitHub Copilot /v1/messages endpoint.

        ``api_base`` here is the value already resolved by
        ``validate_anthropic_messages_environment`` (the authenticated Copilot
        host), not the raw caller-supplied base — that one is discarded there to
        avoid leaking the Copilot bearer token to a caller-controlled URL. We
        reuse it to avoid a second authenticator read, falling back to a fresh
        resolution only if it was not provided.
        """
        resolved = (api_base or self.authenticator.get_api_base() or DEFAULT_GITHUB_COPILOT_API_BASE).rstrip("/")
        if not resolved.endswith("/v1/messages"):
            resolved = f"{resolved}/v1/messages"
        return resolved
