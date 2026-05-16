from typing import Any, List, Optional, Tuple

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


class GithubCopilotAnthropicMessagesConfig(AnthropicMessagesConfig):
    """
    GitHub Copilot implementation of Anthropic messages API.
    Routes requests to Copilot's /v1/messages endpoint with appropriate authentication and headers.
    """

    def __init__(self) -> None:
        super().__init__()
        self.authenticator = Authenticator()

    def validate_anthropic_messages_environment(
        self,
        headers: dict,
        model: str,
        messages: List[Any],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Tuple[dict, Optional[str]]:
        """
        Validate environment for GitHub Copilot and add Copilot-specific headers.

        The caller-supplied ``api_base`` is intentionally ignored. Routing this
        request anywhere other than the authenticated Copilot endpoint would
        leak the Copilot bearer token to a caller-controlled URL.
        """
        # Always use the Copilot endpoint resolved from the authenticated
        # session, never the caller-supplied api_base.
        dynamic_api_base = (
            self.authenticator.get_api_base() or DEFAULT_GITHUB_COPILOT_API_BASE
        )
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

        # Set Anthropic version for messages API
        if "anthropic-version" not in headers:
            headers["anthropic-version"] = "2023-06-01"

        # Auto-inject anthropic-beta headers for advanced features
        # (context_management, tool_search, output_format, speed)
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

        The caller-supplied ``api_base`` is intentionally ignored to avoid
        leaking the Copilot bearer token to a caller-controlled URL.
        """
        resolved = self.authenticator.get_api_base() or DEFAULT_GITHUB_COPILOT_API_BASE
        if not resolved.endswith("/v1/messages"):
            resolved = f"{resolved}/v1/messages"
        return resolved
