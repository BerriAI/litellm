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
        """
        # Get Copilot auth credentials
        dynamic_api_base = (
            api_base
            or self.authenticator.get_api_base()
            or DEFAULT_GITHUB_COPILOT_API_BASE
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
        """
        api_base = api_base or DEFAULT_GITHUB_COPILOT_API_BASE
        if not api_base.endswith("/v1/messages"):
            api_base = f"{api_base}/v1/messages"
        return api_base
