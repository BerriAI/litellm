from typing import Any, Dict, cast, get_type_hints

from litellm.types.llms.anthropic import AnthropicMessagesRequestOptionalParams


class AnthropicMessagesRequestUtils:
    @staticmethod
    def get_requested_anthropic_messages_optional_param(
        params: Dict[str, Any],
    ) -> AnthropicMessagesRequestOptionalParams:
        """
        Filter parameters to only include those defined in AnthropicMessagesRequestOptionalParams.

        Args:
            params: Dictionary of parameters to filter

        Returns:
            AnthropicMessagesRequestOptionalParams instance with only the valid parameters
        """
        valid_keys = get_type_hints(AnthropicMessagesRequestOptionalParams).keys()
        filtered_params = {
            k: v for k, v in params.items() if k in valid_keys and v is not None
        }
        return cast(AnthropicMessagesRequestOptionalParams, filtered_params)
