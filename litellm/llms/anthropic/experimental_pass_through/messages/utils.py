from typing import Dict, cast, get_type_hints

import litellm
from litellm.llms.base_llm.anthropic_messages.transformation import (
    BaseAnthropicMessagesConfig,
)
from litellm.types.llms.anthropic_messages.anthropic_request import (
    AnthropicMessagesRequestOptionalParams,
)


class AnthropicMessagesRequestUtils:
    """Helper utils for constructing Anthropic Messages requests"""

    @staticmethod
    def get_passed_in_optional_params(
        params: Dict,
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

    @staticmethod
    def get_optional_params_anthropic_messages(
        model: str,
        anthropic_messages_provider_config: BaseAnthropicMessagesConfig,
        anthropic_messages_optional_params: AnthropicMessagesRequestOptionalParams,
    ) -> Dict:
        """
        Get optional parameters for the anthropic messages API.

        Args:
            params: Dictionary of all parameters
            model: The model name
            anthropic_messages_provider_config: The provider configuration for anthropic messages API

        Returns:
            A dictionary of supported parameters for the anthropic messages API
        """
        # Remove None values and internal parameters

        # Get supported parameters for the model
        supported_params = anthropic_messages_provider_config.get_supported_anthropic_messages_optional_params(
            model
        )

        # Check for unsupported parameters
        unsupported_params = [
            param
            for param in anthropic_messages_optional_params
            if param not in supported_params
        ]

        if unsupported_params:
            raise litellm.UnsupportedParamsError(
                model=model,
                message=f"The following parameters are not supported for model {model}: {', '.join(unsupported_params)}",
            )

        # Map parameters to provider-specific format
        mapped_params = (
            anthropic_messages_provider_config.map_anthropic_messages_optional_params(
                anthropic_messages_optional_params=anthropic_messages_optional_params,
                model=model,
                drop_params=litellm.drop_params,
            )
        )

        return mapped_params
