from typing import Any, Optional, Tuple, cast, List

from litellm.exceptions import AuthenticationError
from litellm.llms.openai.openai import OpenAIConfig
from litellm.types.llms.openai import AllMessageValues

from ..authenticator import Authenticator
from ..common_utils import GetAPIKeyError, GITHUB_COPILOT_API_BASE


class GithubCopilotConfig(OpenAIConfig):
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        custom_llm_provider: str = "openai",
    ) -> None:
        super().__init__()
        self.authenticator = Authenticator()

    def _get_openai_compatible_provider_info(
        self,
        model: str,
        api_base: Optional[str],
        api_key: Optional[str],
        custom_llm_provider: str,
    ) -> Tuple[Optional[str], Optional[str], str]:
        dynamic_api_base = (
            self.authenticator.get_api_base() or GITHUB_COPILOT_API_BASE
        )
        try:
            dynamic_api_key = self.authenticator.get_api_key()
        except GetAPIKeyError as e:
            raise AuthenticationError(
                model=model,
                llm_provider=custom_llm_provider,
                message=str(e),
            )
        return dynamic_api_base, dynamic_api_key, custom_llm_provider

    def _transform_messages(
        self,
        messages,
        model: str,
    ):
        import litellm

        disable_copilot_system_to_assistant = (
            litellm.disable_copilot_system_to_assistant
        )
        if not disable_copilot_system_to_assistant:
            for message in messages:
                if "role" in message and message["role"] == "system":
                    cast(Any, message)["role"] = "assistant"
        return messages

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        # Get base headers from parent
        validated_headers = super().validate_environment(
            headers, model, messages, optional_params, litellm_params, api_key, api_base
        )

        # Add X-Initiator header based on message roles
        initiator = self._determine_initiator(messages)
        validated_headers["X-Initiator"] = initiator

        # Add Copilot-Vision-Request header if request contains images
        if self._has_vision_content(messages):
            validated_headers["Copilot-Vision-Request"] = "true"

        return validated_headers

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get supported OpenAI parameters for GitHub Copilot.

        For Claude models that support extended thinking (Claude 4 family and Claude 3-7), includes thinking and reasoning_effort parameters.
        For other models, returns standard OpenAI parameters (which may include reasoning_effort for o-series models).
        """
        from litellm.utils import supports_reasoning
        
        # Get base OpenAI parameters
        base_params = super().get_supported_openai_params(model)

        # Add Claude-specific parameters for models that support extended thinking
        if "claude" in model.lower() and supports_reasoning(
            model=model.lower(),
        ):
            if "thinking" not in base_params:
                base_params.append("thinking")
            # reasoning_effort is not included by parent for Claude models, so add it
            if "reasoning_effort" not in base_params:
                base_params.append("reasoning_effort")

        return base_params

    def _determine_initiator(self, messages: List[AllMessageValues]) -> str:
        """
        Determine if request is user or agent initiated based on message roles.
        Returns 'agent' if any message has role 'tool' or 'assistant', otherwise 'user'.
        """
        for message in messages:
            role = message.get("role")
            if role in ["tool", "assistant"]:
                return "agent"
        return "user"

    def _has_vision_content(self, messages: List[AllMessageValues]) -> bool:
        """
        Check if any message contains vision content (images).
        Returns True if any message has content with vision-related types, otherwise False.
        
        Checks for:
        - image_url content type (OpenAI format)
        - Content items with type 'image_url'
        """
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                # Check if any content item indicates vision content
                for content_item in content:
                    if isinstance(content_item, dict):
                        # Check for image_url field (direct image URL)
                        if "image_url" in content_item:
                            return True
                        # Check for type field indicating image content
                        content_type = content_item.get("type")
                        if content_type == "image_url":
                            return True
        return False
