from typing import List, Optional, Tuple

import os

from litellm.exceptions import AuthenticationError
from litellm.llms.openai.openai import OpenAIConfig
from litellm.types.llms.openai import AllMessageValues

from ..authenticator import Authenticator
from ..common_utils import (
    DEFAULT_GITHUB_COPILOT_API_BASE,
    GetAPIKeyError,
    get_copilot_default_headers,
)


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
            api_base
            or self.authenticator.get_api_base()
            or os.getenv("GITHUB_COPILOT_API_BASE")
            or DEFAULT_GITHUB_COPILOT_API_BASE
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

        # Check if system-to-assistant conversion is disabled
        if litellm.disable_copilot_system_to_assistant:
            # GitHub Copilot API now supports system prompts for all models (Claude, GPT, etc.)
            # No conversion needed - just return messages as-is
            return messages

        # Default behavior: convert system messages to assistant for compatibility
        transformed_messages = []
        for message in messages:
            if message.get("role") == "system":
                # Convert system message to assistant message
                transformed_message = message.copy()
                transformed_message["role"] = "assistant"
                transformed_messages.append(transformed_message)
            else:
                transformed_messages.append(message)

        return transformed_messages

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

        # Add Copilot-specific headers (editor-version, user-agent, etc.)
        try:
            copilot_api_key = self.authenticator.get_api_key()
            copilot_headers = get_copilot_default_headers(copilot_api_key)
            validated_headers = {**copilot_headers, **validated_headers}
        except GetAPIKeyError:
            pass  # Will be handled later in the request flow

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

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI params to GitHub Copilot params.

        GitHub Copilot uses an OpenAI-compatible API and does not understand the
        Anthropic-native ``thinking`` parameter. When Claude Code calls the proxy's
        ``/v1/messages`` endpoint with ``thinking`` and the request is routed to a
        ``github_copilot/claude-*`` model, ``thinking`` must be converted to
        ``reasoning_effort`` before being forwarded to the Copilot API.

        ``reasoning_effort`` is written directly to ``optional_params`` rather than
        deferred to the parent: ``OpenAIConfig.map_openai_params`` dispatches to the
        global ``openAIGPTConfig`` whose supported-params list does not include
        ``reasoning_effort`` for Claude models, so a deferred write would be dropped.
        """
        if "claude" in model.lower():
            thinking = non_default_params.pop("thinking", None)
        else:
            thinking = None
        existing_reasoning_effort = non_default_params.get(
            "reasoning_effort"
        ) or optional_params.get("reasoning_effort")
        if (
            thinking is not None
            and isinstance(thinking, dict)
            and thinking.get("type") == "enabled"
            and existing_reasoning_effort is None
        ):
            budget_tokens = thinking.get("budget_tokens") or 0
            if budget_tokens >= 10000:
                reasoning_effort = "high"
            elif budget_tokens >= 5000:
                reasoning_effort = "medium"
            elif budget_tokens >= 2000:
                reasoning_effort = "low"
            else:
                reasoning_effort = "minimal"
            optional_params["reasoning_effort"] = reasoning_effort
        elif (
            "claude" in model.lower()
            and non_default_params.get("reasoning_effort") is not None
        ):
            optional_params["reasoning_effort"] = non_default_params.pop(
                "reasoning_effort"
            )

        return super().map_openai_params(
            non_default_params, optional_params, model, drop_params
        )

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
