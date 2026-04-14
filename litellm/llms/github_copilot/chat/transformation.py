from typing import List, Optional, Tuple

from litellm.exceptions import AuthenticationError
from litellm.llms.openai.openai import OpenAIConfig
from litellm.types.llms.openai import AllMessageValues

from ..authenticator import Authenticator
from ..common_utils import (
    GITHUB_COPILOT_API_BASE,
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
        dynamic_api_base = self.authenticator.get_api_base() or GITHUB_COPILOT_API_BASE
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

        For Claude models, always includes ``thinking`` and ``reasoning_effort``
        since all Claude models available on Copilot support extended thinking.
        Even if a model doesn't use them, passing these params is harmless —
        Copilot will simply ignore unsupported ones.

        For other models, returns standard OpenAI parameters (which may include
        ``reasoning_effort`` for O-series models via the parent class).
        """
        base_params = super().get_supported_openai_params(model)

        if "claude" in model.lower():
            if "thinking" not in base_params:
                base_params.append("thinking")
            if "reasoning_effort" not in base_params:
                base_params.append("reasoning_effort")

        return base_params

    @staticmethod
    def _is_thinking_enabled(thinking: dict) -> bool:
        """Check if an Anthropic ``thinking`` param has thinking enabled."""
        return (
            isinstance(thinking, dict)
            and thinking.get("type") == "enabled"
        )

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI parameters for GitHub Copilot.

        Two fixes over the inherited ``OpenAIConfig.map_openai_params``:

        1. For Claude models, call ``self._map_openai_params`` directly so that
           ``self.get_supported_openai_params`` (which adds ``reasoning_effort``
           for Claude) is used instead of the hardcoded ``openAIGPTConfig``
           singleton which does not know about Claude-specific params.

        2. When the request arrives from Claude Code's Anthropic adapter
           (``POST /v1/messages``), it may contain a ``thinking`` parameter
           instead of ``reasoning_effort``.  Copilot's API is OpenAI-compatible
           and does not understand ``thinking``, so when thinking is enabled and
           no explicit ``reasoning_effort`` is set, default to ``"high"``.
           Users can override this by setting ``reasoning_effort`` in
           ``litellm_params`` in the proxy config YAML.
        """
        if "claude" in model.lower():
            # Convert Anthropic-native ``thinking`` -> ``reasoning_effort``.
            # If the user already set reasoning_effort (e.g. via litellm config
            # YAML), respect that and don't overwrite.
            thinking = non_default_params.get("thinking")
            if thinking and "reasoning_effort" not in non_default_params:
                if self._is_thinking_enabled(thinking):
                    non_default_params["reasoning_effort"] = "medium"
            # Remove ``thinking`` -- Copilot's OpenAI API doesn't understand it
            non_default_params.pop("thinking", None)

            mapped = self._map_openai_params(
                non_default_params=non_default_params,
                optional_params=optional_params,
                model=model,
            )
            # Belt-and-suspenders: strip ``thinking`` from mapped output too,
            # in case it was already in optional_params before we got here.
            mapped.pop("thinking", None)
            return mapped
        return super().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params,
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
