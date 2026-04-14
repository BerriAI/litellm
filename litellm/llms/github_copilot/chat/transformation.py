import re
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

    @staticmethod
    def _is_claude_reasoning_model(model: str) -> bool:
        """
        Check if a Claude model supports extended thinking / reasoning_effort.

        GitHub Copilot uses its own model naming (dots instead of hyphens, e.g.
        ``claude-sonnet-4.5``, ``claude-opus-4.6-fast``), which may not be in
        the model registry.  This helper uses model-name pattern matching as a
        fallback so that newly added Copilot models work without waiting for a
        registry update.

        Models that support reasoning:
        - Claude 4+ family (sonnet-4, opus-4.5, opus-4.6, haiku-4.5, etc.)
        - Claude 3-7 Sonnet
        Models that do NOT support reasoning:
        - Claude 3.5 and earlier
        """
        model_lower = model.lower()

        # Exclude Claude 3.5 and earlier (no extended thinking)
        if "claude-3.5" in model_lower or "claude-3-5" in model_lower:
            return False
        if "claude-3.0" in model_lower or "claude-3-0" in model_lower:
            return False

        # Claude 3-7 Sonnet supports reasoning
        if "claude-3-7" in model_lower or "claude-3.7" in model_lower:
            return True

        # Claude 4+ family: any model containing "claude-<variant>-4"
        # Matches: claude-sonnet-4, claude-opus-4.5, claude-opus-4.6-fast,
        #          claude-haiku-4.5, claude-opus-41, etc.
        if re.search(r"claude-\w+-4", model_lower):
            return True

        return False

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get supported OpenAI parameters for GitHub Copilot.

        For Claude models that support extended thinking (Claude 4 family and
        Claude 3-7), includes ``thinking`` and ``reasoning_effort`` parameters.

        Uses model-name pattern matching as the primary check (Copilot model
        names may not be in the model registry), with ``supports_reasoning()``
        as a fallback for models registered with ``supports_reasoning=True``.

        For other models, returns standard OpenAI parameters (which may include
        ``reasoning_effort`` for O-series models via the parent class).
        """
        from litellm.utils import supports_reasoning

        # Get base OpenAI parameters
        base_params = super().get_supported_openai_params(model)

        # Add Claude-specific parameters for models that support extended thinking
        if "claude" in model.lower() and (
            self._is_claude_reasoning_model(model)
            or supports_reasoning(
                model=model.lower(),
                custom_llm_provider="github_copilot",
            )
        ):
            if "thinking" not in base_params:
                base_params.append("thinking")
            if "reasoning_effort" not in base_params:
                base_params.append("reasoning_effort")

        return base_params

    @staticmethod
    def _translate_thinking_to_reasoning_effort(thinking: dict) -> Optional[str]:
        """
        Convert Anthropic ``thinking`` param to OpenAI ``reasoning_effort``.

        GitHub Copilot exposes an OpenAI-compatible API, so the Anthropic-native
        ``thinking`` parameter must be translated.  The budget-token thresholds
        mirror ``LiteLLMAnthropicMessagesAdapter.translate_anthropic_thinking_to_reasoning_effort``.
        """
        if not isinstance(thinking, dict):
            return None
        if thinking.get("type") == "disabled":
            return None
        if thinking.get("type") == "enabled":
            budget = thinking.get("budget_tokens", 0)
            if budget >= 10000:
                return "high"
            if budget >= 5000:
                return "medium"
            if budget >= 2000:
                return "low"
            return "minimal"
        return None

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
           and does not understand ``thinking``, so convert it to
           ``reasoning_effort`` and strip ``thinking`` from the output.
        """
        if "claude" in model.lower():
            # Convert Anthropic-native ``thinking`` -> ``reasoning_effort``
            # before mapping params, so _map_openai_params sees it as a
            # standard OpenAI param.
            thinking = non_default_params.get("thinking")
            if thinking and "reasoning_effort" not in non_default_params:
                effort = self._translate_thinking_to_reasoning_effort(thinking)
                if effort:
                    non_default_params["reasoning_effort"] = effort
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
