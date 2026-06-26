"""
DeepSeek Anthropic-compatible messages transformation config.
"""

from typing import Any, Dict, List, Optional, Tuple

import litellm
from litellm._logging import verbose_logger
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams


class DeepSeekAnthropicMessagesConfig(AnthropicMessagesConfig):
    """
    DeepSeek exposes an Anthropic-compatible Messages API at
    https://api.deepseek.com/anthropic.

    It accepts the native Anthropic Messages conversation shape, including
    thinking blocks in assistant history, but rejects Anthropic's explicit
    custom-tool discriminator (`{"type": "custom"}`).
    """

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "deepseek"

    def should_strip_billing_metadata(self) -> bool:
        return True

    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        return api_key or get_secret_str("DEEPSEEK_API_KEY") or litellm.api_key

    @staticmethod
    def get_api_base(api_base: Optional[str] = None) -> str:
        return (
            api_base
            or get_secret_str("DEEPSEEK_ANTHROPIC_API_BASE")
            or get_secret_str("DEEPSEEK_API_BASE")
            or "https://api.deepseek.com/anthropic"
        )

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
        dynamic_api_key = self.get_api_key(api_key=api_key)

        if (
            "x-api-key" not in headers
            and "authorization" not in headers
            and dynamic_api_key is not None
        ):
            headers["x-api-key"] = dynamic_api_key

        if "anthropic-version" not in headers:
            headers["anthropic-version"] = "2023-06-01"
        if "content-type" not in headers:
            headers["content-type"] = "application/json"

        headers = self._update_headers_with_anthropic_beta(
            headers=headers,
            optional_params=optional_params,
            custom_llm_provider=self.custom_llm_provider or "deepseek",
        )

        return headers, api_base

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        base_url = self.get_api_base(api_base=api_base).rstrip("/")

        if base_url.endswith("/v1/messages") and "/anthropic/" in base_url:
            return base_url
        if base_url.endswith("/v1/messages"):
            base_url = base_url[: -len("/v1/messages")]
        if base_url.endswith("/v1"):
            base_url = base_url[: -len("/v1")]
        if base_url.endswith("/beta"):
            base_url = base_url[: -len("/beta")]

        if not base_url.endswith("/anthropic") and "/anthropic/" not in base_url:
            base_url = f"{base_url}/anthropic"

        return f"{base_url}/v1/messages"

    @staticmethod
    def _sanitize_tools_for_deepseek(tools: Any) -> Any:
        if not isinstance(tools, list):
            return tools

        sanitized_tools = []
        for tool in tools:
            if isinstance(tool, dict) and tool.get("type") == "custom":
                sanitized_tool = dict(tool)
                sanitized_tool.pop("type", None)
                sanitized_tools.append(sanitized_tool)
            else:
                sanitized_tools.append(tool)
        return sanitized_tools

    def _fill_reasoning_content(self, messages: List[Dict]) -> List[Dict]:
        """
        DeepSeek thinking mode requires `reasoning_content` to be passed back on
        every assistant message in multi-turn conversations. If it is missing,
        the anthropic-compatible API returns:
          "The reasoning_content in the thinking mode must be passed back to the API."

        This mirrors `DeepSeekChatConfig._fill_reasoning_content`
        (litellm/llms/deepseek/chat/transformation.py) for the anthropic-compat
        endpoint — both endpoints enforce the same requirement, but only the
        OpenAI-compat path filled the field.

        For each assistant message that is missing `reasoning_content`:
          1. Promote it from `provider_specific_fields["reasoning_content"]` if
             present (LiteLLM stores provider-specific response fields there).
          2. Otherwise inject a single space — the minimum value the API accepts.
        """
        result: List[Dict] = []
        for msg in messages:
            if (
                isinstance(msg, dict)
                and msg.get("role") == "assistant"
                and not msg.get("reasoning_content")
            ):
                patched = dict(msg)
                provider_fields = patched.get("provider_specific_fields") or {}
                stored = provider_fields.get("reasoning_content")
                if stored:
                    patched["reasoning_content"] = stored
                    cleaned = dict(provider_fields)
                    cleaned.pop("reasoning_content", None)
                    patched["provider_specific_fields"] = cleaned
                else:
                    verbose_logger.warning(
                        "DeepSeek thinking mode: assistant message is missing "
                        "`reasoning_content` and none was saved in "
                        "`provider_specific_fields`. A single-space placeholder "
                        "is being injected to satisfy API validation, but the "
                        "model will receive a blank reasoning chain for this turn, "
                        "which may silently degrade multi-turn response quality. "
                        "Preserve `reasoning_content` from the original assistant "
                        "response when building multi-turn conversation history."
                    )
                    patched["reasoning_content"] = " "
                result.append(patched)
            else:
                result.append(msg)
        return result

    def _thinking_mode_active(self, model: str, optional_params: dict) -> bool:
        """
        Returns True only when the request explicitly carries
        `thinking: {"type": "enabled"}` — the same opt-in signal the
        upstream DeepSeek anthropic-compat endpoint enforces.

        Unlike the chat-side `DeepSeekChatConfig._thinking_mode_active`,
        we do NOT gate on `supports_reasoning(model, custom_llm_provider)`.
        The anthropic-compat endpoint accepts the native anthropic message
        shape (where `thinking` is always opt-in), so any request that
        actually turns thinking on is one we must backfill for.
        """
        return (optional_params.get("thinking") or {}).get("type") == "enabled"

    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: List[Dict],
        anthropic_messages_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Dict:
        anthropic_messages_request = super().transform_anthropic_messages_request(
            model=model,
            messages=messages,
            anthropic_messages_optional_request_params=anthropic_messages_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )
        if "tools" in anthropic_messages_request:
            anthropic_messages_request["tools"] = self._sanitize_tools_for_deepseek(
                anthropic_messages_request["tools"]
            )
        if (
            self._thinking_mode_active(
                model=model, optional_params=anthropic_messages_optional_request_params
            )
            and "messages" in anthropic_messages_request
        ):
            anthropic_messages_request["messages"] = self._fill_reasoning_content(
                anthropic_messages_request["messages"]
            )
        return anthropic_messages_request
