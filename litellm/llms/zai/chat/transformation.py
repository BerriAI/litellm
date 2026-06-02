from typing import Any, Coroutine, List, Optional, Tuple, Union

from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolParam

from ...openai.chat.gpt_transformation import OpenAIGPTConfig

ZAI_API_BASE = "https://api.z.ai/api/paas/v4"


def _flatten_content_parts(content: Any) -> Any:
    """Flatten OpenAI multi-part content to a plain string.

    The OpenAI spec allows tool/assistant message content as either a plain
    string or a list of content parts (e.g. [{"type": "text", "text": "..."}]).
    GLM's chat template checks ``m.content is string`` and silently drops
    list-format content (same root cause as vllm-project/vllm#39614).
    This helper normalises both forms to a plain string.
    """
    if isinstance(content, str) or content is None:
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if text:
                    parts.append(text)
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts) if parts else ""
    return content


class ZAIChatConfig(OpenAIGPTConfig):
    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "zai"

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = api_base or get_secret_str("ZAI_API_BASE") or ZAI_API_BASE
        dynamic_api_key = api_key or get_secret_str("ZAI_API_KEY")
        return api_base, dynamic_api_key

    def remove_cache_control_flag_from_messages_and_tools(
        self,
        model: str,
        messages: List[AllMessageValues],
        tools: Optional[List[ChatCompletionToolParam]] = None,
    ) -> Tuple[List[AllMessageValues], Optional[List[ChatCompletionToolParam]]]:
        """
        Override to preserve cache_control for GLM/ZAI.
        GLM supports cache_control - don't strip it.
        """
        # GLM/ZAI supports cache_control, so return messages and tools unchanged
        return messages, tools

    def get_supported_openai_params(self, model: str) -> list:
        base_params = [
            "max_tokens",
            "stream",
            "stream_options",
            "temperature",
            "top_p",
            "stop",
            "tools",
            "tool_choice",
        ]

        import litellm

        try:
            if litellm.supports_reasoning(
                model=model, custom_llm_provider=self.custom_llm_provider
            ):
                base_params.append("thinking")
        except Exception:
            pass

        return base_params

    def _transform_messages(
        self, messages: List[AllMessageValues], model: str, is_async: bool = False
    ) -> Union[List[AllMessageValues], Coroutine[Any, Any, List[AllMessageValues]]]:
        """Flatten list-format content in tool and assistant messages before sending to ZAI.

        GLM's chat template checks ``m.content is string`` and silently drops list-format
        content (e.g. [{"type": "text", "text": "..."}]).  This ensures tool results and
        assistant messages always reach the model as plain strings.

        Issue: https://github.com/BerriAI/litellm/issues/25868
        """
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role in ("tool", "assistant") and isinstance(content, list):
                message["content"] = _flatten_content_parts(content)  # type: ignore
        return super()._transform_messages(messages=messages, model=model, is_async=is_async)  # type: ignore
