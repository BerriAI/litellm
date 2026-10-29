from collections.abc import Mapping
from typing import Final

import litellm
from litellm.exceptions import UnsupportedParamsError
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolParam

from ...openai.chat.gpt_transformation import OpenAIGPTConfig

ZAI_API_BASE: Final = "https://api.z.ai/api/paas/v4"


class ZAIChatConfig(OpenAIGPTConfig):
    @property
    def custom_llm_provider(self) -> str | None:
        return "zai"

    def _get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        api_base = api_base or get_secret_str("ZAI_API_BASE") or ZAI_API_BASE
        dynamic_api_key: Final = api_key or get_secret_str("ZAI_API_KEY")
        return api_base, dynamic_api_key

    def get_openai_compatible_provider_info(
        self,
        api_base: str | None,
        api_key: str | None,
    ) -> tuple[str | None, str | None]:
        return self._get_openai_compatible_provider_info(api_base, api_key)

    def remove_cache_control_flag_from_messages_and_tools(
        self,
        model: str,
        messages: list[AllMessageValues],
        tools: list[ChatCompletionToolParam] | None = None,
    ) -> tuple[list[AllMessageValues], list[ChatCompletionToolParam] | None]:
        """
        Override to preserve cache_control for GLM/ZAI.
        GLM supports cache_control - don't strip it.
        """
        # GLM/ZAI supports cache_control, so return messages and tools unchanged
        return messages, tools

    def get_supported_openai_params(self, model: str) -> list[str]:
        base_params: Final = [
            "max_tokens",
            "max_completion_tokens",
            "response_format",
            "stream",
            "stream_options",
            "temperature",
            "top_p",
            "stop",
            "tools",
            "tool_choice",
        ]

        try:
            if litellm.supports_reasoning(model=model, custom_llm_provider=self.custom_llm_provider):
                base_params.append("thinking")
        except Exception:
            pass

        return base_params

    def map_openai_params(
        self,
        non_default_params: Mapping[str, object],
        optional_params: Mapping[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: request pipeline adds parameters to the returned body
        supported_params: Final = self.get_supported_openai_params(model)
        response_format: Final = non_default_params.get("response_format")
        unsupported_response_format: Final = response_format is not None and not _is_json_mode(response_format)
        if unsupported_response_format and not (drop_params or litellm.drop_params):
            raise UnsupportedParamsError(
                message=f"zai supports only response_format {{'type': 'json_object'}} for model={model}. To drop it, set `litellm.drop_params=True`",
                llm_provider="zai",
                model=model,
            )
        return {
            **optional_params,
            **{
                ("max_tokens" if param == "max_completion_tokens" else param): value
                for param, value in non_default_params.items()
                if param in supported_params and not (param == "response_format" and unsupported_response_format)
            },
        }


def _is_json_mode(response_format: object) -> bool:
    match response_format:
        case {"type": "json_object" | "text"}:
            return True
        case _:
            return False
