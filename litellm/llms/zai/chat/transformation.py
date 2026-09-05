from typing import Final

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

    def get_supported_openai_params(self, model: str) -> list:
        base_params: Final = [
            "max_tokens",
            "stream",
            "stream_options",
            "temperature",
            "top_p",
            "stop",
            "tools",
            "tool_choice",
            "response_format",
        ]

        import litellm

        try:
            if litellm.supports_reasoning(model=model, custom_llm_provider=self.custom_llm_provider):
                base_params.append("thinking")
        except Exception:
            pass

        return base_params

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Translate response_format into a forced tool call.

        GLM ignores response_format but honours a forced tool call, so a schema only
        survives the round trip as one. See BerriAI/litellm#37720.
        """
        response_format: Final = non_default_params.get("response_format")
        optional_params = super().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params,
        )
        if not self._should_translate_response_format(non_default_params, response_format):
            return optional_params

        # super() copied response_format through as an allowlisted param. GLM ignores
        # it, so trade it for the tool call the model does honour.
        optional_params.pop("response_format", None)
        return self._add_response_format_to_tools(
            optional_params=optional_params,
            value=response_format,
            is_response_format_supported=False,
        )

    @staticmethod
    def _should_translate_response_format(non_default_params: dict, response_format: dict | None) -> bool:
        """
        Only translate when the forced tool call can actually be unwrapped again.

        Otherwise response_format is left to pass through: GLM ignores it, which is
        the pre-existing behaviour and better than an unusable tool call.
        """
        if response_format is None:
            return False
        # Forcing json_tool_call would hijack a caller that is genuinely using tools.
        if non_default_params.get("tools"):
            return False
        # The unwrap back into message.content only runs on non-streaming responses.
        if non_default_params.get("stream"):
            return False
        # Mirror _add_response_format_to_tools' own extraction, key presence included,
        # so this never pops response_format for a schema the helper would ignore.
        if "response_schema" in response_format:
            json_schema = response_format["response_schema"]
        elif "json_schema" in response_format:
            json_schema = response_format["json_schema"].get("schema")
        else:
            json_schema = None
        return bool(json_schema)
