from typing import List, Optional, Tuple

from litellm.litellm_core_utils.prompt_templates.common_utils import (
    strip_name_from_messages,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

from ...openai.chat.gpt_transformation import OpenAIGPTConfig

XAI_API_BASE = "https://api.x.ai/v1"


class XAIChatConfig(OpenAIGPTConfig):
    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = api_base or get_secret_str("XAI_API_BASE") or XAI_API_BASE  # type: ignore
        dynamic_api_key = api_key or get_secret_str("XAI_API_KEY")
        return api_base, dynamic_api_key

    def get_supported_openai_params(self, model: str) -> list:
        return [
            "frequency_penalty",
            "logit_bias",
            "logprobs",
            "max_tokens",
            "n",
            "presence_penalty",
            "response_format",
            "seed",
            "stop",
            "stream",
            "stream_options",
            "temperature",
            "tool_choice",
            "tools",
            "top_logprobs",
            "top_p",
            "user",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool = False,
    ) -> dict:
        supported_openai_params = self.get_supported_openai_params(model=model)
        for param, value in non_default_params.items():
            if param == "max_completion_tokens":
                optional_params["max_tokens"] = value
            elif param in supported_openai_params:
                if value is not None:
                    optional_params[param] = value
        return optional_params

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Handle https://github.com/BerriAI/litellm/issues/9720

        Filter out 'name' from messages
        """
        messages = strip_name_from_messages(messages)
        return super().transform_request(
            model, messages, optional_params, litellm_params, headers
        )
