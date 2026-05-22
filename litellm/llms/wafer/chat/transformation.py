from typing import Optional, Tuple

from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.secret_managers.main import get_secret_str


class WaferConfig(OpenAIGPTConfig):
    """
    Reference: https://docs.wafer.ai

    Wafer is an OpenAI-compatible inference gateway. All Wafer models are
    served via `https://api.wafer.ai/v1` and accept the standard OpenAI
    chat-completions request/response shape, including streaming SSE and
    tool/function calling.
    """

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_supported_openai_params(self, model: str) -> list:
        return [
            "stream",
            "stream_options",
            "frequency_penalty",
            "function_call",
            "functions",
            "logit_bias",
            "logprobs",
            "top_logprobs",
            "max_tokens",
            "max_completion_tokens",
            "n",
            "presence_penalty",
            "stop",
            "temperature",
            "top_p",
            "response_format",
            "seed",
            "tool_choice",
            "tools",
            "parallel_tool_calls",
            "user",
            "extra_headers",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_openai_params = self.get_supported_openai_params(model=model)
        for param, value in non_default_params.items():
            if param == "max_completion_tokens":
                optional_params["max_tokens"] = value
            elif param in supported_openai_params:
                if value is not None:
                    optional_params[param] = value
        return optional_params

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = (
            api_base or get_secret_str("WAFER_API_BASE") or "https://api.wafer.ai/v1"
        )
        dynamic_api_key = api_key or get_secret_str("WAFER_API_KEY")
        return api_base, dynamic_api_key

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list,
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        if not api_key:
            raise ValueError(
                "Missing Wafer API Key - set WAFER_API_KEY in your environment "
                "or pass api_key= to completion()."
            )

        headers["Authorization"] = f"Bearer {api_key}"
        headers["Content-Type"] = "application/json"

        return headers
