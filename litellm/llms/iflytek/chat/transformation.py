"""
iFlytek Spark Chat Completions API - Transformation

iFlytek Spark exposes an OpenAI-compatible `/v1/chat/completions` endpoint
(https://spark-api-open.xf-yun.com/v1), so no request/response translation is
required beyond the standard OpenAI-compatible handling.
"""

from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig


class IFlytekConfig(OpenAIGPTConfig):
    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        map max_completion_tokens param to max_tokens
        """
        supported_openai_params = self.get_supported_openai_params(model=model)
        for param, value in non_default_params.items():
            if param == "max_completion_tokens":
                optional_params["max_tokens"] = value
            elif param in supported_openai_params:
                optional_params[param] = value
        return optional_params
