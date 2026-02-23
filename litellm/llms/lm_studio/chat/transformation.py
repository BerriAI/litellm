"""
Translate from OpenAI's `/v1/chat/completions` to LM Studio's `/chat/completions`
"""

from typing import Optional, Tuple

from litellm.secret_managers.main import get_secret_str

from ...openai.chat.gpt_transformation import OpenAIGPTConfig


class LMStudioChatConfig(OpenAIGPTConfig):
    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = api_base or get_secret_str("LM_STUDIO_API_BASE")  # type: ignore
        dynamic_api_key = (
            api_key or get_secret_str("LM_STUDIO_API_KEY") or "fake-api-key"
        )  # LM Studio does not require an api key, but OpenAI client requires non-None value
        return api_base, dynamic_api_key
    
    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        for param, value in list(non_default_params.items()):
            if param == "response_format" and isinstance(value, dict):
                if value.get("type") == "json_schema":
                    if "json_schema" not in value and "schema" in value:
                        optional_params["response_format"] = {
                            "type": "json_schema",
                            "json_schema": {"schema": value.get("schema")},
                        }
                    else:
                        optional_params["response_format"] = value
                    non_default_params.pop(param, None)
                elif value.get("type") == "json_object":
                    optional_params["response_format"] = value
                    non_default_params.pop(param, None)

        return super().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params,
        )