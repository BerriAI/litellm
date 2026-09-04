"""
Translate from OpenAI's `/v1/chat/completions` to LM Studio's `/chat/completions`
"""

from typing import Final

from litellm.secret_managers.main import get_secret_str

from ...openai.chat.gpt_transformation import OpenAIGPTConfig


class LMStudioChatConfig(OpenAIGPTConfig):
    def _get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        api_base = api_base or get_secret_str("LM_STUDIO_API_BASE")
        dynamic_api_key: Final = (
            api_key or get_secret_str("LM_STUDIO_API_KEY") or "fake-api-key"
        )  # LM Studio does not require an api key, but OpenAI client requires non-None value
        return api_base, dynamic_api_key

    def get_models(self, api_key: str | None = None, api_base: str | None = None) -> list[str]:
        """
        Calls LM Studio's `/v1/models` endpoint and returns the list of models,
        prefixed with "lm_studio/" (matching the OllamaModelInfo convention)
        so discovered ids are directly callable without the caller having to
        add the provider prefix themselves.

        Reuses the same api_base/api_key resolution as chat completions
        (LM_STUDIO_API_BASE / LM_STUDIO_API_KEY env vars, "fake-api-key"
        fallback) instead of OpenAIGPTConfig's default of
        https://api.openai.com, which would be wrong for a local/self-hosted
        LM Studio server.
        """
        api_base, api_key = self._get_openai_compatible_provider_info(api_base=api_base, api_key=api_key)
        models: Final = super().get_models(api_key=api_key, api_base=api_base)
        return [  # mutable-ok: matches OllamaModelInfo.get_models' list-of-prefixed-ids contract
            m if m.startswith("lm_studio/") else f"lm_studio/{m}" for m in models
        ]

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
