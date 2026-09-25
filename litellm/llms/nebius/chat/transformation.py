"""
Nebius AI Studio Chat Completions API - Transformation

This is OpenAI compatible - no translation needed / occurs
"""

from types import MappingProxyType
from typing import Final

from litellm._logging import verbose_logger
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.secret_managers.main import get_secret_str


class NebiusConfig(OpenAIGPTConfig):
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
        supported_openai_params: Final = self.get_supported_openai_params(model=model)
        for param, value in non_default_params.items():
            if param == "max_completion_tokens":
                optional_params["max_tokens"] = value
            elif param in supported_openai_params:
                optional_params[param] = value
        return optional_params

    def get_model_cost_key(self, model: str) -> str | None:
        import litellm

        stripped: Final = model.removeprefix("nebius/")
        prefixed: Final = f"nebius/{stripped}"
        catalog: Final = litellm.nebius_models | litellm.nebius_embedding_models
        if prefixed in litellm.model_cost:
            return prefixed
        prefixed_folded: Final = prefixed.casefold()
        prefixed_matches: Final = tuple(key for key in catalog if key.casefold() == prefixed_folded)
        if len(prefixed_matches) == 1:
            return prefixed_matches[0]
        suffix: Final = f"/{stripped}".casefold()
        matches: Final = tuple(key for key in catalog if key.casefold().endswith(suffix))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            verbose_logger.warning(
                "Nebius cost map has %s keys ending in %s; not using a suffix match. keys=%s",
                len(matches),
                suffix,
                matches,
            )
        return None

    def get_models(self, api_key: str | None = None, api_base: str | None = None) -> list[str]:
        import litellm

        resolved_key: Final = self.get_api_key(api_key)
        resolved_base: Final = self.get_api_base(api_base)
        if resolved_key is None or resolved_base is None:
            raise ValueError(
                "NEBIUS_API_KEY is not set. Pass api_key or set NEBIUS_API_KEY to query Nebius /v1/models."
            )
        response: Final = litellm.module_level_client.get(
            url=f"{resolved_base.rstrip('/')}/models",
            headers=MappingProxyType({"Authorization": f"Bearer {resolved_key}"}),
        )
        if response.status_code != 200:
            raise Exception(f"Failed to get models: {response.text}")
        models: Final = response.json()["data"]
        return [model["id"] for model in models]  # mutable-ok: OpenAI-compatible get_models returns list[str]

    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        return api_key or get_secret_str("NEBIUS_API_KEY")

    @staticmethod
    def get_api_base(api_base: str | None = None) -> str | None:
        return api_base or get_secret_str("NEBIUS_API_BASE") or "https://api.studio.nebius.ai/v1"
