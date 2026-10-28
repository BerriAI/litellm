from typing import Final

from litellm.constants import PLACEHOLDER_API_KEY
from litellm.llms.base_llm.decisions.transformation import BaseDecisionsConfig


class OllamaDecisionsConfig(BaseDecisionsConfig):
    api_key_env = ("OLLAMA_API_KEY",)
    api_base_env = ("OLLAMA_API_BASE",)
    api_key_required = False

    def get_default_api_base(self) -> str | None:
        return "http://localhost:11434"

    def resolve_api_key(self, api_key: str | None) -> str | None:
        resolved: Final = super().resolve_api_key(api_key)
        return None if resolved == PLACEHOLDER_API_KEY else resolved
