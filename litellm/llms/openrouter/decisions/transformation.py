from typing import Final

from litellm.llms.base_llm.decisions.transformation import BaseDecisionsConfig
from litellm.types.decisions import DecisionsIRResponse


class OpenRouterDecisionsConfig(BaseDecisionsConfig):
    path = "/alpha/decisions"
    api_key_env = ("OPENROUTER_API_KEY",)
    api_base_env = ("OPENROUTER_API_BASE",)

    def get_default_api_base(self) -> str | None:
        return "https://openrouter.ai/api"

    def provider_reported_cost(self, response: DecisionsIRResponse) -> float | None:
        cost: Final = response.usage.extra.get("cost")
        if not isinstance(cost, (int, float)):
            return None
        return float(cost)
