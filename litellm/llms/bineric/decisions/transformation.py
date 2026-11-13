from typing import Final

from litellm.llms.base_llm.decisions.transformation import BaseDecisionsConfig
from litellm.types.decisions import DecisionsIRResponse


class BinericDecisionsConfig(BaseDecisionsConfig):
    api_key_env = ("BINERIC_API_KEY",)
    api_base_env = ("BINERIC_API_BASE",)

    def get_default_api_base(self) -> str | None:
        return "https://api.bineric.com/v1"

    def provider_reported_cost(self, response: DecisionsIRResponse) -> float | None:
        cost: Final = response.usage.extra.get("cost") if response.usage is not None else None
        if not isinstance(cost, (int, float)):
            return None
        return float(cost)
