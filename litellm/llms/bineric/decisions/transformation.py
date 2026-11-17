from litellm.llms.base_llm.decisions.transformation import BaseDecisionsConfig, usage_reported_cost
from litellm.types.decisions import DecisionsIRResponse


class BinericDecisionsConfig(BaseDecisionsConfig):
    api_key_env = ("BINERIC_API_KEY",)
    api_base_env = ("BINERIC_API_BASE",)

    def get_default_api_base(self) -> str | None:
        return "https://api.bineric.com/v1"

    def provider_reported_cost(self, response: DecisionsIRResponse) -> float | None:
        return usage_reported_cost(response)
