from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from litellm.llms.base_llm.decisions.transformation import BaseDecisionsConfig
from litellm.llms.cloudflare.decisions.transformation import CloudflareDecisionsConfig
from litellm.llms.perplexity.decisions.transformation import PerplexityDecisionsConfig
from litellm.llms.typesafe.decisions.transformation import TypeSafeDecisionsConfig
from litellm.types.decisions import DecisionsIRMessages, DecisionsIRResponse, DecisionsIRState

_ORG_IMAGE_RULES: Final[Mapping[str, BaseDecisionsConfig]] = MappingProxyType(
    {
        "typesafe": TypeSafeDecisionsConfig(),
        "cloudflare": CloudflareDecisionsConfig(),
        "perplexity": PerplexityDecisionsConfig(),
    }
)


def _upstream_org(model: str) -> str:
    return model.removeprefix("~").partition("/")[0]


class OpenRouterDecisionsConfig(BaseDecisionsConfig):
    path = "/alpha/decisions"
    api_key_env = ("OPENROUTER_API_KEY",)
    api_base_env = ("OPENROUTER_API_BASE",)
    inlines_remote_images = True
    images_in_state = True

    def get_default_api_base(self) -> str | None:
        return "https://openrouter.ai/api"

    def supports_images_in(
        self, model: str, decision_input: DecisionsIRState | DecisionsIRMessages, headers: Mapping[str, str]
    ) -> bool:
        return _upstream_org(model) in _ORG_IMAGE_RULES

    def image_rules_for(self, model: str) -> BaseDecisionsConfig:
        return _ORG_IMAGE_RULES.get(_upstream_org(model), self)

    def provider_reported_cost(self, response: DecisionsIRResponse) -> float | None:
        cost: Final = response.usage.extra.get("cost") if response.usage is not None else None
        if not isinstance(cost, (int, float)):
            return None
        return float(cost)
