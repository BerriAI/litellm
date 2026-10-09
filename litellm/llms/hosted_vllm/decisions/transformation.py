from types import MappingProxyType
from typing import Final

from litellm.constants import PLACEHOLDER_API_KEY
from litellm.llms.base_llm.decisions.transformation import BaseDecisionsConfig


class HostedVLLMDecisionsConfig(BaseDecisionsConfig):
    api_key_env = ("HOSTED_VLLM_API_KEY",)
    api_base_env = ("HOSTED_VLLM_API_BASE",)
    api_key_required = False
    health_check_questions = MappingProxyType(
        {
            "reachable": MappingProxyType(
                {
                    "type": "choice",
                    "instructions": "Is the service reachable?",
                    "criteria": {"yes": None, "no": None},
                }
            )
        }
    )

    def resolve_api_key(self, api_key: str | None) -> str | None:
        resolved: Final = super().resolve_api_key(api_key)
        return None if resolved == PLACEHOLDER_API_KEY else resolved
