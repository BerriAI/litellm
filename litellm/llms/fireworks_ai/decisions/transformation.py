from dataclasses import replace
from typing import Final

from litellm.llms.base_llm.decisions.transformation import BaseDecisionsConfig, parse_systemone_response
from litellm.types.decisions import DecisionsIRRequest, DecisionsIRResponse


class FireworksAIDecisionsConfig(BaseDecisionsConfig):
    api_key_env = ("FIREWORKS_API_KEY", "FIREWORKS_AI_API_KEY")
    api_base_env = ("FIREWORKS_API_BASE",)

    def get_default_api_base(self) -> str | None:
        return "https://api.fireworks.ai/inference/v1"

    def parse_response(self, payload: object, request: DecisionsIRRequest) -> DecisionsIRResponse:
        response: Final = parse_systemone_response(self.unwrap_response(payload), request)
        cached: Final = response.usage.extra.get("cached_input_tokens")
        if isinstance(cached, bool) or not isinstance(cached, int):
            return response
        return replace(response, usage=replace(response.usage, cached_tokens=cached))
