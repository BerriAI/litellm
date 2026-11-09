from collections.abc import Mapping
from dataclasses import replace
from typing import Final

from pydantic import BaseModel, ConfigDict, ValidationError

from litellm.llms.base_llm.decisions.transformation import usage_reported_cost
from litellm.llms.openai.decisions.transformation import OpenAIDecisionsConfig
from litellm.types.decisions import DecisionsIRRequest, DecisionsIRResponse


class _GatewayCharge(BaseModel):
    model_config = ConfigDict(frozen=True)

    cost: float | None = None


class _GatewayProviderMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    gateway: _GatewayCharge = _GatewayCharge()


class _GatewayDecisionResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider_metadata: _GatewayProviderMetadata = _GatewayProviderMetadata()


def _gateway_charge(payload: object) -> float | None:
    try:
        return _GatewayDecisionResponse.model_validate(payload).provider_metadata.gateway.cost
    except ValidationError:
        return None


class VercelAIGatewayDecisionsConfig(OpenAIDecisionsConfig):
    """get_llm_provider already resolves the gateway's base and key, so never fall back to OpenAI credentials."""

    def resolve_api_key(self, api_key: str | None) -> str | None:
        return api_key

    def resolve_credential(
        self, api_key: str | None, api_base: str, litellm_params: Mapping[str, object]
    ) -> str | None:
        return api_key

    def parse_response(self, payload: object, request: DecisionsIRRequest) -> DecisionsIRResponse:
        response: Final = super().parse_response(payload, request)
        charge: Final = _gateway_charge(payload)
        if charge is None or response.usage is None:
            return response
        return replace(response, usage=replace(response.usage, extra={**response.usage.extra, "cost": charge}))

    def provider_reported_cost(self, response: DecisionsIRResponse) -> float | None:
        return usage_reported_cost(response)
