from collections.abc import Mapping
from dataclasses import replace
from itertools import chain
from typing import Final, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from litellm.llms.base_llm.decisions.transformation import usage_reported_cost
from litellm.llms.openai.decisions.transformation import OpenAIDecisionsConfig
from litellm.types.decisions import (
    DecisionsIRRequest,
    DecisionsIRResponse,
    DecisionsIRUsage,
    OpenAIDecisionInputTokensDetails,
)

_Model: Final = TypeVar("_Model", bound=BaseModel)


class _ProviderAttempt(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str | None = None
    credential_type: str | None = Field(default=None, alias="credentialType")
    success: bool = False


class _ModelAttempt(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider_attempts: tuple[_ProviderAttempt, ...] = Field(default=(), alias="providerAttempts")


class _Routing(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_attempts: tuple[_ModelAttempt, ...] = Field(default=(), alias="modelAttempts")

    def served_by(self) -> _ProviderAttempt | None:
        attempts: Final = chain.from_iterable(model.provider_attempts for model in self.model_attempts)
        return next((attempt for attempt in attempts if attempt.success), None)


class _GatewayRouting(BaseModel):
    model_config = ConfigDict(frozen=True)

    routing: _Routing = _Routing()


class _GatewayCharge(BaseModel):
    model_config = ConfigDict(frozen=True)

    cost: float
    market_cost: float | None = Field(default=None, alias="marketCost")

    def billed_cost(self, served_by: _ProviderAttempt | None) -> float:
        if self.market_cost is None or served_by is None or served_by.credential_type != "byok":
            return self.cost
        return self.cost + self.market_cost


class _UpstreamUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens: int
    input_tokens_details: OpenAIDecisionInputTokensDetails


class _UpstreamReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    usage: _UpstreamUsage


class _GatewayDecisionResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider_metadata: Mapping[str, object]


def _validated(model: type[_Model], value: object) -> _Model | None:
    try:
        return model.model_validate(value)
    except ValidationError:
        return None


def _served_by(gateway: object) -> _ProviderAttempt | None:
    gateway_routing: Final = _validated(_GatewayRouting, gateway)
    return None if gateway_routing is None else gateway_routing.routing.served_by()


def _upstream_usage(metadata: Mapping[str, object], served_by: _ProviderAttempt | None) -> _UpstreamUsage | None:
    if served_by is None or served_by.provider is None:
        return None
    report: Final = _validated(_UpstreamReport, metadata.get(served_by.provider))
    return None if report is None else report.usage


def _gateway_usage(usage: DecisionsIRUsage, metadata: Mapping[str, object]) -> DecisionsIRUsage:
    gateway: Final = metadata.get("gateway")
    served_by: Final = _served_by(gateway)
    charge: Final = _validated(_GatewayCharge, gateway)
    priced: Final = (
        usage if charge is None else replace(usage, extra={**usage.extra, "cost": charge.billed_cost(served_by)})
    )
    upstream: Final = _upstream_usage(metadata, served_by)
    if upstream is None or upstream.input_tokens != usage.input_tokens:
        return priced
    return replace(
        priced,
        cached_tokens=upstream.input_tokens_details.cached_tokens,
        cache_write_tokens=upstream.input_tokens_details.cache_write_tokens,
    )


class VercelAIGatewayDecisionsConfig(OpenAIDecisionsConfig):
    """get_llm_provider already resolves the gateway's base and key, so never fall back to OpenAI credentials."""

    inlines_remote_images = True

    def resolve_api_key(self, api_key: str | None) -> str | None:
        return api_key

    def resolve_credential(
        self, api_key: str | None, api_base: str, litellm_params: Mapping[str, object]
    ) -> str | None:
        return api_key

    def parse_response(self, payload: object, request: DecisionsIRRequest) -> DecisionsIRResponse:
        response: Final = super().parse_response(payload, request)
        metadata: Final = _validated(_GatewayDecisionResponse, payload)
        if metadata is None:
            return response
        return replace(response, usage=_gateway_usage(response.usage, metadata.provider_metadata))

    def provider_reported_cost(self, response: DecisionsIRResponse) -> float | None:
        return usage_reported_cost(response)
