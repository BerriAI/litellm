from collections.abc import Mapping
from dataclasses import replace
from itertools import chain
from types import MappingProxyType
from typing import Final, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from litellm.llms.base_llm.decisions.transformation import usage_reported_cost
from litellm.llms.openai.decisions.transformation import OpenAIDecisionsConfig
from litellm.types.decisions import DecisionsIRRequest, DecisionsIRResponse, DecisionsIRUsage, OpenAIDecisionUsage
from litellm.utils import is_vision_explicitly_disabled

_Model: Final = TypeVar("_Model", bound=BaseModel)
_PROVIDER_OPTIONS: Final = "providerOptions"


class _ProviderAttempt(BaseModel):
    model_config = ConfigDict(frozen=True)

    credential_type: Literal["system", "byok"] | None = Field(default=None, alias="credentialType")
    success: bool = False


class _ModelAttempt(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider_attempts: tuple[_ProviderAttempt, ...] = Field(default=(), alias="providerAttempts")


class _Routing(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_attempts: tuple[_ModelAttempt, ...] = Field(default=(), alias="modelAttempts")


class _GatewayCharge(BaseModel):
    model_config = ConfigDict(frozen=True)

    cost: float
    market_cost: float | None = Field(default=None, alias="marketCost")
    routing: _Routing = _Routing()

    def served_with_own_key(self) -> bool:
        attempts: Final = chain.from_iterable(model.provider_attempts for model in self.routing.model_attempts)
        return any(attempt.success and attempt.credential_type == "byok" for attempt in attempts)

    def billed_cost(self) -> float:
        if self.market_cost is None or not self.served_with_own_key():
            return self.cost
        return self.cost + self.market_cost


class _UpstreamReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    usage: OpenAIDecisionUsage


class _GatewayDecisionResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider_metadata: Mapping[str, object]


def _validated(model: type[_Model], value: object) -> _Model | None:
    try:
        return model.model_validate(value)
    except ValidationError:
        return None


def _gateway_usage(usage: DecisionsIRUsage, metadata: Mapping[str, object]) -> DecisionsIRUsage:
    charge: Final = _validated(_GatewayCharge, metadata.get("gateway"))
    reports: Final = (_validated(_UpstreamReport, entry) for entry in metadata.values())
    upstream: Final = next(
        (report.usage for report in reports if report is not None and report.usage.input_tokens == usage.input_tokens),
        None,
    )
    priced: Final = usage if charge is None else replace(usage, extra={**usage.extra, "cost": charge.billed_cost()})
    if upstream is None:
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

    def supports_image_input(self, model: str, custom_llm_provider: str) -> bool:
        return not is_vision_explicitly_disabled(model, custom_llm_provider)

    def provider_request_fields(self, litellm_params: Mapping[str, object]) -> Mapping[str, object]:
        if _PROVIDER_OPTIONS not in litellm_params:
            return MappingProxyType({})
        return MappingProxyType({_PROVIDER_OPTIONS: litellm_params[_PROVIDER_OPTIONS]})

    def parse_response(self, payload: object, request: DecisionsIRRequest) -> DecisionsIRResponse:
        response: Final = super().parse_response(payload, request)
        metadata: Final = _validated(_GatewayDecisionResponse, payload)
        if metadata is None or response.usage is None:
            return response
        return replace(response, usage=_gateway_usage(response.usage, metadata.provider_metadata))

    def provider_reported_cost(self, response: DecisionsIRResponse) -> float | None:
        return usage_reported_cost(response)
