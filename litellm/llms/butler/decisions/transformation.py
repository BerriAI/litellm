from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel, ConfigDict, TypeAdapter

from litellm.litellm_core_utils.core_helpers import finite_reported_cost
from litellm.llms.base_llm.decisions.transformation import BaseDecisionsConfig
from litellm.llms.openai.decisions.transformation import ir_to_openai_request, openai_response_to_ir
from litellm.types.decisions import (
    DecisionsIRRequest,
    DecisionsIRResponse,
    OpenAIDecisionResponse,
    UnsupportedDecisionsRequest,
)


class _ButlerUsage(BaseModel):
    cost: float | None = None

    model_config = ConfigDict(frozen=True)


class _ButlerResponse(BaseModel):
    usage: _ButlerUsage

    model_config = ConfigDict(frozen=True)


_RESPONSE_ADAPTER: Final[TypeAdapter[OpenAIDecisionResponse]] = TypeAdapter(OpenAIDecisionResponse)
_COST_ADAPTER: Final[TypeAdapter[_ButlerResponse]] = TypeAdapter(_ButlerResponse)


class ButlerDecisionsConfig(BaseDecisionsConfig):
    """The OpenAI Decisions body; usage.cost is what Butler charged."""

    path = "/v1/decisions"
    api_key_env = ("BUTLER_API_KEY",)
    api_base_env = ("BUTLER_API_BASE",)
    supports_safety_identifier = True
    supports_images = True
    inlines_remote_images = True
    max_images = 4
    # Butler takes a decisions body of up to 64 MiB of JSON; base64 makes 32 MiB of images about 43 MiB. Remote images
    # are fetched against this budget, so a caller's URLs cannot make the proxy buffer more.
    max_image_bytes = 16 * 1024 * 1024
    max_total_image_bytes = 32 * 1024 * 1024

    def get_default_api_base(self) -> str | None:
        return "https://api.butlercompute.com"

    def transform_decisions_request(
        self, model: str, request: DecisionsIRRequest, custom_llm_provider: str
    ) -> Mapping[str, object] | UnsupportedDecisionsRequest:
        return ir_to_openai_request(model, request)

    def parse_response(self, payload: object, request: DecisionsIRRequest) -> DecisionsIRResponse:
        response: Final = openai_response_to_ir(_RESPONSE_ADAPTER.validate_python(payload), request)
        extra: Final = MappingProxyType({"cost": _COST_ADAPTER.validate_python(payload).usage.cost})
        return replace(response, usage=response.usage and replace(response.usage, extra=extra))

    def provider_reported_cost(self, response: DecisionsIRResponse) -> float | None:
        return finite_reported_cost(response.usage and response.usage.extra.get("cost"))
