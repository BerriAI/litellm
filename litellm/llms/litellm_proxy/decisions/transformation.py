from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import Final

import httpx
from pydantic import TypeAdapter, ValidationError

from litellm.llms.base_llm.decisions.transformation import BaseDecisionsConfig
from litellm.llms.openai.decisions.transformation import (
    OpenAIDecisionsConfig,
    ir_to_openai_request,
    openai_response_to_ir,
)
from litellm.llms.parallel_ai.decisions.transformation import ParallelAIDecisionsConfig
from litellm.types.decisions import (
    DecisionsAudio,
    DecisionsIRRequest,
    DecisionsIRResponse,
    DecisionsVideo,
    OpenAIDecisionResponse,
    UnsupportedDecisionsRequest,
)

_RESPONSE_ADAPTER: Final[TypeAdapter[OpenAIDecisionResponse]] = TypeAdapter(OpenAIDecisionResponse)
_COST_ADAPTER: Final[TypeAdapter[float]] = TypeAdapter(float)
_COST_HEADER: Final = "x-litellm-response-cost"


def _wire_clip(clip: DecisionsAudio | DecisionsVideo) -> str | Mapping[str, str]:
    return clip if isinstance(clip, str) else {"content_type": clip.content_type, "base64": clip.base64}


def _media_fields(request: DecisionsIRRequest) -> Mapping[str, object]:
    return {
        **({"audio": tuple(_wire_clip(clip) for clip in request.audio)} if request.audio else {}),
        **({"videos": tuple(_wire_clip(clip) for clip in request.videos)} if request.videos else {}),
    }


def _drop_params_field(request: DecisionsIRRequest) -> Mapping[str, bool]:
    return {"drop_params": True} if request.drop_params else {}


def _response_cost(raw_response: httpx.Response) -> float | None:
    if _COST_HEADER not in raw_response.headers:
        return None
    try:
        return _COST_ADAPTER.validate_python(raw_response.headers[_COST_HEADER])
    except ValidationError:
        return None


class _LiteLLMProxyHop(BaseDecisionsConfig):
    confidence_optional = True
    api_base_env = ("LITELLM_PROXY_API_BASE",)
    api_key_env = ("LITELLM_PROXY_API_KEY",)
    api_key_required = False

    def get_default_api_base(self) -> str | None:
        return None

    def missing_api_base_message(self, custom_llm_provider: str) -> str:
        return "api_base is required for the LiteLLM proxy, set api_base or LITELLM_PROXY_API_BASE"

    def resolve_api_base(self, api_base: str | None) -> str | None:
        return BaseDecisionsConfig.resolve_api_base(self, api_base)

    def resolve_api_key(self, api_key: str | None) -> str | None:
        return BaseDecisionsConfig.resolve_api_key(self, api_key)

    def resolve_credential(
        self, api_key: str | None, api_base: str, litellm_params: Mapping[str, object]
    ) -> str | None:
        return api_key

    def supports_box_questions(
        self, model: str, custom_llm_provider: str, headers: Mapping[str, str], deployment_flag: bool | None
    ) -> bool:
        return deployment_flag is not False

    def supports_audio_input(self, model: str, custom_llm_provider: str, deployment_flag: bool | None) -> bool:
        return deployment_flag is not False

    def supports_video_input(self, model: str, custom_llm_provider: str, deployment_flag: bool | None) -> bool:
        return deployment_flag is not False

    def transform_decisions_response(
        self,
        model: str,
        custom_llm_provider: str,
        raw_response: httpx.Response,
        request: DecisionsIRRequest,
    ) -> DecisionsIRResponse:
        response: Final = super().transform_decisions_response(model, custom_llm_provider, raw_response, request)
        return replace(response, reported_cost=_response_cost(raw_response))

    def provider_reported_cost(self, response: DecisionsIRResponse) -> float | None:
        return response.reported_cost


class LiteLLMProxySystemOneConfig(_LiteLLMProxyHop):
    supports_images = True
    supports_safety_identifier = True

    def supports_assets(self, headers: Mapping[str, str]) -> bool:
        return True

    def transform_decisions_request(
        self,
        model: str,
        request: DecisionsIRRequest,
        custom_llm_provider: str,
    ) -> Mapping[str, object] | UnsupportedDecisionsRequest:
        body: Final = super().transform_decisions_request(model, request, custom_llm_provider)
        if isinstance(body, UnsupportedDecisionsRequest):
            return body
        return {
            **body,
            **({"safety_identifier": request.safety_identifier} if request.safety_identifier is not None else {}),
            **_drop_params_field(request),
        }


class LiteLLMProxyDecisionsConfig(_LiteLLMProxyHop, OpenAIDecisionsConfig):
    openai_evidence_fields = ParallelAIDecisionsConfig.openai_evidence_fields

    def systemone_config(self) -> BaseDecisionsConfig:
        return LiteLLMProxySystemOneConfig()

    def transform_decisions_request(
        self,
        model: str,
        request: DecisionsIRRequest,
        custom_llm_provider: str,
    ) -> Mapping[str, object] | UnsupportedDecisionsRequest:
        body: Final = ir_to_openai_request(self.request_model(model), replace(request, audio=(), videos=()))
        if isinstance(body, UnsupportedDecisionsRequest):
            return body
        return {**body, **_media_fields(request), **_drop_params_field(request)}

    def parse_response(self, payload: object, request: DecisionsIRRequest) -> DecisionsIRResponse:
        response: Final = _RESPONSE_ADAPTER.validate_python(payload)
        return replace(openai_response_to_ir(response, request), extra=MappingProxyType(response.model_extra or {}))
