from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

import litellm
from litellm import get_llm_provider
from litellm.completion_extras.litellm_responses_transformation.transformation import (
    LiteLLMResponsesTransformationHandler,
)
from litellm.litellm_core_utils.core_helpers import normalize_drop_params
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.rust_bridge import failures
from litellm.rust_bridge.public_call import inference_decline_reason, optional_str
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams, ResponsesAPIResponse
from litellm.utils import ProviderConfigManager

_CONTROLS: Final = frozenset({"reasoning_effort", "drop_params", "allowed_openai_params", "additional_drop_params"})
PARAMETERS: Final = (*ResponsesAPIOptionalRequestParams.__annotations__, *_CONTROLS)


def body_params(request: Mapping[str, object]) -> Mapping[str, object]:
    model, provider, _, _ = get_llm_provider(
        model=str(request["model"]), custom_llm_provider=optional_str(request.get("custom_llm_provider"))
    )
    reasoning: Final = request.get("reasoning")
    reasoning_effort: Final = request.get("reasoning_effort")
    requested: Final = ResponsesAPIRequestUtils.get_requested_response_api_optional_param(
        {
            **{name: value for name, value in request.items() if name != "reasoning_effort"},
            "custom_llm_provider": provider,
            "reasoning": (
                LiteLLMResponsesTransformationHandler().map_reasoning_effort(reasoning_effort)
                if reasoning is None and reasoning_effort is not None
                else reasoning
            ),
        }
    )
    config: Final = ProviderConfigManager.get_provider_responses_api_config(provider=provider, model=model)
    if config is None:
        return dict(requested)
    allowed: Final = request.get("allowed_openai_params")
    allowed_values: Final[Sequence[object] | None] = allowed if isinstance(allowed, list) else None
    return ResponsesAPIRequestUtils.get_optional_params_responses_api(
        model=model,
        responses_api_provider_config=config,
        response_api_optional_params=requested,
        allowed_openai_params=[str(name) for name in allowed_values] if allowed_values is not None else None,
        drop_params=normalize_drop_params(request.get("drop_params")),
    )


def connection_defaults(provider: str) -> tuple[str | None, str | None]:
    provider_key: Final = litellm.openai_key if provider == "openai" else None
    return litellm.api_key or provider_key, litellm.api_base


def response(value: Mapping[str, object]) -> ResponsesAPIResponse:
    return ResponsesAPIResponse.model_validate(value)


def map_failure(error: Exception, request: Mapping[str, object]) -> Exception:
    provider: Final = optional_str(request.get("custom_llm_provider")) or "openai"
    return failures.map_native_failure(
        error,
        str(request["model"]),
        provider,
        request,
        optional_str(request.get("api_base")) or optional_str(request.get("base_url")),
    )


def decline_reason(request: Mapping[str, object]) -> str | None:
    if optional_str(request.get("custom_llm_provider")) is None and "/" not in str(request["model"]):
        try:
            _, provider, _, _ = get_llm_provider(model=str(request["model"]))
        except litellm.exceptions.BadRequestError:
            return "native Responses could not resolve the provider"
        if provider != "openai":
            return "native HTTP responses provider"
    return inference_decline_reason(PARAMETERS, request)
