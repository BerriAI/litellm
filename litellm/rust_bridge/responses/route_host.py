from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, cast  # noqa: TID251  # narrows caller json the checker cannot see into

import litellm
from litellm import get_llm_provider
from litellm.llms.azure.responses.transformation import AzureOpenAIResponsesAPIConfig
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.rust_bridge import failures
from litellm.rust_bridge.public_call import inference_decline_reason, optional_str
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams, ResponsesAPIResponse
from litellm.types.router import GenericLiteLLMParams

PARAMETERS: Final = tuple(ResponsesAPIOptionalRequestParams.__annotations__)
_NATIVE_PARAMETERS: Final = (*PARAMETERS, "api_version")
_CONFIGS: Final[Mapping[str, OpenAIResponsesAPIConfig]] = MappingProxyType(
    {"openai": OpenAIResponsesAPIConfig(), "azure": AzureOpenAIResponsesAPIConfig()}
)
_AZURE_KEY_ENVIRONMENT: Final = ("AZURE_OPENAI_API_KEY", "AZURE_API_KEY")


def connection_defaults(provider: str) -> tuple[str | None, str | None]:
    provider_key: Final = {"openai": litellm.openai_key, "azure": litellm.azure_key}.get(provider)
    return litellm.api_key or provider_key, litellm.api_base


def response(value: Mapping[str, object]) -> ResponsesAPIResponse:
    return ResponsesAPIResponse.model_validate(value)


def map_failure(error: Exception, request: Mapping[str, object]) -> Exception:
    _, provider = _resolved_model(request)
    return failures.map_native_failure(
        error,
        str(request["model"]),
        provider,
        request,
        optional_str(request.get("api_base")) or optional_str(request.get("base_url")),
        python_responses_shape=True,
    )


def _resolved_model(request: Mapping[str, object]) -> tuple[str, str]:
    model, provider, _, _ = get_llm_provider(
        model=str(request["model"]), custom_llm_provider=optional_str(request.get("custom_llm_provider"))
    )
    return model, provider


def decline_reason(request: Mapping[str, object]) -> str | None:
    try:
        model, provider = _resolved_model(request)
    except litellm.exceptions.BadRequestError:
        return "native Responses could not resolve the provider"
    config: Final = _CONFIGS.get(provider)
    if config is None:
        return "native HTTP responses provider"
    return (
        inference_decline_reason(_NATIVE_PARAMETERS, request)
        or (_azure_decline_reason(model, request) if provider == "azure" else None)
        or _model_rules_decline_reason(config, model, request)
    )


def _azure_decline_reason(model: str, request: Mapping[str, object]) -> str | None:
    if "o_series/" in model:
        return "native Azure Responses does not implement the o-series config"
    if request.get("context_management") is not None:
        return "native Azure Responses does not implement context_management"
    extra_headers: Final = request.get("extra_headers")
    forwarded_key: Final = isinstance(extra_headers, Mapping) and "api-key" in extra_headers
    key: Final = (
        optional_str(request.get("api_key"))
        or litellm.api_key
        or litellm.azure_key
        or next(filter(None, (get_secret_str(name) for name in _AZURE_KEY_ENVIRONMENT)), None)
    )
    if not forwarded_key and not key:
        return "native Azure Responses does not implement Azure AD token auth"
    return None


def _model_rules_decline_reason(
    config: OpenAIResponsesAPIConfig, model: str, request: Mapping[str, object]
) -> str | None:
    """Request rewrites that read the model cost map or Python's regex engine stay on Python"""
    params: Final = {name: request[name] for name in PARAMETERS if request.get(name) is not None}
    try:
        config.map_openai_params(params, model, drop_params=False)  # pyright: ignore[reportArgumentType, reportUnknownMemberType]  # the same rules the Python route applies
    except litellm.UnsupportedParamsError:
        return "native Responses does not implement the model's sampling rules"
    litellm_params: Final = GenericLiteLLMParams(
        api_base=optional_str(request.get("api_base")) or optional_str(request.get("base_url")),
        model_info=request.get("model_info"),
    )
    if _repairs_tool_schemas(config, model, request.get("tools"), litellm_params):
        return "native Responses does not implement tool schema validation and repair"
    if _carries_cache_control(request.get("input")) and config._reads_prompt_cache_breakpoints(  # pyright: ignore[reportPrivateUsage]  # the same rules the Python route applies
        model=model, litellm_params=litellm_params, prompt_cache_options=request.get("prompt_cache_options")
    ):
        return "native Responses does not implement prompt cache breakpoints"
    return None


def _repairs_tool_schemas(
    config: OpenAIResponsesAPIConfig, model: str, tools: object, litellm_params: GenericLiteLLMParams
) -> bool:
    if not isinstance(tools, list):
        return False
    try:
        decoded: Final = config._tools_with_object_parameters(model=model, tools=cast(list[object], tools))  # pyright: ignore[reportPrivateUsage, reportArgumentType]  # cast-ok: caller json
    except litellm.BadRequestError:
        return True
    return (
        config._sanitized_tool_schemas_for_openai(model=model, tools=decoded, litellm_params=litellm_params)  # pyright: ignore[reportPrivateUsage]  # the same rules the Python route applies
        is not decoded
    )


def _carries_cache_control(value: object) -> bool:
    pending: Final[list[object]] = [value]  # mutable-ok: work stack for a depth-first walk of caller json
    while pending:
        current = pending.pop()
        if isinstance(current, Mapping):
            fields = cast(Mapping[str, object], current)  # cast-ok: caller json
            if "cache_control" in fields:
                return True
            pending.extend(fields.values())
        elif isinstance(current, list):
            pending.extend(cast(list[object], current))  # cast-ok: caller json
    return False
