from __future__ import annotations

from collections.abc import Mapping
from typing import Final, cast  # noqa: TID251  # narrows caller json the checker cannot see into

import litellm
from litellm import get_llm_provider
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.rust_bridge import failures
from litellm.rust_bridge.public_call import inference_decline_reason, optional_str
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams, ResponsesAPIResponse
from litellm.types.router import GenericLiteLLMParams

PARAMETERS: Final = tuple(ResponsesAPIOptionalRequestParams.__annotations__)
_OPENAI_CONFIG: Final = OpenAIResponsesAPIConfig()


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
    return inference_decline_reason(PARAMETERS, request) or _model_rules_decline_reason(request)


def _model_rules_decline_reason(request: Mapping[str, object]) -> str | None:
    """Request rewrites that read the model cost map or Python's regex engine stay on Python"""
    model: Final = str(request["model"]).removeprefix("openai/")
    params: Final = {name: request[name] for name in PARAMETERS if request.get(name) is not None}
    try:
        _OPENAI_CONFIG.map_openai_params(params, model, drop_params=False)  # pyright: ignore[reportArgumentType, reportUnknownMemberType]  # the same rules the Python route applies
    except litellm.UnsupportedParamsError:
        return "native Responses does not implement the model's sampling rules"
    litellm_params: Final = GenericLiteLLMParams(
        api_base=optional_str(request.get("api_base")) or optional_str(request.get("base_url")),
        model_info=request.get("model_info"),
    )
    if _repairs_tool_schemas(model, request.get("tools"), litellm_params):
        return "native Responses does not implement tool schema repair"
    if _carries_cache_control(request.get("input")) and _OPENAI_CONFIG._reads_prompt_cache_breakpoints(  # pyright: ignore[reportPrivateUsage]  # the same rules the Python route applies
        model=model, litellm_params=litellm_params, prompt_cache_options=request.get("prompt_cache_options")
    ):
        return "native Responses does not implement prompt cache breakpoints"
    return None


def _repairs_tool_schemas(model: str, tools: object, litellm_params: GenericLiteLLMParams) -> bool:
    if not isinstance(tools, list):
        return False
    try:
        decoded: Final = _OPENAI_CONFIG._tools_with_object_parameters(model=model, tools=cast(list[object], tools))  # pyright: ignore[reportPrivateUsage, reportArgumentType]  # cast-ok: caller json
    except litellm.BadRequestError:
        return False
    return (
        _OPENAI_CONFIG._sanitized_tool_schemas_for_openai(model=model, tools=decoded, litellm_params=litellm_params)  # pyright: ignore[reportPrivateUsage]  # the same rules the Python route applies
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
