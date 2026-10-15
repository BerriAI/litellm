from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final, Protocol, cast  # noqa: TID251  # narrows caller json the checker cannot see into

import httpx

import litellm
from litellm import get_llm_provider
from litellm.litellm_core_utils import core_helpers
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.azure.responses.transformation import AzureOpenAIResponsesAPIConfig
from litellm.llms.bedrock_mantle.responses.transformation import BedrockMantleResponsesAPIConfig
from litellm.llms.databricks.responses.transformation import DatabricksResponsesAPIConfig
from litellm.llms.fireworks_ai.common_utils import get_fireworks_session_id
from litellm.llms.fireworks_ai.responses.transformation import FireworksAIResponsesAPIConfig
from litellm.llms.hosted_vllm.responses.transformation import HostedVLLMResponsesAPIConfig
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.llms.xai.responses.transformation import XAIResponsesAPIConfig
from litellm.rust_bridge import failures
from litellm.rust_bridge.public_call import inference_decline_reason, optional_str
from litellm.rust_bridge.responses.streams import StreamSource
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams, ResponsesAPIResponse
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

PARAMETERS: Final = tuple(ResponsesAPIOptionalRequestParams.__annotations__)
_REQUEST_FIELDS: Final = frozenset((*PARAMETERS, "model", "input", "litellm_logging_obj"))
_NATIVE_PARAMETERS: Final = (*PARAMETERS, "api_version")
_MANTLE_PARAMETERS: Final = (
    *_NATIVE_PARAMETERS,
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
    "aws_region_name",
    "aws_session_name",
    "aws_profile_name",
    "aws_role_name",
    "aws_web_identity_token",
    "aws_sts_endpoint",
    "aws_external_id",
    "aws_bedrock_project_id",
)
_CONFIGS: Final[Mapping[str, OpenAIResponsesAPIConfig]] = MappingProxyType(
    {
        "openai": OpenAIResponsesAPIConfig(),
        "azure": AzureOpenAIResponsesAPIConfig(),
        "xai": XAIResponsesAPIConfig(),
        "hosted_vllm": HostedVLLMResponsesAPIConfig(),
        "fireworks_ai": FireworksAIResponsesAPIConfig(),
        "databricks": DatabricksResponsesAPIConfig(),
    }
)
_AZURE_KEY_ENVIRONMENT: Final = ("AZURE_OPENAI_API_KEY", "AZURE_API_KEY")


class _HeaderProcessor(Protocol):
    def __call__(self, response_headers: httpx.Headers) -> Mapping[str, object]: ...


_PROCESS_HEADERS: Final = cast(  # cast-ok: the shared header projection is untyped
    _HeaderProcessor, core_helpers.process_response_headers
)


def connection_defaults(provider: str) -> tuple[str | None, str | None]:
    if provider in ("databricks", "bedrock_mantle"):
        return None, None
    provider_key: Final = {"openai": litellm.openai_key, "azure": litellm.azure_key}.get(provider)
    return litellm.api_key or provider_key, litellm.api_base


def provider_connection(
    model: str, custom_llm_provider: str | None, api_key: str | None, api_base: str | None
) -> tuple[str | None, str | None]:
    """The key and base `get_llm_provider` fills in before the provider config reads its own defaults"""
    _, _, provider_key, provider_base = get_llm_provider(
        model=model, custom_llm_provider=custom_llm_provider, api_key=api_key, api_base=api_base
    )
    return provider_key, provider_base


def provider_params(model: str, custom_llm_provider: str | None) -> tuple[tuple[str, object], ...]:
    """The provider settings Python reads from the model cost map, as litellm params"""
    resolved, provider = _resolved_model({"model": model, "custom_llm_provider": custom_llm_provider})
    if provider != "bedrock_mantle":
        return ()
    config: Final = _mantle_config(resolved)
    return () if config is None else (("use_openai_responses_path", config.use_openai_path),)


def _mantle_config(model: str) -> BedrockMantleResponsesAPIConfig | None:
    """The config Python's dispatch picks from the model cost map, or None for chat-only models"""
    config: Final = ProviderConfigManager.get_provider_responses_api_config(LlmProviders.BEDROCK_MANTLE, model)
    return config if isinstance(config, BedrockMantleResponsesAPIConfig) else None


def response(value: Mapping[str, object]) -> ResponsesAPIResponse:
    return ResponsesAPIResponse.model_validate(value)


def stream_head(headers: Sequence[tuple[str, str]], request: Mapping[str, object]) -> Mapping[str, object]:
    """The opened stream's headers for call logging, and what Python's Responses stream iterator is built from"""
    model, provider = _resolved_model(request)
    config: Final = _responses_config(provider, model)
    logger: Final = request.get("litellm_logging_obj")
    if config is None or not isinstance(logger, Logging):
        raise RuntimeError("a native Responses stream opened without its provider config or call logger")
    return {
        "additional_headers": dict(_PROCESS_HEADERS(httpx.Headers(list(headers)))),
        "source": StreamSource(
            headers=tuple(headers),
            model=model,
            config=config,
            logging_obj=logger,
            litellm_metadata=_litellm_metadata(request),
            custom_llm_provider=provider,
            request_data={
                "input": request.get("input"),
                **{name: request[name] for name in PARAMETERS if request.get(name) is not None},
                "litellm_params": dict(
                    GenericLiteLLMParams(
                        **{name: value for name, value in request.items() if name not in _REQUEST_FIELDS}
                    )
                ),
            },
            call_kwargs=request,
        ),
    }


def _litellm_metadata(
    request: Mapping[str, object],
) -> dict[str, object] | None:  # mutable-ok: the caller's own dict, which the stream iterator writes into
    metadata: Final = request.get("litellm_metadata", {})
    if not isinstance(metadata, dict):
        return None
    return cast(dict[str, object], metadata)  # cast-ok: caller json


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
    config: Final = _responses_config(provider, model)
    if config is None:
        return "native HTTP responses provider"
    return (
        inference_decline_reason(_MANTLE_PARAMETERS if provider == "bedrock_mantle" else _NATIVE_PARAMETERS, request)
        or _provider_decline_reason(provider, model, request)
        or _model_rules_decline_reason(config, model, request)
        or _stream_decline_reason(config, provider, model, request)
    )


def _stream_decline_reason(
    config: OpenAIResponsesAPIConfig, provider: str, model: str, request: Mapping[str, object]
) -> str | None:
    if request.get("stream") is not True:
        return None
    if litellm.cache is not None:
        return "native Responses streaming does not implement caching"
    if config.should_fake_stream(model=model, stream=True, custom_llm_provider=provider):
        return "native Responses does not implement faked streams"
    return None


def _responses_config(provider: str, model: str) -> OpenAIResponsesAPIConfig | None:
    if provider != "bedrock_mantle":
        return _CONFIGS.get(provider)
    return _mantle_config(model)


def _provider_decline_reason(provider: str, model: str, request: Mapping[str, object]) -> str | None:
    if provider == "azure":
        return _azure_decline_reason(model, request)
    if provider == "databricks":
        return _databricks_decline_reason(model, request)
    if provider not in ("xai", "hosted_vllm", "fireworks_ai"):
        return None
    provider_key, provider_base = provider_connection(
        str(request["model"]),
        optional_str(request.get("custom_llm_provider")),
        optional_str(request.get("api_key")),
        optional_str(request.get("api_base")) or optional_str(request.get("base_url")),
    )
    if provider == "xai" and not (provider_key or litellm.api_key):
        return "native xAI Responses does not implement OAuth"
    if provider == "hosted_vllm" and not provider_base:
        return "native hosted vLLM Responses needs an api_base"
    if provider == "fireworks_ai" and not provider_key:
        return "native Fireworks Responses needs an api key"
    if provider == "fireworks_ai" and get_fireworks_session_id({"metadata": request.get("litellm_metadata")}):
        return "native Fireworks Responses does not implement session affinity"
    return None


def _databricks_decline_reason(model: str, request: Mapping[str, object]) -> str | None:
    if "gpt" not in model.lower():
        return "Databricks serves Responses only on GPT endpoints"
    api_base: Final = (
        optional_str(request.get("api_base"))
        or optional_str(request.get("base_url"))
        or os.getenv("DATABRICKS_API_BASE")
    )
    if not api_base:
        return "native Databricks Responses does not implement Databricks SDK auth"
    if os.getenv("DATABRICKS_CLIENT_ID") and os.getenv("DATABRICKS_CLIENT_SECRET"):
        return "native Databricks Responses does not implement OAuth M2M"
    if not (optional_str(request.get("api_key")) or os.getenv("DATABRICKS_API_KEY")):
        return "native Databricks Responses does not implement Databricks SDK auth"
    return None


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
    if _has_file_search_tool(request.get("tools")) and not config.supports_native_file_search():
        return "native Responses does not implement emulated file_search"
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


def _has_file_search_tool(tools: object) -> bool:
    return isinstance(tools, list) and any(
        isinstance(tool, Mapping)
        and cast(Mapping[str, object], tool).get("type") == "file_search"  # cast-ok: caller json
        for tool in cast(list[object], tools)  # cast-ok: caller json
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
