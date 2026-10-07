from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final

import httpx
from pydantic import TypeAdapter, ValidationError

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.decisions.transformation import BaseDecisionsConfig
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.types.decisions import (
    DecisionQuestion,
    DecisionsJSON,
    DecisionsRequest,
    DecisionsResponse,
)
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager, client

_DECISIONS_REQUEST_ADAPTER: Final[TypeAdapter[DecisionsRequest]] = TypeAdapter(DecisionsRequest)
_HANDLER: Final = BaseLLMHTTPHandler()


@dataclass(frozen=True, slots=True, repr=False)
class _DecisionsCall:
    model: str
    custom_llm_provider: str
    provider_config: BaseDecisionsConfig
    request: DecisionsRequest
    api_base: str
    api_key: str | None = field(repr=False)
    logging_obj: LiteLLMLoggingObj | None
    headers: Mapping[str, str]
    timeout: float | httpx.Timeout | None


def _supported_providers() -> tuple[str, ...]:
    return tuple(
        provider.value
        for provider in LlmProviders
        if ProviderConfigManager.get_provider_decisions_config(model="", provider=provider) is not None
    )


def _provider_config(model: str, custom_llm_provider: str) -> BaseDecisionsConfig:
    provider: Final = next((member for member in LlmProviders if member.value == custom_llm_provider), None)
    provider_config: Final = (
        None if provider is None else ProviderConfigManager.get_provider_decisions_config(model, provider)
    )
    if provider_config is None:
        supported: Final = ", ".join(_supported_providers())
        raise litellm.BadRequestError(
            message=f"Unknown Decisions provider '{custom_llm_provider}'. Supported providers: {supported}",
            model=model,
            llm_provider=custom_llm_provider,
        )
    return provider_config


def _prepare_call(
    *,
    model: str,
    state: DecisionsJSON,
    questions: Mapping[str, DecisionQuestion | Mapping[str, object]],
    api_key: str | None,
    api_base: str | None,
    timeout: float | httpx.Timeout | None,
    custom_llm_provider: str | None,
    extra_headers: Mapping[str, str] | None,
    kwargs: Mapping[str, object],
) -> _DecisionsCall:
    upstream_model, provider, dynamic_api_key, dynamic_api_base = litellm.get_llm_provider(
        model=model,
        custom_llm_provider=custom_llm_provider,
        api_base=api_base,
        api_key=api_key,
    )
    provider_config: Final = _provider_config(upstream_model, provider)
    canonical_model: Final = provider_config.canonical_model(upstream_model)
    if not upstream_model:
        raise litellm.BadRequestError(
            message="A model name is required for the Decisions API",
            model=model,
            llm_provider=provider,
        )
    try:
        request: Final = _DECISIONS_REQUEST_ADAPTER.validate_python(
            {"model": canonical_model, "state": state, "questions": questions}
        )
    except ValidationError as error:
        raise litellm.BadRequestError(
            message=f"Invalid Decisions request: {error}",
            model=model,
            llm_provider=provider,
        ) from error

    resolved_api_base: Final = provider_config.resolve_api_base(dynamic_api_base or api_base)
    if resolved_api_base is None:
        raise litellm.BadRequestError(
            message=provider_config.missing_api_base_message(provider),
            model=model,
            llm_provider=provider,
        )
    resolved_api_key: Final = provider_config.resolve_api_key(dynamic_api_key or api_key)
    if resolved_api_key is None and provider_config.api_key_required:
        raise litellm.AuthenticationError(
            message=f"Missing API key for Decisions provider '{provider}'",
            model=model,
            llm_provider=provider,
        )

    logging_obj: Final = kwargs.get("litellm_logging_obj")
    if isinstance(logging_obj, LiteLLMLoggingObj):
        logging_obj.update_from_kwargs(
            kwargs=dict(kwargs),
            model=canonical_model,
            litellm_params={
                "litellm_call_id": kwargs.get("litellm_call_id"),
                "api_base": provider_config.get_complete_url(resolved_api_base, canonical_model),
            },
            custom_llm_provider=provider,
        )
    return _DecisionsCall(
        model=canonical_model,
        custom_llm_provider=provider,
        provider_config=provider_config,
        request=request,
        api_base=resolved_api_base,
        api_key=resolved_api_key,
        logging_obj=logging_obj if isinstance(logging_obj, LiteLLMLoggingObj) else None,
        headers=extra_headers or {},
        timeout=timeout,
    )


def _map_upstream_exception(error: Exception, call: _DecisionsCall) -> Exception:
    return litellm.exception_type(
        model=f"{call.custom_llm_provider}/{call.model}",
        custom_llm_provider=call.custom_llm_provider,
        original_exception=error,
    )


@client
async def adecisions(
    model: str,
    state: DecisionsJSON,
    questions: Mapping[str, DecisionQuestion | Mapping[str, object]],
    api_key: str | None = None,
    api_base: str | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
    **kwargs: object,
) -> DecisionsResponse:
    call: Final = _prepare_call(
        model=model,
        state=state,
        questions=questions,
        api_key=api_key,
        api_base=api_base,
        timeout=timeout,
        custom_llm_provider=custom_llm_provider,
        extra_headers=extra_headers,
        kwargs=kwargs,
    )
    try:
        return await _HANDLER.adecisions(
            model=call.model,
            custom_llm_provider=call.custom_llm_provider,
            logging_obj=call.logging_obj,
            provider_config=call.provider_config,
            request=call.request,
            api_base=call.api_base,
            api_key=call.api_key,
            headers=call.headers,
            timeout=call.timeout,
        )
    except Exception as error:
        raise _map_upstream_exception(error, call) from error


@client
def decisions(
    model: str,
    state: DecisionsJSON,
    questions: Mapping[str, DecisionQuestion | Mapping[str, object]],
    api_key: str | None = None,
    api_base: str | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
    **kwargs: object,
) -> DecisionsResponse:
    call: Final = _prepare_call(
        model=model,
        state=state,
        questions=questions,
        api_key=api_key,
        api_base=api_base,
        timeout=timeout,
        custom_llm_provider=custom_llm_provider,
        extra_headers=extra_headers,
        kwargs=kwargs,
    )
    try:
        return _HANDLER.decisions(
            model=call.model,
            custom_llm_provider=call.custom_llm_provider,
            logging_obj=call.logging_obj,
            provider_config=call.provider_config,
            request=call.request,
            api_base=call.api_base,
            api_key=call.api_key,
            headers=call.headers,
            timeout=call.timeout,
        )
    except Exception as error:
        raise _map_upstream_exception(error, call) from error


__all__ = ["adecisions", "decisions"]
