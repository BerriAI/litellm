from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, TypeAlias

from pydantic import TypeAdapter

import litellm
from litellm._logging import verbose_logger
from litellm._uuid import uuid
from litellm.constants import INTERNAL_KWARG_PREFIX
from litellm.litellm_core_utils.add_retry_fallback_headers import (
    add_fallback_headers_to_response,
)
from litellm.litellm_core_utils.clientside_credential_handler import UPSTREAM_BOUND_PARAMS
from litellm.litellm_core_utils.core_helpers import (
    filter_internal_params,
    safe_deep_copy,
)
from litellm.types.utils import CustomPricingLiteLLMParams

FALLBACK_ATTEMPT_KWARG: Final = f"{INTERNAL_KWARG_PREFIX}fallback_attempt"

FallbackChain: TypeAlias = tuple[str | dict[str, object], ...]
_FALLBACK_CHAIN: Final = TypeAdapter(FallbackChain)
_REQUEST_KWARGS: Final = TypeAdapter(dict[str, object])
_NO_OVERRIDES: Final[Mapping[str, object]] = MappingProxyType({})
_RUNNER_PARAMS: Final = frozenset({"kwargs", "fallbacks", "acompletion", "model", "litellm_logging_obj"})
_PROVIDER_BOUND_PARAMS: Final = UPSTREAM_BOUND_PARAMS | {"custom_llm_provider", "client"}
_MODEL_BOUND_PARAMS: Final = frozenset(CustomPricingLiteLLMParams.model_fields)


def _configured_fallbacks(request_kwargs: Mapping[str, object]) -> object:
    if request_kwargs.get(FALLBACK_ATTEMPT_KWARG) is True:
        return None
    per_call: Final = request_kwargs.get("fallbacks")
    return litellm.model_fallbacks if per_call is None else per_call


def uses_fallbacks(request_kwargs: Mapping[str, object]) -> bool:
    return bool(_configured_fallbacks(request_kwargs))


def resolve_fallbacks(request_kwargs: Mapping[str, object]) -> FallbackChain | None:
    configured: Final = _configured_fallbacks(request_kwargs)
    return _FALLBACK_CHAIN.validate_python(configured) if configured else None


@dataclass(frozen=True, slots=True)
class _FallbackAttempt:
    model: str
    kwargs: Mapping[str, object]


def _provider_of(model: str, custom_llm_provider: object) -> str | None:
    try:
        return litellm.get_llm_provider(
            model=model, custom_llm_provider=custom_llm_provider if isinstance(custom_llm_provider, str) else None
        )[1]
    except litellm.exceptions.BadRequestError:
        return None


def _inherited_kwargs(
    base_kwargs: Mapping[str, object], original_model: str, model: str, custom_llm_provider: object
) -> Mapping[str, object]:
    if model == original_model:
        return base_kwargs
    original_provider: Final = _provider_of(original_model, base_kwargs.get("custom_llm_provider"))
    provider: Final = _provider_of(model, custom_llm_provider)
    switches_provider: Final = None not in (original_provider, provider) and original_provider != provider
    dropped: Final = (_MODEL_BOUND_PARAMS | _PROVIDER_BOUND_PARAMS) if switches_provider else _MODEL_BOUND_PARAMS
    return {key: value for key, value in base_kwargs.items() if key not in dropped}


def _fallback_attempt(
    base_kwargs: Mapping[str, object], original_model: str, fallback: str | Mapping[str, object], logging_obj: object
) -> _FallbackAttempt:
    overrides: Final = _NO_OVERRIDES if isinstance(fallback, str) else fallback
    model: Final = fallback if isinstance(fallback, str) else str(overrides.get("model", original_model))
    inherited: Final = _inherited_kwargs(base_kwargs, original_model, model, overrides.get("custom_llm_provider"))
    request: Final = filter_internal_params(safe_deep_copy({**inherited, **overrides}))
    return _FallbackAttempt(
        model=model,
        kwargs={**request, "model": model, "litellm_logging_obj": logging_obj, FALLBACK_ATTEMPT_KWARG: True},
    )


def _fallback_attempts(runner_kwargs: Mapping[str, object]) -> tuple[_FallbackAttempt, ...]:
    nested_kwargs: Final = _REQUEST_KWARGS.validate_python(runner_kwargs.get("kwargs") or {})
    merged: Final = {**runner_kwargs, **nested_kwargs}
    original_model: Final = str(runner_kwargs["model"])
    base_kwargs: Final = {
        **{key: value for key, value in merged.items() if key not in _RUNNER_PARAMS},
        "litellm_call_id": merged.get("litellm_call_id") or str(uuid.uuid4()),
    }
    chain: Final = (original_model, *_FALLBACK_CHAIN.validate_python(nested_kwargs.get("fallbacks") or ()))
    return tuple(
        _fallback_attempt(base_kwargs, original_model, fallback, merged.get("litellm_logging_obj"))
        for fallback in chain
    )


async def async_completion_with_fallbacks(**kwargs: object) -> object:
    primary, *fallbacks = _fallback_attempts(kwargs)
    try:
        return add_fallback_headers_to_response(
            response=await litellm.acompletion(**primary.kwargs), attempted_fallbacks=0
        )
    except Exception as primary_error:
        verbose_logger.exception("Fallback attempt failed for model %s: %s", primary.model, primary_error)
        for attempted_fallbacks, attempt in enumerate(fallbacks, start=1):
            try:
                return add_fallback_headers_to_response(
                    response=await litellm.acompletion(**attempt.kwargs), attempted_fallbacks=attempted_fallbacks
                )
            except Exception as fallback_error:  # noqa: BLE001  # any failed attempt moves on to the next fallback
                verbose_logger.exception("Fallback attempt failed for model %s: %s", attempt.model, fallback_error)
        raise


def completion_with_fallbacks(**kwargs: object) -> object:
    primary, *fallbacks = _fallback_attempts(kwargs)
    try:
        return add_fallback_headers_to_response(response=litellm.completion(**primary.kwargs), attempted_fallbacks=0)
    except Exception as primary_error:
        verbose_logger.exception("Fallback attempt failed for model %s: %s", primary.model, primary_error)
        for attempted_fallbacks, attempt in enumerate(fallbacks, start=1):
            try:
                return add_fallback_headers_to_response(
                    response=litellm.completion(**attempt.kwargs), attempted_fallbacks=attempted_fallbacks
                )
            except Exception as fallback_error:  # noqa: BLE001  # any failed attempt moves on to the next fallback
                verbose_logger.exception("Fallback attempt failed for model %s: %s", attempt.model, fallback_error)
        raise
