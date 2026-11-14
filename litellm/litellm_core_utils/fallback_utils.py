from collections.abc import Generator, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
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
_GLOBAL_FALLBACKS_OFF: Final[ContextVar[bool]] = ContextVar("litellm_global_model_fallbacks_off", default=False)


@contextmanager
def without_global_fallbacks() -> Generator[None]:
    token: Final = _GLOBAL_FALLBACKS_OFF.set(True)
    try:
        yield
    finally:
        _GLOBAL_FALLBACKS_OFF.reset(token)


def _configured_fallbacks(request_kwargs: Mapping[str, object]) -> object:
    if request_kwargs.get(FALLBACK_ATTEMPT_KWARG) is True:
        return None
    per_call: Final = request_kwargs.get("fallbacks")
    if per_call is not None:
        return per_call
    return None if _GLOBAL_FALLBACKS_OFF.get() else litellm.model_fallbacks


def uses_fallbacks(request_kwargs: Mapping[str, object]) -> bool:
    return bool(_configured_fallbacks(request_kwargs))


def resolve_fallbacks(request_kwargs: Mapping[str, object]) -> FallbackChain | None:
    configured: Final = _configured_fallbacks(request_kwargs)
    return _FALLBACK_CHAIN.validate_python(configured) if configured else None


@dataclass(frozen=True, slots=True)
class _FallbackAttempt:
    model: str
    kwargs: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _FallbackPlan:
    original_model: str
    base_kwargs: Mapping[str, object]
    fallbacks: FallbackChain
    logging_obj: object


def _provider_of(model: str, custom_llm_provider: object) -> str | None:
    from litellm.litellm_core_utils.get_llm_provider_logic import resolve_model_and_provider_for_metadata

    try:
        return resolve_model_and_provider_for_metadata(
            model, custom_llm_provider if isinstance(custom_llm_provider, str) else None
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


def _fallback_attempt(plan: _FallbackPlan, fallback: str | Mapping[str, object]) -> _FallbackAttempt:
    overrides: Final = _NO_OVERRIDES if isinstance(fallback, str) else fallback
    model: Final = fallback if isinstance(fallback, str) else str(overrides.get("model", plan.original_model))
    inherited: Final = _inherited_kwargs(
        plan.base_kwargs, plan.original_model, model, overrides.get("custom_llm_provider")
    )
    request: Final = filter_internal_params(safe_deep_copy({**inherited, **overrides}))
    return _FallbackAttempt(
        model=model,
        kwargs={**request, "model": model, "litellm_logging_obj": plan.logging_obj, FALLBACK_ATTEMPT_KWARG: True},
    )


def _fallback_plan(runner_kwargs: Mapping[str, object]) -> _FallbackPlan:
    nested_kwargs: Final = _REQUEST_KWARGS.validate_python(runner_kwargs.get("kwargs") or {})
    merged: Final = {**runner_kwargs, **nested_kwargs}
    return _FallbackPlan(
        original_model=str(runner_kwargs["model"]),
        base_kwargs={
            **{key: value for key, value in merged.items() if key not in _RUNNER_PARAMS},
            "litellm_call_id": merged.get("litellm_call_id") or str(uuid.uuid4()),
        },
        fallbacks=_FALLBACK_CHAIN.validate_python(nested_kwargs.get("fallbacks") or ()),
        logging_obj=merged.get("litellm_logging_obj"),
    )


def _fallback_attempts(plan: _FallbackPlan) -> Iterator[_FallbackAttempt]:
    return (_fallback_attempt(plan, fallback) for fallback in plan.fallbacks)


async def async_completion_with_fallbacks(**kwargs: object) -> object:
    plan: Final = _fallback_plan(kwargs)
    primary: Final = _fallback_attempt(plan, plan.original_model)
    try:
        return add_fallback_headers_to_response(
            response=await litellm.acompletion(**primary.kwargs), attempted_fallbacks=0
        )
    except Exception as primary_error:
        verbose_logger.exception("Fallback attempt failed for model %s: %s", primary.model, primary_error)
        for attempted_fallbacks, attempt in enumerate(_fallback_attempts(plan), start=1):
            try:
                return add_fallback_headers_to_response(
                    response=await litellm.acompletion(**attempt.kwargs), attempted_fallbacks=attempted_fallbacks
                )
            except Exception as fallback_error:  # noqa: BLE001  # any failed attempt moves on to the next fallback
                verbose_logger.exception("Fallback attempt failed for model %s: %s", attempt.model, fallback_error)
        raise


def completion_with_fallbacks(**kwargs: object) -> object:
    plan: Final = _fallback_plan(kwargs)
    primary: Final = _fallback_attempt(plan, plan.original_model)
    try:
        return add_fallback_headers_to_response(response=litellm.completion(**primary.kwargs), attempted_fallbacks=0)
    except Exception as primary_error:
        verbose_logger.exception("Fallback attempt failed for model %s: %s", primary.model, primary_error)
        for attempted_fallbacks, attempt in enumerate(_fallback_attempts(plan), start=1):
            try:
                return add_fallback_headers_to_response(
                    response=litellm.completion(**attempt.kwargs), attempted_fallbacks=attempted_fallbacks
                )
            except Exception as fallback_error:  # noqa: BLE001  # any failed attempt moves on to the next fallback
                verbose_logger.exception("Fallback attempt failed for model %s: %s", attempt.model, fallback_error)
        raise
