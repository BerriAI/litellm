import functools
import inspect
from collections.abc import Awaitable, Callable, Generator, Iterator, Mapping
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
from litellm.litellm_core_utils.asyncify import run_async_function
from litellm.litellm_core_utils.clientside_credential_handler import UPSTREAM_BOUND_PARAMS
from litellm.litellm_core_utils.core_helpers import (
    filter_internal_params,
    safe_deep_copy,
)
from litellm.types.utils import CallTypes, CustomPricingLiteLLMParams

FALLBACK_ATTEMPT_KWARG: Final = f"{INTERNAL_KWARG_PREFIX}fallback_attempt"

FallbackChain: TypeAlias = tuple[str | dict[str, object], ...]
_FALLBACK_CHAIN: Final = TypeAdapter(FallbackChain)
_REQUEST_KWARGS: Final = TypeAdapter(dict[str, object])
_NO_OVERRIDES: Final[Mapping[str, object]] = MappingProxyType({})
_RUNNER_PARAMS: Final = frozenset({"kwargs", "fallbacks", "acompletion", "model", "litellm_logging_obj"})
_PROVIDER_BOUND_PARAMS: Final = UPSTREAM_BOUND_PARAMS | {
    "custom_llm_provider",
    "client",
    "headers",
    "litellm_credential_name",
}
_MODEL_BOUND_PARAMS: Final = frozenset(CustomPricingLiteLLMParams.model_fields)
_GLOBAL_FALLBACKS_OFF: Final[ContextVar[bool]] = ContextVar("litellm_global_model_fallbacks_off", default=False)


@contextmanager
def without_global_fallbacks() -> Generator[None]:
    token: Final = _GLOBAL_FALLBACKS_OFF.set(True)
    try:
        yield
    finally:
        _GLOBAL_FALLBACKS_OFF.reset(token)


async def _awaited(pending: Awaitable[object]) -> object:
    return await pending


def without_global_fallbacks_around(endpoint: Callable[..., object]) -> Callable[..., object]:
    if not inspect.iscoroutinefunction(endpoint):

        @functools.wraps(endpoint)
        def call(*args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
            with without_global_fallbacks():
                return endpoint(*args, **kwargs)

        return call

    @functools.wraps(endpoint)
    async def acall(*args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        with without_global_fallbacks():
            result: Final = endpoint(*args, **kwargs)
            return await _awaited(result) if inspect.isawaitable(result) else result

    return acall


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
    requested_provider: Final = base_kwargs.get("custom_llm_provider")
    original_provider: Final = _provider_of(original_model, requested_provider)
    provider: Final = _provider_of(model, custom_llm_provider) or _provider_of(model, requested_provider)
    switches_provider: Final = original_provider is None or original_provider != provider
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


def _all_attempts_failed(last_error: Exception) -> Exception:
    return Exception(
        f"{last_error}. All fallback attempts failed. Enable verbose logging with `litellm.set_verbose=True` for details."
    )


async def async_completion_with_fallbacks(**kwargs: object) -> object:
    plan: Final = _fallback_plan(kwargs)
    primary: Final = _fallback_attempt(plan, plan.original_model)
    try:
        return add_fallback_headers_to_response(
            response=await litellm.acompletion(**primary.kwargs), attempted_fallbacks=0
        )
    except Exception as primary_error:  # noqa: BLE001  # every failed attempt moves on to the next fallback
        verbose_logger.exception("Fallback attempt failed for model %s: %s", primary.model, primary_error)
        last_error: Exception = primary_error  # rebind-ok: the newest failure names the final error
        for attempted_fallbacks, attempt in enumerate(_fallback_attempts(plan), start=1):
            try:
                return add_fallback_headers_to_response(
                    response=await litellm.acompletion(**attempt.kwargs), attempted_fallbacks=attempted_fallbacks
                )
            except Exception as fallback_error:  # noqa: BLE001  # any failed attempt moves on to the next fallback
                verbose_logger.exception("Fallback attempt failed for model %s: %s", attempt.model, fallback_error)
                last_error = fallback_error
        raise _all_attempts_failed(last_error) from last_error


def _without_cache_writes(request: Mapping[str, object]) -> Mapping[str, object]:
    return {**request, "cache": {**_REQUEST_KWARGS.validate_python(request.get("cache") or {}), "no-store": True}}


def _completion_with_deployment_hooks(request_kwargs: Mapping[str, object]) -> object:
    from litellm.utils import async_post_call_success_deployment_hook, async_pre_call_deployment_hook

    request: Final = _REQUEST_KWARGS.validate_python(
        run_async_function(async_pre_call_deployment_hook, request_kwargs, CallTypes.completion.value, include_v2=True)
    )
    if request.get("stream"):
        return litellm.completion(**request)
    response: Final = litellm.completion(**_without_cache_writes(request))
    return run_async_function(
        async_post_call_success_deployment_hook, request, response, CallTypes.completion, include_v2=True
    )


def completion_with_fallbacks(**kwargs: object) -> object:
    plan: Final = _fallback_plan(kwargs)
    primary: Final = _fallback_attempt(plan, plan.original_model)
    try:
        return add_fallback_headers_to_response(
            response=_completion_with_deployment_hooks(primary.kwargs), attempted_fallbacks=0
        )
    except Exception as primary_error:  # noqa: BLE001  # every failed attempt moves on to the next fallback
        verbose_logger.exception("Fallback attempt failed for model %s: %s", primary.model, primary_error)
        last_error: Exception = primary_error  # rebind-ok: the newest failure names the final error
        for attempted_fallbacks, attempt in enumerate(_fallback_attempts(plan), start=1):
            try:
                return add_fallback_headers_to_response(
                    response=_completion_with_deployment_hooks(attempt.kwargs), attempted_fallbacks=attempted_fallbacks
                )
            except Exception as fallback_error:  # noqa: BLE001  # any failed attempt moves on to the next fallback
                verbose_logger.exception("Fallback attempt failed for model %s: %s", attempt.model, fallback_error)
                last_error = fallback_error
        raise _all_attempts_failed(last_error) from last_error
