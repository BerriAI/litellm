"""
Separate ITPM/OTPM (input/output tokens per minute) deployment rate limits.

- Pre-call: reserve input_tokens + max_tokens against ITPM
- Post-call: charge completion_tokens on OTPM; reconcile ITPM
- Cached prompt-read tokens are excluded from ITPM post-call accounting

Used by ModelRateLimitingCheck when a deployment sets itpm/otpm.
"""

from __future__ import annotations

import contextvars
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import httpx

import litellm
from litellm import token_counter
from litellm._logging import verbose_router_logger
from litellm.caching.dual_cache import DualCache
from litellm.types.router import RouterCacheEnum, RouterErrors
from litellm.types.utils import StandardLoggingPayload
from litellm.utils import get_utc_datetime

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span = Union[_Span, Any]
else:
    Span = Any

RoutingArgsTTL = 60

_io_token_rate_limit_request_kwargs: contextvars.ContextVar[
    Optional[Dict[str, Any]]
] = contextvars.ContextVar(
    "io_token_rate_limit_request_kwargs",
    default=None,
)

ITPM_RESERVED_KEY = "_litellm_itpm_reserved"
MAX_TOKENS_RESERVED_KEY = "_litellm_max_tokens_reserved"
ESTIMATED_INPUT_KEY = "_litellm_estimated_input"


def set_io_token_rate_limit_request_kwargs(kwargs: Optional[Dict[str, Any]]) -> None:
    _io_token_rate_limit_request_kwargs.set(kwargs)


def get_io_token_rate_limit_request_kwargs() -> Optional[Dict[str, Any]]:
    return _io_token_rate_limit_request_kwargs.get()


def seconds_until_minute_reset() -> int:
    dt = get_utc_datetime()
    return max(1, 60 - dt.second)


def get_deployment_io_token_limits(
    deployment: Dict,
) -> Tuple[Optional[int], Optional[int]]:
    itpm = deployment.get("itpm")
    otpm = deployment.get("otpm")
    litellm_params = deployment.get("litellm_params") or {}
    model_info = deployment.get("model_info") or {}
    if itpm is None:
        itpm = litellm_params.get("itpm")
    if otpm is None:
        otpm = litellm_params.get("otpm")
    if itpm is None:
        itpm = model_info.get("itpm")
    if otpm is None:
        otpm = model_info.get("otpm")
    return itpm, otpm


def deployment_has_io_token_limits(deployment: Dict) -> bool:
    itpm, otpm = get_deployment_io_token_limits(deployment)
    return itpm is not None or otpm is not None


def _get_cache_keys(deployment: Dict, current_minute: str) -> Tuple[str, str]:
    model_id = deployment.get("model_info", {}).get("id")
    deployment_name = deployment.get("litellm_params", {}).get("model")
    itpm_key = RouterCacheEnum.ITPM.value.format(
        id=model_id, model=deployment_name, current_minute=current_minute
    )
    otpm_key = RouterCacheEnum.OTPM.value.format(
        id=model_id, model=deployment_name, current_minute=current_minute
    )
    return itpm_key, otpm_key


def _estimate_input_tokens(request_kwargs: Optional[Dict[str, Any]]) -> int:
    if not request_kwargs:
        return 0
    messages = request_kwargs.get("messages")
    prompt = request_kwargs.get("prompt")
    input_text = request_kwargs.get("input")
    try:
        return max(0, int(token_counter(messages=messages, text=prompt or input_text)))
    except Exception:
        return 0


def _resolve_max_tokens(
    request_kwargs: Optional[Dict[str, Any]], deployment: Dict
) -> int:
    if request_kwargs:
        explicit = request_kwargs.get("max_tokens") or request_kwargs.get(
            "max_completion_tokens"
        )
        if explicit is not None:
            return max(0, int(explicit))

    model_name = (deployment.get("litellm_params") or {}).get("model")
    if model_name:
        try:
            info = litellm.get_model_info(model=model_name)
            model_max = info.get("max_output_tokens") or info.get("max_tokens")
            if model_max is not None:
                return max(0, int(model_max))
        except Exception:
            pass
    return 4096


def _get_usage_tokens(usage: Any) -> Tuple[int, int, int]:
    if usage is None:
        return 0, 0, 0
    if hasattr(usage, "prompt_tokens"):
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        cached = 0
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            cached = int(getattr(details, "cached_tokens", 0) or 0)
        return prompt, completion, cached
    if isinstance(usage, dict):
        prompt = int(usage.get("prompt_tokens", 0) or 0)
        completion = int(usage.get("completion_tokens", 0) or 0)
        details = usage.get("prompt_tokens_details") or {}
        cached = (
            int(details.get("cached_tokens", 0) or 0)
            if isinstance(details, dict)
            else 0
        )
        return prompt, completion, cached
    return 0, 0, 0


def _stash_reservation_in_metadata(
    request_kwargs: Optional[Dict[str, Any]],
    *,
    itpm_reserved: int,
    max_tokens: int,
    estimated_input: int,
) -> None:
    if not request_kwargs:
        return
    for channel in ("metadata", "litellm_metadata"):
        existing = request_kwargs.get(channel)
        if isinstance(existing, dict):
            existing[ITPM_RESERVED_KEY] = itpm_reserved
            existing[MAX_TOKENS_RESERVED_KEY] = max_tokens
            existing[ESTIMATED_INPUT_KEY] = estimated_input
        elif channel == "metadata":
            request_kwargs[channel] = {
                ITPM_RESERVED_KEY: itpm_reserved,
                MAX_TOKENS_RESERVED_KEY: max_tokens,
                ESTIMATED_INPUT_KEY: estimated_input,
            }


def _read_reservation_from_kwargs(kwargs: Any) -> Tuple[int, int, int]:
    for channel in ("metadata", "litellm_metadata"):
        channel_dict = None
        if isinstance(kwargs, dict):
            channel_dict = kwargs.get(channel)
            litellm_params = kwargs.get("litellm_params")
            if isinstance(litellm_params, dict) and isinstance(
                litellm_params.get("metadata"), dict
            ):
                channel_dict = litellm_params.get("metadata")
        if isinstance(channel_dict, dict) and ITPM_RESERVED_KEY in channel_dict:
            return (
                int(channel_dict.get(ITPM_RESERVED_KEY, 0) or 0),
                int(channel_dict.get(MAX_TOKENS_RESERVED_KEY, 0) or 0),
                int(channel_dict.get(ESTIMATED_INPUT_KEY, 0) or 0),
            )
    standard_logging_object = (
        kwargs.get("standard_logging_object") if isinstance(kwargs, dict) else None
    )
    if isinstance(standard_logging_object, dict):
        metadata = standard_logging_object.get("metadata") or {}
        if ITPM_RESERVED_KEY in metadata:
            return (
                int(metadata.get(ITPM_RESERVED_KEY, 0) or 0),
                int(metadata.get(MAX_TOKENS_RESERVED_KEY, 0) or 0),
                int(metadata.get(ESTIMATED_INPUT_KEY, 0) or 0),
            )
    return 0, 0, 0


async def _increment_with_rollback(
    dual_cache: DualCache,
    key: str,
    value: int,
    limit: Optional[int],
    *,
    parent_otel_span: Optional[Span] = None,
    limit_label: str,
) -> None:
    if value <= 0 or limit is None:
        return
    current = await dual_cache.async_increment_cache(
        key=key,
        value=value,
        ttl=RoutingArgsTTL,
        parent_otel_span=parent_otel_span,
    )
    if current is not None and current > limit:
        await dual_cache.async_increment_cache(
            key=key,
            value=-value,
            ttl=RoutingArgsTTL,
            parent_otel_span=parent_otel_span,
        )
        raise litellm.RateLimitError(
            message=f"Model rate limit exceeded. {limit_label} limit={limit}, current usage={current}",
            llm_provider="",
            model="",
            response=httpx.Response(
                status_code=429,
                content=(
                    f"{RouterErrors.user_defined_ratelimit_error.value} "
                    f"{limit_label} limit={limit}. current usage={current}."
                ),
                headers={"retry-after": str(RoutingArgsTTL)},
                request=httpx.Request(
                    method="io_token_rate_limit_check",
                    url="https://github.com/BerriAI/litellm",
                ),
            ),
            num_retries=0,
        )


async def async_io_token_pre_call_check(
    dual_cache: DualCache,
    deployment: Dict,
    parent_otel_span: Optional[Span] = None,
) -> Optional[Dict]:
    itpm_limit, otpm_limit = get_deployment_io_token_limits(deployment)
    if itpm_limit is None and otpm_limit is None:
        return deployment

    request_kwargs = get_io_token_rate_limit_request_kwargs()
    estimated_input = _estimate_input_tokens(request_kwargs)
    max_tokens = _resolve_max_tokens(request_kwargs, deployment)
    itpm_reserve = estimated_input + max_tokens

    dt = get_utc_datetime()
    current_minute = dt.strftime("%H-%M")
    itpm_key, otpm_key = _get_cache_keys(deployment, current_minute)

    if itpm_limit is not None and itpm_reserve > 0:
        await _increment_with_rollback(
            dual_cache,
            itpm_key,
            itpm_reserve,
            itpm_limit,
            parent_otel_span=parent_otel_span,
            limit_label="ITPM",
        )

    if otpm_limit is not None and max_tokens > 0:
        current_otpm = await dual_cache.async_get_cache(
            key=otpm_key, parent_otel_span=parent_otel_span
        )
        projected_otpm = (current_otpm or 0) + max_tokens
        if projected_otpm > otpm_limit:
            if itpm_limit is not None and itpm_reserve > 0:
                await dual_cache.async_increment_cache(
                    key=itpm_key,
                    value=-itpm_reserve,
                    ttl=RoutingArgsTTL,
                    parent_otel_span=parent_otel_span,
                )
            raise litellm.RateLimitError(
                message=(
                    f"Model rate limit exceeded. OTPM limit={otpm_limit}, current usage={current_otpm or 0}"
                ),
                llm_provider="",
                model=(deployment.get("litellm_params") or {}).get("model", ""),
                response=httpx.Response(
                    status_code=429,
                    content=(
                        f"{RouterErrors.user_defined_ratelimit_error.value} "
                        f"OTPM limit={otpm_limit}. current usage={current_otpm or 0}."
                    ),
                    headers={"retry-after": str(RoutingArgsTTL)},
                    request=httpx.Request(
                        method="io_token_rate_limit_check",
                        url="https://github.com/BerriAI/litellm",
                    ),
                ),
                num_retries=0,
            )

    _stash_reservation_in_metadata(
        request_kwargs,
        itpm_reserved=itpm_reserve,
        max_tokens=max_tokens,
        estimated_input=estimated_input,
    )
    return deployment


async def async_io_token_reconcile_success(
    dual_cache: DualCache,
    kwargs: Any,
    response_obj: Any,
    *,
    parent_otel_span: Optional[Span] = None,
) -> None:
    standard_logging_object: Optional[StandardLoggingPayload] = kwargs.get(
        "standard_logging_object"
    )
    if standard_logging_object is None:
        return
    model_id = standard_logging_object.get("model_id")
    if model_id is None:
        return
    model = (standard_logging_object.get("hidden_params") or {}).get(
        "litellm_model_name"
    )
    if not model:
        return

    itpm_reserved, max_tokens_reserved, _estimated_input = (
        _read_reservation_from_kwargs(kwargs)
    )
    usage = getattr(response_obj, "usage", None)
    prompt_tokens, completion_tokens, cached_tokens = _get_usage_tokens(usage)
    billable_input = max(0, prompt_tokens - cached_tokens)

    dt = get_utc_datetime()
    current_minute = dt.strftime("%H-%M")
    itpm_key = RouterCacheEnum.ITPM.value.format(
        id=model_id, model=model, current_minute=current_minute
    )
    otpm_key = RouterCacheEnum.OTPM.value.format(
        id=model_id, model=model, current_minute=current_minute
    )

    if itpm_reserved > 0:
        target_itpm = billable_input + completion_tokens
        itpm_delta = target_itpm - itpm_reserved
        if itpm_delta != 0:
            await dual_cache.async_increment_cache(
                key=itpm_key,
                value=itpm_delta,
                ttl=RoutingArgsTTL,
                parent_otel_span=parent_otel_span,
            )
    elif billable_input > 0 or completion_tokens > 0:
        await dual_cache.async_increment_cache(
            key=itpm_key,
            value=billable_input + completion_tokens,
            ttl=RoutingArgsTTL,
            parent_otel_span=parent_otel_span,
        )

    if completion_tokens > 0:
        await dual_cache.async_increment_cache(
            key=otpm_key,
            value=completion_tokens,
            ttl=RoutingArgsTTL,
            parent_otel_span=parent_otel_span,
        )

    verbose_router_logger.debug(
        f"[IO TOKEN LIMIT] model_id={model_id} itpm reconciled "
        f"(reserved={itpm_reserved}, billable_input={billable_input}, output={completion_tokens}, "
        f"max_tokens={max_tokens_reserved})"
    )


async def async_io_token_refund_failure(
    dual_cache: DualCache,
    kwargs: Any,
    *,
    parent_otel_span: Optional[Span] = None,
) -> None:
    itpm_reserved, max_tokens_reserved, _ = _read_reservation_from_kwargs(kwargs)
    if itpm_reserved <= 0:
        return
    standard_logging_object = kwargs.get("standard_logging_object") or {}
    model_id = standard_logging_object.get("model_id")
    model = (standard_logging_object.get("hidden_params") or {}).get(
        "litellm_model_name"
    )
    if not model_id or not model:
        return
    dt = get_utc_datetime()
    current_minute = dt.strftime("%H-%M")
    itpm_key = RouterCacheEnum.ITPM.value.format(
        id=model_id, model=model, current_minute=current_minute
    )
    await dual_cache.async_increment_cache(
        key=itpm_key,
        value=-itpm_reserved,
        ttl=RoutingArgsTTL,
        parent_otel_span=parent_otel_span,
    )
    verbose_router_logger.debug(
        f"[IO TOKEN LIMIT] refunded ITPM reservation={itpm_reserved} max_tokens={max_tokens_reserved}"
    )


def build_io_token_rate_limit_headers(
    *,
    itpm_limit: Optional[int],
    otpm_limit: Optional[int],
    current_itpm: Optional[int],
    current_otpm: Optional[int],
) -> Dict[str, int]:
    headers: Dict[str, int] = {}
    reset = seconds_until_minute_reset()
    if itpm_limit is not None:
        usage = current_itpm or 0
        headers["x-ratelimit-limit-input-tokens"] = itpm_limit
        headers["x-ratelimit-remaining-input-tokens"] = max(0, itpm_limit - usage)
        headers["x-ratelimit-reset-input-tokens"] = reset
    if otpm_limit is not None:
        usage = current_otpm or 0
        headers["x-ratelimit-limit-output-tokens"] = otpm_limit
        headers["x-ratelimit-remaining-output-tokens"] = max(0, otpm_limit - usage)
        headers["x-ratelimit-reset-output-tokens"] = reset
    return headers
