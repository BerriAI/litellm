"""
Separate ITPM/OTPM (input/output tokens per minute) deployment rate limits.

- Pre-call: atomically reserve estimated_input against ITPM and max_tokens against OTPM
- Post-call: reconcile ITPM to actual input tokens and OTPM to actual output tokens
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
from litellm.utils import get_utc_datetime

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span = Union[_Span, Any]
else:
    Span = Any

RoutingArgsTTL = 60

_io_token_rate_limit_request_kwargs: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    "io_token_rate_limit_request_kwargs",
    default=None,
)

ITPM_RESERVED_KEY = "_litellm_itpm_reserved"
OTPM_RESERVED_KEY = "_litellm_otpm_reserved"
ITPM_CACHE_KEY = "_litellm_itpm_cache_key"
OTPM_CACHE_KEY = "_litellm_otpm_cache_key"


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
    itpm_key = RouterCacheEnum.ITPM.value.format(id=model_id, model=deployment_name, current_minute=current_minute)
    otpm_key = RouterCacheEnum.OTPM.value.format(id=model_id, model=deployment_name, current_minute=current_minute)
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


def _resolve_max_tokens(request_kwargs: Optional[Dict[str, Any]], deployment: Dict) -> int:
    if request_kwargs:
        explicit = request_kwargs.get("max_tokens") or request_kwargs.get("max_completion_tokens")
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
        cached = int(details.get("cached_tokens", 0) or 0) if isinstance(details, dict) else 0
        return prompt, completion, cached
    return 0, 0, 0


def _stash_reservation_in_metadata(
    request_kwargs: Optional[Dict[str, Any]],
    *,
    itpm_reserved: int,
    otpm_reserved: int,
    itpm_cache_key: Optional[str],
    otpm_cache_key: Optional[str],
) -> None:
    if not request_kwargs:
        return
    reservation = {
        ITPM_RESERVED_KEY: itpm_reserved,
        OTPM_RESERVED_KEY: otpm_reserved,
        ITPM_CACHE_KEY: itpm_cache_key,
        OTPM_CACHE_KEY: otpm_cache_key,
    }
    for channel in ("metadata", "litellm_metadata"):
        existing = request_kwargs.get(channel)
        if isinstance(existing, dict):
            existing.update(reservation)
        elif channel == "metadata":
            request_kwargs[channel] = dict(reservation)


def _extract_reservation(reservation: Dict[str, Any]) -> Tuple[int, int, Optional[str], Optional[str]]:
    itpm_cache_key = reservation.get(ITPM_CACHE_KEY)
    otpm_cache_key = reservation.get(OTPM_CACHE_KEY)
    return (
        int(reservation.get(ITPM_RESERVED_KEY, 0) or 0),
        int(reservation.get(OTPM_RESERVED_KEY, 0) or 0),
        itpm_cache_key if isinstance(itpm_cache_key, str) else None,
        otpm_cache_key if isinstance(otpm_cache_key, str) else None,
    )


def _read_reservation_from_kwargs(kwargs: Any) -> Tuple[int, int, Optional[str], Optional[str]]:
    for channel in ("metadata", "litellm_metadata"):
        channel_dict = None
        if isinstance(kwargs, dict):
            channel_dict = kwargs.get(channel)
            if not isinstance(channel_dict, dict):
                litellm_params = kwargs.get("litellm_params")
                if isinstance(litellm_params, dict) and isinstance(litellm_params.get("metadata"), dict):
                    channel_dict = litellm_params.get("metadata")
        if isinstance(channel_dict, dict) and ITPM_RESERVED_KEY in channel_dict:
            return _extract_reservation(channel_dict)
    standard_logging_object = kwargs.get("standard_logging_object") if isinstance(kwargs, dict) else None
    if isinstance(standard_logging_object, dict):
        metadata = standard_logging_object.get("metadata") or {}
        if ITPM_RESERVED_KEY in metadata:
            return _extract_reservation(metadata)
    return 0, 0, None, None


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

    dt = get_utc_datetime()
    current_minute = dt.strftime("%H-%M")
    itpm_key, otpm_key = _get_cache_keys(deployment, current_minute)

    itpm_reserved = 0
    otpm_reserved = 0

    if itpm_limit is not None and estimated_input > 0:
        await _increment_with_rollback(
            dual_cache,
            itpm_key,
            estimated_input,
            itpm_limit,
            parent_otel_span=parent_otel_span,
            limit_label="ITPM",
        )
        itpm_reserved = estimated_input

    if otpm_limit is not None and max_tokens > 0:
        try:
            await _increment_with_rollback(
                dual_cache,
                otpm_key,
                max_tokens,
                otpm_limit,
                parent_otel_span=parent_otel_span,
                limit_label="OTPM",
            )
        except litellm.RateLimitError:
            if itpm_reserved > 0:
                await dual_cache.async_increment_cache(
                    key=itpm_key,
                    value=-itpm_reserved,
                    ttl=RoutingArgsTTL,
                    parent_otel_span=parent_otel_span,
                )
            raise
        otpm_reserved = max_tokens

    _stash_reservation_in_metadata(
        request_kwargs,
        itpm_reserved=itpm_reserved,
        otpm_reserved=otpm_reserved,
        itpm_cache_key=itpm_key if itpm_limit is not None else None,
        otpm_cache_key=otpm_key if otpm_limit is not None else None,
    )
    return deployment


async def async_io_token_reconcile_success(
    dual_cache: DualCache,
    kwargs: Any,
    response_obj: Any,
    *,
    parent_otel_span: Optional[Span] = None,
) -> None:
    itpm_reserved, otpm_reserved, itpm_key, otpm_key = _read_reservation_from_kwargs(kwargs)
    if itpm_key is None and otpm_key is None:
        return

    usage = getattr(response_obj, "usage", None)
    prompt_tokens, completion_tokens, cached_tokens = _get_usage_tokens(usage)
    billable_input = max(0, prompt_tokens - cached_tokens)

    # Reconcile against the exact key that held the reservation (which encodes
    # the reservation's minute), not a key recomputed at response time. This
    # tracks actual usage even when the pre-call estimate was 0, and avoids a
    # minute-boundary mismatch for calls that span into the next minute.
    if itpm_key is not None:
        itpm_delta = billable_input - itpm_reserved
        if itpm_delta != 0:
            await dual_cache.async_increment_cache(
                key=itpm_key,
                value=itpm_delta,
                ttl=RoutingArgsTTL,
                parent_otel_span=parent_otel_span,
            )

    if otpm_key is not None:
        otpm_delta = completion_tokens - otpm_reserved
        if otpm_delta != 0:
            await dual_cache.async_increment_cache(
                key=otpm_key,
                value=otpm_delta,
                ttl=RoutingArgsTTL,
                parent_otel_span=parent_otel_span,
            )

    verbose_router_logger.debug(
        f"[IO TOKEN LIMIT] reconciled "
        f"(itpm_reserved={itpm_reserved}, billable_input={billable_input}, "
        f"otpm_reserved={otpm_reserved}, output={completion_tokens})"
    )


async def async_io_token_refund_failure(
    dual_cache: DualCache,
    kwargs: Any,
    *,
    parent_otel_span: Optional[Span] = None,
) -> None:
    itpm_reserved, otpm_reserved, itpm_key, otpm_key = _read_reservation_from_kwargs(kwargs)
    if itpm_key is not None and itpm_reserved > 0:
        await dual_cache.async_increment_cache(
            key=itpm_key,
            value=-itpm_reserved,
            ttl=RoutingArgsTTL,
            parent_otel_span=parent_otel_span,
        )
    if otpm_key is not None and otpm_reserved > 0:
        await dual_cache.async_increment_cache(
            key=otpm_key,
            value=-otpm_reserved,
            ttl=RoutingArgsTTL,
            parent_otel_span=parent_otel_span,
        )
    verbose_router_logger.debug(f"[IO TOKEN LIMIT] refunded ITPM={itpm_reserved} OTPM={otpm_reserved}")


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
