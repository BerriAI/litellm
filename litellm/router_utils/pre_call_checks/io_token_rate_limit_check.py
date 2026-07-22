"""
Separate ITPM/OTPM (input/output tokens per minute) deployment rate limits.

- Pre-call: atomically reserve estimated_input against ITPM and max_tokens against OTPM
- Post-call: reconcile ITPM to actual input tokens and OTPM to actual output tokens
- Cached prompt-read tokens are excluded from ITPM post-call accounting

Used by ModelRateLimitingCheck when a deployment sets itpm/otpm.
"""

from __future__ import annotations

import contextlib
import contextvars
from typing import TYPE_CHECKING, Any, Optional

import httpx

import litellm
from litellm import token_counter
from litellm._logging import verbose_router_logger
from litellm.caching.dual_cache import DualCache
from litellm.types.router import RouterCacheEnum, RouterErrors
from litellm.utils import get_utc_datetime

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span = _Span | Any
else:
    Span = Any

RoutingArgsTTL = 60

_io_token_rate_limit_request_kwargs: contextvars.ContextVar[Optional[dict[str, Any]]] = contextvars.ContextVar(
    "io_token_rate_limit_request_kwargs",
    default=None,
)

ITPM_RESERVED_KEY = "_litellm_itpm_reserved"
OTPM_RESERVED_KEY = "_litellm_otpm_reserved"
ITPM_CACHE_KEY = "_litellm_itpm_cache_key"
OTPM_CACHE_KEY = "_litellm_otpm_cache_key"


def set_io_token_rate_limit_request_kwargs(kwargs: Optional[dict[str, Any]], store_in_context: bool = True) -> None:
    # The reservation sentinels are server-only, but `metadata` is caller
    # controlled on proxy requests. Strip any client-supplied copies here (this
    # runs before the router stashes its own reservation) so a forged
    # reservation can't drive the post-call reconcile/refund against an
    # arbitrary counter and bypass the configured limits.
    _clear_reservation_from_kwargs(kwargs)
    # The context slot pins the entire request kwargs (messages included) for
    # the lifetime of the surrounding context, which outlives the request when
    # the context is captured by pooled resources (e.g. a redis connection
    # created mid-request). Only ITPM/OTPM-limited deployments read it, so for
    # every other deployment overwrite the slot with None instead of the
    # kwargs; overwriting (rather than skipping) also releases a previous
    # request's kwargs when a context is reused.
    _io_token_rate_limit_request_kwargs.set(kwargs if store_in_context else None)


def get_io_token_rate_limit_request_kwargs() -> Optional[dict[str, Any]]:
    return _io_token_rate_limit_request_kwargs.get()


def seconds_until_minute_reset() -> int:
    dt = get_utc_datetime()
    return max(1, 60 - dt.second)


def get_deployment_io_token_limits(
    deployment: dict,
) -> tuple[Optional[int], Optional[int]]:
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


def deployment_has_io_token_limits(deployment: dict) -> bool:
    itpm, otpm = get_deployment_io_token_limits(deployment)
    return itpm is not None or otpm is not None


def _get_cache_keys(deployment: dict, current_minute: str) -> Optional[tuple[str, str]]:
    model_id = deployment.get("model_info", {}).get("id")
    deployment_name = deployment.get("litellm_params", {}).get("model")
    # Without both a deployment id and model name the key would collapse to a
    # shared "None:None" bucket across misconfigured deployments, so bail out.
    if model_id is None or deployment_name is None:
        return None
    itpm_key = RouterCacheEnum.ITPM.value.format(id=model_id, model=deployment_name, current_minute=current_minute)
    otpm_key = RouterCacheEnum.OTPM.value.format(id=model_id, model=deployment_name, current_minute=current_minute)
    return itpm_key, otpm_key


def _estimate_input_tokens(request_kwargs: Optional[dict[str, Any]], model: str = "") -> int:
    if not request_kwargs:
        return 0
    messages = request_kwargs.get("messages")
    prompt = request_kwargs.get("prompt")
    input_text = request_kwargs.get("input")
    # token_counter can raise from any of its tokenizer backends; this is a
    # best-effort estimate for the ITPM reservation and must never fail the
    # underlying request. Passing the deployment model name uses a model-specific
    # tokenizer when available, reducing the reservation over/under-estimate window
    # between pre-call and post-call reconcile.
    with contextlib.suppress(Exception):
        return max(0, int(token_counter(model=model, messages=messages, text=prompt or input_text)))
    return 0


def _model_max_output_tokens(model_name: str) -> Optional[int]:
    # litellm.get_model_info raises a bare Exception for an unrecognized model;
    # this lookup is a fallback default and must never fail the request.
    with contextlib.suppress(Exception):
        info = litellm.get_model_info(model=model_name)
        model_max = info.get("max_output_tokens") or info.get("max_tokens")
        if model_max is not None:
            return max(0, int(model_max))
    return None


def _resolve_max_tokens(request_kwargs: Optional[dict[str, Any]], deployment: dict) -> int:
    if request_kwargs:
        # An explicit max_tokens=0 must be honored, not treated as absent and
        # replaced by the model default.
        explicit_caps = [
            int(value)
            for value in (
                request_kwargs.get("max_tokens"),
                request_kwargs.get("max_completion_tokens"),
                request_kwargs.get("max_output_tokens"),
            )
            if value is not None
        ]
        if explicit_caps:
            return max(0, max(explicit_caps))

    model_name = (deployment.get("litellm_params") or {}).get("model")
    if model_name:
        model_max = _model_max_output_tokens(model_name)
        if model_max is not None:
            return model_max
    return 4096


def _get_usage_tokens(usage: Any) -> tuple[int, int, int]:
    if usage is None:
        return 0, 0, 0
    if hasattr(usage, "prompt_tokens") or hasattr(usage, "input_tokens"):
        prompt = int(getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", 0) or 0)
        cached = 0
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            cached = int(getattr(details, "cached_tokens", 0) or 0)
        if not cached:
            cached = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        return prompt, completion, cached
    if isinstance(usage, dict):
        prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        details = usage.get("prompt_tokens_details") or {}
        cached = int(details.get("cached_tokens", 0) or 0) if isinstance(details, dict) else 0
        if not cached:
            cached = int(usage.get("cache_read_input_tokens") or 0)
        return prompt, completion, cached
    return 0, 0, 0


def _extract_response_usage(response_obj: Any) -> Any:
    if isinstance(response_obj, dict):
        return response_obj.get("usage")
    return getattr(response_obj, "usage", None)


def _usage_is_present(usage: Any) -> bool:
    """
    True only if usage carries an actual input/output breakdown.

    ``total_tokens`` alone is deliberately excluded: ``_get_usage_tokens`` has
    no way to split a bare total into input vs. output, so treating it as
    "present" would resolve to (0, 0) and refund the full reservation as if
    zero tokens were used.
    """
    if usage is None:
        return False
    fields = ("prompt_tokens", "completion_tokens", "input_tokens", "output_tokens")
    if isinstance(usage, dict):
        return any(key in usage for key in fields)
    return any(hasattr(usage, key) for key in fields)


def _resolve_reconcile_usage_tokens(
    kwargs: Any,
    response_obj: Any,
) -> tuple[int, int, bool]:
    """
    Resolve billable input and output tokens for post-call reconcile.

    Prefer the response usage object; fall back to standard_logging_object token
    fields. When usage cannot be resolved, return ``usage_resolved=False`` so
    callers keep the pre-call reservation instead of refunding it as zero usage.
    """
    usage = _extract_response_usage(response_obj)
    if _usage_is_present(usage):
        prompt_tokens, completion_tokens, cached_tokens = _get_usage_tokens(usage)
        return max(0, prompt_tokens - cached_tokens), completion_tokens, True

    if isinstance(kwargs, dict):
        standard_logging_object = kwargs.get("standard_logging_object")
        if isinstance(standard_logging_object, dict):
            prompt_tokens = int(standard_logging_object.get("prompt_tokens") or 0)
            completion_tokens = int(standard_logging_object.get("completion_tokens") or 0)
            cached_tokens = 0
            metadata = standard_logging_object.get("metadata")
            if isinstance(metadata, dict):
                cached_tokens = int(metadata.get("cache_read_input_tokens") or 0)
            # Same rationale as _usage_is_present: a bare total_tokens with no
            # prompt/completion breakdown can't be split, so it isn't treated
            # as resolved usage - the reservation is kept instead of refunded.
            if prompt_tokens or completion_tokens:
                return max(0, prompt_tokens - cached_tokens), completion_tokens, True

    return 0, 0, False


def _stash_reservation_in_metadata(
    request_kwargs: Optional[dict[str, Any]],
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


def _extract_reservation(reservation: dict[str, Any]) -> tuple[int, int, Optional[str], Optional[str]]:
    itpm_cache_key = reservation.get(ITPM_CACHE_KEY)
    otpm_cache_key = reservation.get(OTPM_CACHE_KEY)
    return (
        int(reservation.get(ITPM_RESERVED_KEY, 0) or 0),
        int(reservation.get(OTPM_RESERVED_KEY, 0) or 0),
        itpm_cache_key if isinstance(itpm_cache_key, str) else None,
        otpm_cache_key if isinstance(otpm_cache_key, str) else None,
    )


def _reservation_channels(kwargs: Any) -> tuple[Any, ...]:
    """
    Places a reservation may live, in priority order: the top-level metadata
    channels win over litellm_params.metadata (so a top-level stash is never
    shadowed), which win over the standard_logging_object copy.
    """
    if not isinstance(kwargs, dict):
        return ()
    channels = [kwargs.get("metadata"), kwargs.get("litellm_metadata")]
    litellm_params = kwargs.get("litellm_params")
    if isinstance(litellm_params, dict):
        channels.append(litellm_params.get("metadata"))
    standard_logging_object = kwargs.get("standard_logging_object")
    if isinstance(standard_logging_object, dict):
        channels.append(standard_logging_object.get("metadata"))
    return tuple(channels)


def _read_reservation_from_kwargs(kwargs: Any) -> tuple[int, int, Optional[str], Optional[str]]:
    for channel_dict in _reservation_channels(kwargs):
        if isinstance(channel_dict, dict) and ITPM_RESERVED_KEY in channel_dict:
            return _extract_reservation(channel_dict)
    return 0, 0, None, None


def _clear_reservation_from_kwargs(kwargs: Any) -> None:
    """
    Remove the stashed reservation so a retry on a different (e.g. non-IO)
    deployment does not re-process the already-reconciled/refunded reservation.
    """
    for channel_dict in _reservation_channels(kwargs):
        if isinstance(channel_dict, dict):
            for key in (ITPM_RESERVED_KEY, OTPM_RESERVED_KEY, ITPM_CACHE_KEY, OTPM_CACHE_KEY):
                channel_dict.pop(key, None)


def _reservation_value(value: int, limit: Optional[int]) -> int:
    if limit is None:
        return 0
    if value > 0:
        return value
    # Estimation failed (empty messages, unsupported model, tokenizer error).
    # Reserve a minimal 1-token slot rather than the full limit: the latter
    # would let one request whose estimate failed fill the entire bucket,
    # serializing every concurrent request to the deployment until it
    # completes and reconciles against actual usage.
    return 1


def _rate_limit_error(limit_label: str, limit: int, current: float) -> litellm.RateLimitError:
    return litellm.RateLimitError(
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


def _sync_increment_with_rollback(
    dual_cache: DualCache,
    key: str,
    value: int,
    limit: Optional[int],
    *,
    limit_label: str,
) -> None:
    if value <= 0 or limit is None:
        return
    current = dual_cache.increment_cache(
        key=key,
        value=value,
        ttl=RoutingArgsTTL,
    )
    if current is not None and current > limit:
        dual_cache.increment_cache(
            key=key,
            value=-value,
            ttl=RoutingArgsTTL,
        )
        raise _rate_limit_error(limit_label, limit, current)


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
        raise _rate_limit_error(limit_label, limit, current)


def io_token_pre_call_check(
    dual_cache: DualCache,
    deployment: dict,
) -> Optional[dict]:
    itpm_limit, otpm_limit = get_deployment_io_token_limits(deployment)
    if itpm_limit is None and otpm_limit is None:
        return deployment

    request_kwargs = get_io_token_rate_limit_request_kwargs()
    _model = (deployment.get("litellm_params") or {}).get("model") or ""
    estimated_input = _estimate_input_tokens(request_kwargs, model=_model)
    max_tokens = _resolve_max_tokens(request_kwargs, deployment)

    dt = get_utc_datetime()
    current_minute = dt.strftime("%H-%M")
    cache_keys = _get_cache_keys(deployment, current_minute)
    if cache_keys is None:
        return deployment
    itpm_key, otpm_key = cache_keys

    itpm_reserved = 0
    otpm_reserved = 0

    if itpm_limit is not None:
        itpm_reserved = _reservation_value(estimated_input, itpm_limit)
        _sync_increment_with_rollback(
            dual_cache,
            itpm_key,
            itpm_reserved,
            itpm_limit,
            limit_label="ITPM",
        )

    if otpm_limit is not None:
        otpm_reserved = 0 if max_tokens == 0 else _reservation_value(max_tokens, otpm_limit)
        try:
            _sync_increment_with_rollback(
                dual_cache,
                otpm_key,
                otpm_reserved,
                otpm_limit,
                limit_label="OTPM",
            )
        except Exception:
            if itpm_reserved > 0:
                dual_cache.increment_cache(
                    key=itpm_key,
                    value=-itpm_reserved,
                    ttl=RoutingArgsTTL,
                )
            raise

    _stash_reservation_in_metadata(
        request_kwargs,
        itpm_reserved=itpm_reserved,
        otpm_reserved=otpm_reserved,
        itpm_cache_key=itpm_key if itpm_limit is not None else None,
        otpm_cache_key=otpm_key if otpm_limit is not None else None,
    )
    return deployment


async def async_io_token_pre_call_check(
    dual_cache: DualCache,
    deployment: dict,
    parent_otel_span: Optional[Span] = None,
) -> Optional[dict]:
    itpm_limit, otpm_limit = get_deployment_io_token_limits(deployment)
    if itpm_limit is None and otpm_limit is None:
        return deployment

    request_kwargs = get_io_token_rate_limit_request_kwargs()
    _model = (deployment.get("litellm_params") or {}).get("model") or ""
    estimated_input = _estimate_input_tokens(request_kwargs, model=_model)
    max_tokens = _resolve_max_tokens(request_kwargs, deployment)

    dt = get_utc_datetime()
    current_minute = dt.strftime("%H-%M")
    cache_keys = _get_cache_keys(deployment, current_minute)
    if cache_keys is None:
        return deployment
    itpm_key, otpm_key = cache_keys

    itpm_reserved = 0
    otpm_reserved = 0

    if itpm_limit is not None:
        itpm_reserved = _reservation_value(estimated_input, itpm_limit)
        await _increment_with_rollback(
            dual_cache,
            itpm_key,
            itpm_reserved,
            itpm_limit,
            parent_otel_span=parent_otel_span,
            limit_label="ITPM",
        )

    if otpm_limit is not None:
        otpm_reserved = 0 if max_tokens == 0 else _reservation_value(max_tokens, otpm_limit)
        try:
            await _increment_with_rollback(
                dual_cache,
                otpm_key,
                otpm_reserved,
                otpm_limit,
                parent_otel_span=parent_otel_span,
                limit_label="OTPM",
            )
        except Exception:
            # Any failure reserving OTPM (a 429 or a transient cache error) must
            # release the ITPM reservation already made, or it stays inflated
            # until the TTL expires.
            if itpm_reserved > 0:
                await dual_cache.async_increment_cache(
                    key=itpm_key,
                    value=-itpm_reserved,
                    ttl=RoutingArgsTTL,
                    parent_otel_span=parent_otel_span,
                )
            raise

    _stash_reservation_in_metadata(
        request_kwargs,
        itpm_reserved=itpm_reserved,
        otpm_reserved=otpm_reserved,
        itpm_cache_key=itpm_key if itpm_limit is not None else None,
        otpm_cache_key=otpm_key if otpm_limit is not None else None,
    )
    return deployment


def io_token_reconcile_success(
    dual_cache: DualCache,
    kwargs: Any,
    response_obj: Any,
) -> None:
    itpm_reserved, otpm_reserved, itpm_key, otpm_key = _read_reservation_from_kwargs(kwargs)
    if itpm_key is None and otpm_key is None:
        return

    billable_input, completion_tokens, usage_resolved = _resolve_reconcile_usage_tokens(kwargs, response_obj)

    try:
        if usage_resolved:
            if itpm_key is not None:
                itpm_delta = billable_input - itpm_reserved
                if itpm_delta != 0:
                    dual_cache.increment_cache(
                        key=itpm_key,
                        value=itpm_delta,
                        ttl=RoutingArgsTTL,
                    )

            if otpm_key is not None:
                otpm_delta = completion_tokens - otpm_reserved
                if otpm_delta != 0:
                    dual_cache.increment_cache(
                        key=otpm_key,
                        value=otpm_delta,
                        ttl=RoutingArgsTTL,
                    )
        else:
            verbose_router_logger.debug(
                "[IO TOKEN LIMIT] usage missing; keeping reservation "
                f"(itpm_reserved={itpm_reserved}, otpm_reserved={otpm_reserved})"
            )
    finally:
        _clear_reservation_from_kwargs(kwargs)

    verbose_router_logger.debug(
        f"[IO TOKEN LIMIT] reconciled "
        f"(usage_resolved={usage_resolved}, itpm_reserved={itpm_reserved}, "
        f"billable_input={billable_input}, otpm_reserved={otpm_reserved}, output={completion_tokens})"
    )


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

    billable_input, completion_tokens, usage_resolved = _resolve_reconcile_usage_tokens(kwargs, response_obj)

    # Reconcile against the exact key that held the reservation (which encodes
    # the reservation's minute), not a key recomputed at response time. This
    # tracks actual usage even when the pre-call estimate was 0, and avoids a
    # minute-boundary mismatch for calls that span into the next minute. Always
    # clear the stash afterwards (even if an increment throws) so a retry or a
    # duplicate success event can't re-process it.
    try:
        if usage_resolved:
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
        else:
            verbose_router_logger.debug(
                "[IO TOKEN LIMIT] usage missing; keeping reservation "
                f"(itpm_reserved={itpm_reserved}, otpm_reserved={otpm_reserved})"
            )
    finally:
        _clear_reservation_from_kwargs(kwargs)

    verbose_router_logger.debug(
        f"[IO TOKEN LIMIT] reconciled "
        f"(usage_resolved={usage_resolved}, itpm_reserved={itpm_reserved}, "
        f"billable_input={billable_input}, otpm_reserved={otpm_reserved}, output={completion_tokens})"
    )


def io_token_refund_failure(
    dual_cache: DualCache,
    kwargs: Any,
) -> None:
    itpm_reserved, otpm_reserved, itpm_key, otpm_key = _read_reservation_from_kwargs(kwargs)
    if itpm_key is None and otpm_key is None:
        return
    if itpm_key is not None and itpm_reserved > 0:
        dual_cache.increment_cache(
            key=itpm_key,
            value=-itpm_reserved,
            ttl=RoutingArgsTTL,
        )
    if otpm_key is not None and otpm_reserved > 0:
        dual_cache.increment_cache(
            key=otpm_key,
            value=-otpm_reserved,
            ttl=RoutingArgsTTL,
        )
    _clear_reservation_from_kwargs(kwargs)
    verbose_router_logger.debug(f"[IO TOKEN LIMIT] refunded ITPM={itpm_reserved} OTPM={otpm_reserved}")


def refund_stale_reservation_before_retry(dual_cache: DualCache, kwargs: Optional[dict[str, Any]]) -> None:
    """
    Synchronously refund and clear any reservation a previous deployment
    attempt stashed in ``kwargs``, before it's overwritten for the next
    attempt (retry/fallback).

    ``set_io_token_rate_limit_request_kwargs`` strips reservation sentinels
    from ``kwargs`` on every deployment pick (a security measure so a
    caller-forged reservation can't be replayed). Without this refund, a
    retry after a non-RateLimitError failure (e.g. an upstream 500) would
    wipe deployment A's still-unreconciled reservation before its failure
    event - which may be scheduled as a background task - gets a chance to
    refund it, permanently stranding the reservation until its TTL expires
    and causing false rate-limit errors for subsequent requests.

    ponytail: uses sync ``DualCache.increment_cache`` which issues a blocking
    Redis INCR when a Redis backend is configured. This only triggers on
    streaming mid-stream retries (non-streaming failures await their failure
    handler before retrying, so the sentinels are already cleared). Upgrade
    path: make ``_update_kwargs_with_deployment`` async and switch to
    ``async_io_token_refund_failure`` — requires touching all callers.
    """
    if not kwargs:
        return
    io_token_refund_failure(dual_cache, kwargs)


async def async_io_token_refund_failure(
    dual_cache: DualCache,
    kwargs: Any,
    *,
    parent_otel_span: Optional[Span] = None,
) -> None:
    itpm_reserved, otpm_reserved, itpm_key, otpm_key = _read_reservation_from_kwargs(kwargs)
    if itpm_key is None and otpm_key is None:
        return
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
    _clear_reservation_from_kwargs(kwargs)
    verbose_router_logger.debug(f"[IO TOKEN LIMIT] refunded ITPM={itpm_reserved} OTPM={otpm_reserved}")


def build_io_token_rate_limit_headers(
    *,
    itpm_limit: Optional[int],
    otpm_limit: Optional[int],
    current_itpm: Optional[int],
    current_otpm: Optional[int],
) -> dict[str, int]:
    headers: dict[str, int] = {}
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
