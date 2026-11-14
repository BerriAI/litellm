"""
Separate ITPM/OTPM (input/output tokens per minute) deployment rate limits.

- Pre-call: atomically reserve estimated_input against ITPM and max_tokens against OTPM; a request
  without max_tokens reserves the model's max output, capped at a fraction of the OTPM limit
- Post-call: reconcile ITPM to actual input tokens and OTPM to actual output tokens
- Cached prompt-read tokens are excluded from ITPM post-call accounting

Used by ModelRateLimitingCheck when a deployment sets itpm/otpm.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from types import MappingProxyType
from typing import Any, Final

import httpx
from pydantic import TypeAdapter, ValidationError

import litellm
from litellm import token_counter
from litellm._internal_context import with_service_target
from litellm._logging import verbose_router_logger
from litellm._uuid import uuid
from litellm.constants import NO_MAX_TOKENS_TPM_RESERVATION_FRACTION
from litellm.litellm_core_utils.attempt_settlement import current_attempt_id, release_in_background, released
from litellm.litellm_core_utils.core_helpers import coerce_token_limit
from litellm.litellm_core_utils.internal_call_metadata import (
    IO_TOKEN_ATTEMPT_KEY,
    IO_TOKEN_RESERVATION_ID_KEY,
    IO_TOKEN_RESERVATION_KEYS,
    ITPM_COUNTER_KEY,
    ITPM_RESERVED_KEY,
    OTPM_COUNTER_KEY,
    OTPM_RESERVED_KEY,
)
from litellm.litellm_core_utils.token_counter import deployment_openai_model, offload_token_count
from litellm.llms.azure.ptu_capacity import is_str_keyed_mapping
from litellm.redis.stores.router_usage import MAX_INCREMENT, ROUTER_USAGE_TARGET, RouterUsage, UsageCounter
from litellm.types.router import RouterErrors
from litellm.utils import get_utc_datetime

_MODEL_INFO_ADAPTER: Final[TypeAdapter[Mapping[str, object]]] = TypeAdapter(Mapping[str, object])
_STASHED_COUNTER: Final[TypeAdapter[UsageCounter]] = TypeAdapter(UsageCounter)

RoutingArgsTTL: Final = 60


class _SettledReservations:
    """The ids of reservations settled within the counters' window, so a copy of a stash that outlives
    the dict it was claimed from (the logging object copies litellm_metadata) is never settled twice.
    Ids leave only once their window has passed, never to make room."""

    def __init__(self, window: float, clock: Callable[[], float] = time.monotonic) -> None:
        self._window: Final = window
        self._clock: Final = clock
        self._expiries: Final[OrderedDict[str, float]] = OrderedDict()  # mutable-ok: pruned oldest first as ids expire
        self._lock: Final = threading.Lock()

    def claim(self, reservation_id: str) -> bool:
        with self._lock:
            now: Final = self._clock()
            while self._expiries and next(iter(self._expiries.values())) <= now:
                _ = self._expiries.popitem(last=False)
            if reservation_id in self._expiries:
                return False
            self._expiries[reservation_id] = now + self._window
            return True


_SETTLED_RESERVATIONS: Final = _SettledReservations(window=RoutingArgsTTL * 2)


@dataclass(frozen=True, slots=True)
class _IoTokenRateLimitRequest:
    kwargs: dict[str, object] | None
    opens_responses_socket: bool
    metadata_channel: str


_io_token_rate_limit_request: Final[contextvars.ContextVar[_IoTokenRateLimitRequest | None]] = contextvars.ContextVar(
    "io_token_rate_limit_request",
    default=None,
)


def set_io_token_rate_limit_request_kwargs(
    kwargs: dict[str, object] | None,
    store_in_context: bool = True,
    *,
    opens_responses_socket: bool = False,
    metadata_channel: str,
) -> None:
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
    _io_token_rate_limit_request.set(
        _IoTokenRateLimitRequest(kwargs, opens_responses_socket, metadata_channel) if store_in_context else None
    )


def get_io_token_rate_limit_request_kwargs() -> dict[str, object] | None:
    request: Final = _io_token_rate_limit_request.get()
    return None if request is None else request.kwargs


def seconds_until_minute_reset() -> int:
    dt: Final = get_utc_datetime()
    return max(1, 60 - dt.second)


def get_deployment_io_token_limits(
    deployment: dict,
) -> tuple[int | None, int | None]:
    itpm = deployment.get("itpm")
    otpm = deployment.get("otpm")
    litellm_params: Final = deployment.get("litellm_params") or {}
    model_info: Final = deployment.get("model_info") or {}
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


_IO_TOKEN_LIMIT_KEYS: Final = frozenset({"itpm", "otpm"})
_IO_TOKEN_LIMIT_SECTIONS: Final = frozenset({"litellm_params", "model_info"})


def without_io_token_limits(deployment: Mapping[str, object]) -> Mapping[str, object]:
    return {
        key: _without_io_token_limit_keys(value) if key in _IO_TOKEN_LIMIT_SECTIONS else value
        for key, value in deployment.items()
        if key not in _IO_TOKEN_LIMIT_KEYS
    }


def _without_io_token_limit_keys(section: object) -> object:
    if not is_str_keyed_mapping(section):
        return section
    return {key: value for key, value in section.items() if key not in _IO_TOKEN_LIMIT_KEYS}


def _get_counters(deployment: dict, current_minute: str) -> tuple[UsageCounter, UsageCounter] | None:
    model_id: Final[object] = deployment.get("model_info", {}).get("id")
    deployment_name: Final[object] = deployment.get("litellm_params", {}).get("model")
    # Without both a deployment id and model name the key would collapse to a
    # shared "None:None" bucket across misconfigured deployments, so bail out.
    if model_id is None or deployment_name is None:
        return None
    return (
        UsageCounter("global_router", f"{model_id}", f"{deployment_name}", current_minute, "itpm"),
        UsageCounter("global_router", f"{model_id}", f"{deployment_name}", current_minute, "otpm"),
    )


def _served_model(deployment: object) -> str:
    params: Final = deployment.get("litellm_params") if is_str_keyed_mapping(deployment) else None
    served: Final = params.get("model") if is_str_keyed_mapping(params) else None
    return served if isinstance(served, str) else ""


def _tokenizer_and_served_models(deployment: object) -> tuple[str, str]:
    served: Final = _served_model(deployment)
    return deployment_openai_model(deployment) or served, served


def _estimate_input_tokens(request_kwargs: dict[str, Any] | None, model: str = "") -> int:
    if not request_kwargs:
        return 0
    messages: Final = request_kwargs.get("messages")
    prompt: Final = request_kwargs.get("prompt")
    input_text: Final = request_kwargs.get("input")
    # token_counter can raise from any of its tokenizer backends; this is a
    # best-effort estimate for the ITPM reservation and must never fail the
    # underlying request. Passing the deployment model name uses a model-specific
    # tokenizer when available, reducing the reservation over/under-estimate window
    # between pre-call and post-call reconcile.
    with contextlib.suppress(Exception):
        return max(0, int(token_counter(model=model, messages=messages, text=prompt or input_text)))
    return 0


def _model_max_output_tokens(model_name: str) -> int | None:
    # litellm.get_model_info raises a bare Exception for an unrecognized model;
    # this lookup is a fallback default and must never fail the request.
    with contextlib.suppress(Exception):
        info: Final = litellm.get_model_info(model=model_name)
        model_max: Final = info.get("max_output_tokens")
        if model_max is not None:
            return max(0, int(model_max))
    return None


_CHAT_OUTPUT_CAP_KEYS: Final = ("max_tokens", "max_completion_tokens")
_RESPONSES_OUTPUT_CAP_KEYS: Final = ("max_output_tokens",)
_OUTPUT_CAP_KEYS: Final = _CHAT_OUTPUT_CAP_KEYS + _RESPONSES_OUTPUT_CAP_KEYS


def _explicit_max_tokens(
    request_kwargs: Mapping[str, object] | None, cap_keys: tuple[str, ...] = _OUTPUT_CAP_KEYS
) -> int | None:
    if not request_kwargs:
        return None
    explicit: Final = next(
        (request_kwargs[key] for key in cap_keys if request_kwargs.get(key) is not None),
        None,
    )
    match explicit:
        case int() | float() | str():
            return max(0, int(explicit))
        case _:
            return None


def _sent_deployment_cap_keys(request_kwargs: Mapping[str, object] | None) -> tuple[str, ...]:
    is_responses_request: Final = (
        request_kwargs is not None and "input" in request_kwargs and "messages" not in request_kwargs
    )
    return _RESPONSES_OUTPUT_CAP_KEYS if is_responses_request else _CHAT_OUTPUT_CAP_KEYS


def _deployment_output_cap(request_kwargs: Mapping[str, object] | None, deployment: Mapping[str, object]) -> int | None:
    params: Final = deployment.get("litellm_params")
    return (
        _explicit_max_tokens(params, _sent_deployment_cap_keys(request_kwargs))
        if is_str_keyed_mapping(params)
        else None
    )


def _default_otpm_reservation(deployment: Mapping[str, object], model_name: str, otpm_limit: int) -> int:
    model_info: Final[object] = deployment.get("model_info")
    configured_model_max: Final = (
        coerce_token_limit(_MODEL_INFO_ADAPTER.validate_python(model_info).get("max_output_tokens"))
        if isinstance(model_info, Mapping)
        else None
    )
    model_max: Final = (
        max(0, configured_model_max)
        if configured_model_max is not None
        else _model_max_output_tokens(model_name)
        if model_name
        else None
    )
    share_of_limit: Final = max(1, int(otpm_limit * NO_MAX_TOKENS_TPM_RESERVATION_FRACTION))
    return min(4096 if model_max is None else model_max, share_of_limit)


def _otpm_reservation(
    request_kwargs: Mapping[str, object] | None,
    deployment: Mapping[str, object],
    model_name: str,
    otpm_limit: int,
) -> int:
    from_request: Final = _explicit_max_tokens(request_kwargs)
    if from_request is not None:
        return from_request
    default: Final = _default_otpm_reservation(deployment, model_name, otpm_limit)
    from_deployment: Final = _deployment_output_cap(request_kwargs, deployment)
    return default if from_deployment is None else min(from_deployment, default)


def _usage_field(container: object, name: str) -> object:
    if is_str_keyed_mapping(container):
        return container.get(name)
    return getattr(container, name, None)


def _cached_prompt_tokens(usage: object) -> int:
    reported: Final = (
        _usage_field(_usage_field(usage, "prompt_tokens_details"), "cached_tokens"),
        _usage_field(_usage_field(usage, "input_tokens_details"), "cached_tokens"),
        _usage_field(usage, "cache_read_input_tokens"),
    )
    return next((int(tokens) for tokens in reported if isinstance(tokens, int | float) and tokens), 0)


def _get_usage_tokens(usage: object) -> tuple[int, int, int]:
    if usage is None:
        return 0, 0, 0
    if hasattr(usage, "prompt_tokens") or hasattr(usage, "input_tokens"):
        prompt = int(getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", 0) or 0)
        return prompt, completion, _cached_prompt_tokens(usage)
    if isinstance(usage, dict):
        prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        return prompt, completion, _cached_prompt_tokens(usage)
    return 0, 0, 0


def _extract_response_usage(response_obj: object) -> object:
    if isinstance(response_obj, dict):
        return response_obj.get("usage")
    return getattr(response_obj, "usage", None)


def _openai_shaped_usage(usage: object) -> object:
    if isinstance(usage, dict) and litellm.AnthropicConfig.is_anthropic_usage_object(usage):
        return litellm.AnthropicConfig().calculate_usage(usage_object=usage, reasoning_content=None)
    return usage


def _usage_is_present(usage: object) -> bool:
    """
    True only if usage carries an actual input/output breakdown.

    ``total_tokens`` alone is deliberately excluded: ``_get_usage_tokens`` has
    no way to split a bare total into input vs. output, so treating it as
    "present" would resolve to (0, 0) and refund the full reservation as if
    zero tokens were used.
    """
    if usage is None:
        return False
    fields: Final = ("prompt_tokens", "completion_tokens", "input_tokens", "output_tokens")
    if isinstance(usage, dict):
        return any(key in usage for key in fields)
    return any(hasattr(usage, key) for key in fields)


def _resolve_reconcile_usage_tokens(
    kwargs: Mapping[str, object] | None,
    response_obj: object,
) -> tuple[int, int, bool]:
    """
    Resolve billable input and output tokens for post-call reconcile.

    Prefer the response usage object; fall back to standard_logging_object token
    fields. When usage cannot be resolved, return ``usage_resolved=False`` so
    callers keep the pre-call reservation instead of refunding it as zero usage.
    """
    usage: Final = _openai_shaped_usage(_extract_response_usage(response_obj))
    if _usage_is_present(usage):
        prompt_tokens, completion_tokens, cached_tokens = _get_usage_tokens(usage)
        return max(0, prompt_tokens - cached_tokens), completion_tokens, True

    if isinstance(kwargs, dict):
        standard_logging_object: Final = kwargs.get("standard_logging_object")
        if isinstance(standard_logging_object, dict):
            prompt_tokens = int(standard_logging_object.get("prompt_tokens") or 0)
            completion_tokens = int(standard_logging_object.get("completion_tokens") or 0)
            logged_usage: Final = _usage_field(standard_logging_object.get("metadata"), "usage_object")
            logged_cached_tokens: Final = _cached_prompt_tokens(logged_usage)
            # Same rationale as _usage_is_present: a bare total_tokens with no
            # prompt/completion breakdown can't be split, so it isn't treated
            # as resolved usage - the reservation is kept instead of refunded.
            if prompt_tokens or completion_tokens:
                return max(0, prompt_tokens - logged_cached_tokens), completion_tokens, True

    return 0, 0, False


def _reservation(
    itpm_reserved: int, otpm_reserved: int, itpm: UsageCounter | None, otpm: UsageCounter | None
) -> Mapping[str, object]:
    """The reservation as plain fields, which survive the copies logging makes of the request metadata."""
    return MappingProxyType(
        {
            IO_TOKEN_RESERVATION_ID_KEY: uuid.uuid4().hex,
            ITPM_RESERVED_KEY: itpm_reserved,
            OTPM_RESERVED_KEY: otpm_reserved,
            ITPM_COUNTER_KEY: None if itpm is None else asdict(itpm),
            OTPM_COUNTER_KEY: None if otpm is None else asdict(otpm),
        }
    )


def _stash_reservation_in_metadata(request: _IoTokenRateLimitRequest | None, reservation: Mapping[str, object]) -> None:
    metadata: Final = None if request is None or not request.kwargs else request.kwargs.get(request.metadata_channel)
    if isinstance(metadata, dict):
        metadata.update(reservation)
        metadata[IO_TOKEN_ATTEMPT_KEY] = current_attempt_id()


def _this_minute(counter: UsageCounter) -> UsageCounter:
    return replace(counter, window=get_utc_datetime().strftime("%H-%M"))


_Reservation = tuple[int, int, UsageCounter | None, UsageCounter | None]


def _reconcile_increments(
    reservation: _Reservation, billable_input: int, completion_tokens: int
) -> tuple[tuple[UsageCounter, int], ...]:
    itpm_reserved, otpm_reserved, itpm, otpm = reservation
    deltas: Final = ((itpm, billable_input - itpm_reserved), (otpm, completion_tokens - otpm_reserved))
    return tuple(
        (counter if delta < 0 else _this_minute(counter), delta)
        for counter, delta in deltas
        if counter is not None and delta
    )


def _stashed_counter(value: object) -> UsageCounter | None:
    if value is None:
        return None
    try:
        return _STASHED_COUNTER.validate_python(value)
    except ValidationError:
        return None


def _reserved_tokens(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _extract_reservation(reservation: Mapping[str, object]) -> _Reservation:
    return (
        _reserved_tokens(reservation.get(ITPM_RESERVED_KEY)),
        _reserved_tokens(reservation.get(OTPM_RESERVED_KEY)),
        _stashed_counter(reservation.get(ITPM_COUNTER_KEY)),
        _stashed_counter(reservation.get(OTPM_COUNTER_KEY)),
    )


def _as_mutable_mapping(value: object) -> MutableMapping[str, object] | None:
    """``value`` when it is a dict, else ``None``."""
    return value if isinstance(value, dict) else None


def _reservation_channels(kwargs: Mapping[str, object] | None) -> tuple[object, ...]:
    """
    Places a reservation may live, in priority order: the top-level metadata
    channels win over litellm_params.metadata (so a top-level stash is never
    shadowed), which win over the standard_logging_object copy.
    """
    if not isinstance(kwargs, dict):
        return ()
    top_level: Final = (kwargs.get("metadata"), kwargs.get("litellm_metadata"))
    litellm_params: Final = _as_mutable_mapping(kwargs.get("litellm_params"))
    from_params: Final = (
        () if litellm_params is None else (litellm_params.get("metadata"), litellm_params.get("litellm_metadata"))
    )
    standard_logging_object: Final = _as_mutable_mapping(kwargs.get("standard_logging_object"))
    from_logging_object: Final = () if standard_logging_object is None else (standard_logging_object.get("metadata"),)
    return top_level + from_params + from_logging_object


def _stashed_reservation(kwargs: Mapping[str, object] | None) -> Mapping[str, object] | None:
    channels: Final = (_as_mutable_mapping(channel) for channel in _reservation_channels(kwargs))
    return next((channel for channel in channels if channel is not None and ITPM_RESERVED_KEY in channel), None)


def _claim_reservation_from_kwargs(kwargs: Mapping[str, object] | None, *, any_attempt: bool = False) -> _Reservation:
    """
    Take the stashed reservation so nothing else settles it. Under a routed attempt only that attempt's
    own reservation is taken, unless ``any_attempt``: a late callback of an earlier attempt sharing these
    kwargs leaves the next attempt's reservation alone.
    """
    stashed: Final = _stashed_reservation(kwargs)
    if stashed is None:
        return 0, 0, None, None
    attempt: Final = current_attempt_id()
    if not any_attempt and attempt is not None and stashed.get(IO_TOKEN_ATTEMPT_KEY) not in (None, attempt):
        return 0, 0, None, None
    reservation: Final = _extract_reservation(stashed)
    reservation_id: Final = stashed.get(IO_TOKEN_RESERVATION_ID_KEY)
    _clear_reservation_from_kwargs(kwargs)
    if isinstance(reservation_id, str) and not _SETTLED_RESERVATIONS.claim(reservation_id):
        return 0, 0, None, None
    return reservation


def _clear_reservation_from_kwargs(kwargs: Mapping[str, object] | None) -> None:
    """
    Remove the stashed reservation so a retry on a different (e.g. non-IO)
    deployment does not re-process the already-reconciled/refunded reservation.
    """
    for channel in _reservation_channels(kwargs):
        channel_dict = _as_mutable_mapping(channel)
        if channel_dict is not None:
            for key in IO_TOKEN_RESERVATION_KEYS:
                channel_dict.pop(key, None)


def _reservation_value(value: int, limit: int | None) -> int:
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


@with_service_target(ROUTER_USAGE_TARGET)
def _sync_increment_with_rollback(
    usage: RouterUsage,
    counter: UsageCounter,
    value: int,
    limit: int | None,
    *,
    limit_label: str,
) -> None:
    if value <= 0 or limit is None:
        return
    if value > min(limit, MAX_INCREMENT):
        raise _rate_limit_error(limit_label, limit, value)
    current: Final = usage.increment_sync(counter, value, RoutingArgsTTL)
    if current is not None and current > limit:
        usage.increment_sync(counter, -value, RoutingArgsTTL)
        raise _rate_limit_error(limit_label, limit, current)


@with_service_target(ROUTER_USAGE_TARGET)
async def _increment_with_rollback(
    usage: RouterUsage,
    counter: UsageCounter,
    value: int,
    limit: int | None,
    *,
    limit_label: str,
) -> None:
    if value <= 0 or limit is None:
        return
    if value > min(limit, MAX_INCREMENT):
        raise _rate_limit_error(limit_label, limit, value)
    current: Final = await usage.increment(counter, value, RoutingArgsTTL)
    if current is not None and current > limit:
        await usage.increment(counter, -value, RoutingArgsTTL)
        raise _rate_limit_error(limit_label, limit, current)


@with_service_target(ROUTER_USAGE_TARGET)
def io_token_pre_call_check(
    usage: RouterUsage,
    deployment: dict,
) -> dict | None:
    itpm_limit, otpm_limit = get_deployment_io_token_limits(deployment)
    if itpm_limit is None and otpm_limit is None:
        return deployment

    request: Final = _io_token_rate_limit_request.get()
    request_kwargs: Final = None if request is None else request.kwargs
    _model, served_model = _tokenizer_and_served_models(deployment)
    estimated_input: Final = _estimate_input_tokens(request_kwargs, model=_model)
    otpm_to_reserve: Final = (
        _otpm_reservation(request_kwargs, deployment, served_model, otpm_limit) if otpm_limit is not None else 0
    )

    dt: Final = get_utc_datetime()
    current_minute: Final = dt.strftime("%H-%M")
    counters: Final = _get_counters(deployment, current_minute)
    if counters is None:
        return deployment
    itpm_counter, otpm_counter = counters

    itpm_reserved = 0
    otpm_reserved = 0

    if itpm_limit is not None:
        itpm_reserved = _reservation_value(estimated_input, itpm_limit)
        _sync_increment_with_rollback(
            usage,
            itpm_counter,
            itpm_reserved,
            itpm_limit,
            limit_label="ITPM",
        )

    if otpm_limit is not None:
        otpm_reserved = otpm_to_reserve
        try:
            _sync_increment_with_rollback(
                usage,
                otpm_counter,
                otpm_reserved,
                otpm_limit,
                limit_label="OTPM",
            )
        except Exception:
            if itpm_reserved > 0:
                usage.increment_sync(itpm_counter, -itpm_reserved, RoutingArgsTTL)
            raise

    _stash_reservation_in_metadata(
        request,
        _reservation(
            itpm_reserved,
            otpm_reserved,
            itpm_counter if itpm_limit is not None else None,
            otpm_counter if otpm_limit is not None else None,
        ),
    )
    return deployment


@with_service_target(ROUTER_USAGE_TARGET)
async def async_io_token_pre_call_check(
    usage: RouterUsage,
    deployment: dict,
) -> dict | None:
    itpm_limit, otpm_limit = get_deployment_io_token_limits(deployment)
    if itpm_limit is None and otpm_limit is None:
        return deployment

    request: Final = _io_token_rate_limit_request.get()
    request_kwargs: Final = None if request is None else request.kwargs
    if request is not None and request.opens_responses_socket:
        _stash_reservation_in_metadata(
            request,
            await async_admit_io_token_turn(usage, deployment, get_utc_datetime()),
        )
        return deployment
    _model, served_model = _tokenizer_and_served_models(deployment)
    estimated_input: Final = await offload_token_count(_estimate_input_tokens)(request_kwargs, model=_model)
    otpm_to_reserve: Final = (
        _otpm_reservation(request_kwargs, deployment, served_model, otpm_limit) if otpm_limit is not None else 0
    )

    dt: Final = get_utc_datetime()
    current_minute: Final = dt.strftime("%H-%M")
    counters: Final = _get_counters(deployment, current_minute)
    if counters is None:
        return deployment
    itpm_counter, otpm_counter = counters

    # The reservation runs in a task of its own: a request cancelled mid-reservation never knows how much
    # it took, so the task finishes and then hands back whatever it reserved.
    reserving: Final = asyncio.get_running_loop().create_task(
        _async_reserve(
            usage,
            itpm_counter,
            otpm_counter,
            _reservation_value(estimated_input, itpm_limit) if itpm_limit is not None else 0,
            otpm_to_reserve,
            itpm_limit,
            otpm_limit,
        )
    )
    try:
        reservation: Final = await asyncio.shield(reserving)
    except asyncio.CancelledError:
        _ = release_in_background(_refund_once_reserved(usage, reserving))
        raise
    _stash_reservation_in_metadata(request, reservation)
    return deployment


async def _refund_once_reserved(usage: RouterUsage, reserving: asyncio.Task[Mapping[str, object]]) -> None:
    try:
        reservation: Final = await reserving
    except Exception:  # noqa: BLE001  # a refused or failed reservation holds nothing
        return
    await _async_increment_all(usage, _refunds(_extract_reservation(reservation)))


async def _async_reserve(
    usage: RouterUsage,
    itpm_counter: UsageCounter,
    otpm_counter: UsageCounter,
    itpm_reserved: int,
    otpm_reserved: int,
    itpm_limit: int | None,
    otpm_limit: int | None,
) -> Mapping[str, object]:
    if itpm_limit is not None:
        await _increment_with_rollback(
            usage,
            itpm_counter,
            itpm_reserved,
            itpm_limit,
            limit_label="ITPM",
        )

    if otpm_limit is not None:
        try:
            await _increment_with_rollback(
                usage,
                otpm_counter,
                otpm_reserved,
                otpm_limit,
                limit_label="OTPM",
            )
        except Exception:
            # Any failure reserving OTPM (a 429 or a transient cache error) must
            # release the ITPM reservation already made, or it stays inflated
            # until the TTL expires.
            if itpm_reserved > 0:
                await usage.increment(itpm_counter, -itpm_reserved, RoutingArgsTTL)
            raise

    return _reservation(
        itpm_reserved,
        otpm_reserved if otpm_limit is not None else 0,
        itpm_counter if itpm_limit is not None else None,
        otpm_counter if otpm_limit is not None else None,
    )


async def _raise_if_at_limit(usage: RouterUsage, counter: UsageCounter, limit: int | None, limit_label: str) -> None:
    if limit is None:
        return
    count: Final = await usage.shared_or_local(counter)
    if count is not None and count >= limit:
        raise _rate_limit_error(limit_label, limit, count)


@with_service_target(ROUTER_USAGE_TARGET)
async def async_admit_io_token_turn(
    usage: RouterUsage,
    deployment: Mapping[str, object],
    now: datetime,
) -> Mapping[str, object]:
    configured: Final = dict(deployment)
    itpm_limit, otpm_limit = get_deployment_io_token_limits(configured)
    if itpm_limit is None and otpm_limit is None:
        return MappingProxyType({})
    counters: Final = _get_counters(configured, now.strftime("%H-%M"))
    if counters is None:
        return MappingProxyType({})
    itpm_counter, otpm_counter = counters
    await _raise_if_at_limit(usage, itpm_counter, itpm_limit, "ITPM")
    await _raise_if_at_limit(usage, otpm_counter, otpm_limit, "OTPM")
    return _reservation(
        0, 0, itpm_counter if itpm_limit is not None else None, otpm_counter if otpm_limit is not None else None
    )


@with_service_target(ROUTER_USAGE_TARGET)
def io_token_reconcile_success(
    usage: RouterUsage,
    kwargs: Mapping[str, object] | None,
    response_obj: object,
) -> None:
    request_kwargs: Final[Mapping[str, object] | None] = kwargs
    response: Final[object] = response_obj
    reservation: Final = _claim_reservation_from_kwargs(request_kwargs)
    itpm_reserved, otpm_reserved, itpm_counter, otpm_counter = reservation
    if itpm_counter is None and otpm_counter is None:
        return

    billable_input, completion_tokens, usage_resolved = _resolve_reconcile_usage_tokens(request_kwargs, response)

    if usage_resolved:
        for counter, delta in _reconcile_increments(reservation, billable_input, completion_tokens):
            usage.increment_sync(counter, delta, RoutingArgsTTL)
    else:
        verbose_router_logger.debug(
            "[IO TOKEN LIMIT] usage missing; keeping reservation (itpm_reserved=%s, otpm_reserved=%s)",
            itpm_reserved,
            otpm_reserved,
        )

    verbose_router_logger.debug(
        "[IO TOKEN LIMIT] reconciled (usage_resolved=%s, itpm_reserved=%s, billable_input=%s, otpm_reserved=%s, output=%s)",
        usage_resolved,
        itpm_reserved,
        billable_input,
        otpm_reserved,
        completion_tokens,
    )


@with_service_target(ROUTER_USAGE_TARGET)
async def async_io_token_reconcile_success(
    usage: RouterUsage,
    kwargs: Mapping[str, object] | None,
    response_obj: object,
) -> None:
    request_kwargs: Final[Mapping[str, object] | None] = kwargs
    response: Final[object] = response_obj
    reservation: Final = _claim_reservation_from_kwargs(request_kwargs)
    itpm_reserved, otpm_reserved, itpm_counter, otpm_counter = reservation
    if itpm_counter is None and otpm_counter is None:
        return

    billable_input, completion_tokens, usage_resolved = _resolve_reconcile_usage_tokens(request_kwargs, response)

    # A refund goes back to the minute that held the reservation, while usage above the reservation is charged
    # to the minute the response finished in, so a call that runs past its minute still counts against a live
    # window.
    if usage_resolved:
        await released(
            (
                release_in_background(
                    _async_increment_all(
                        usage,
                        _reconcile_increments(reservation, billable_input, completion_tokens),
                    )
                ),
            )
        )
    else:
        verbose_router_logger.debug(
            "[IO TOKEN LIMIT] usage missing; keeping reservation (itpm_reserved=%s, otpm_reserved=%s)",
            itpm_reserved,
            otpm_reserved,
        )

    verbose_router_logger.debug(
        "[IO TOKEN LIMIT] reconciled (usage_resolved=%s, itpm_reserved=%s, billable_input=%s, otpm_reserved=%s, output=%s)",
        usage_resolved,
        itpm_reserved,
        billable_input,
        otpm_reserved,
        completion_tokens,
    )


@with_service_target(ROUTER_USAGE_TARGET)
def io_token_refund_failure(
    usage: RouterUsage,
    kwargs: Mapping[str, object] | None,
    *,
    any_attempt: bool = False,
) -> None:
    request_kwargs: Final[Mapping[str, object] | None] = kwargs
    reservation: Final = _claim_reservation_from_kwargs(request_kwargs, any_attempt=any_attempt)
    itpm_reserved, otpm_reserved, _, _ = reservation
    for counter, delta in _refunds(reservation):
        usage.increment_sync(counter, delta, RoutingArgsTTL)
    verbose_router_logger.debug("[IO TOKEN LIMIT] refunded ITPM=%s OTPM=%s", itpm_reserved, otpm_reserved)


def refund_stale_reservation_before_retry(usage: RouterUsage, kwargs: Mapping[str, object] | None) -> None:
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
    io_token_refund_failure(usage, kwargs, any_attempt=True)


@with_service_target(ROUTER_USAGE_TARGET)
async def async_io_token_refund_failure(
    usage: RouterUsage,
    kwargs: Mapping[str, object] | None,
) -> None:
    refund: Final = release_io_token_reservation(usage, kwargs)
    if refund is not None:
        await released((refund,))


def release_io_token_reservation(
    usage: RouterUsage,
    kwargs: Mapping[str, object] | None,
) -> asyncio.Task[None] | None:
    reservation: Final = _claim_reservation_from_kwargs(kwargs)
    refunds: Final = _refunds(reservation)
    if not refunds:
        return None
    verbose_router_logger.debug("[IO TOKEN LIMIT] refunding ITPM=%s OTPM=%s", reservation[0], reservation[1])
    return release_in_background(_async_increment_all(usage, refunds))


def _refunds(reservation: _Reservation) -> tuple[tuple[UsageCounter, int], ...]:
    itpm_reserved, otpm_reserved, itpm_counter, otpm_counter = reservation
    held: Final = ((itpm_counter, itpm_reserved), (otpm_counter, otpm_reserved))
    return tuple((counter, -reserved) for counter, reserved in held if counter is not None and reserved > 0)


async def _async_increment_all(usage: RouterUsage, increments: tuple[tuple[UsageCounter, int], ...]) -> None:
    for counter, delta in increments:
        await usage.increment(counter, delta, RoutingArgsTTL)


def build_io_token_rate_limit_headers(
    *,
    itpm_limit: int | None,
    otpm_limit: int | None,
    current_itpm: int | None,
    current_otpm: int | None,
) -> dict[str, int]:
    headers: Final[dict[str, int]] = {}
    reset: Final = seconds_until_minute_reset()
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
