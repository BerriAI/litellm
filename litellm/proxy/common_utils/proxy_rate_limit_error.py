"""
ProxyRateLimitError — a unified rate-limit exception used by litellm's
proxy-side hooks.

Background
----------
LiteLLM previously surfaced rate-limit conditions through *several* unrelated
exception types:

* :class:`litellm.exceptions.RateLimitError` — raised by exception mapping when
  an upstream LLM provider returns 429.
* :class:`fastapi.HTTPException` (status 429) — raised directly by proxy hooks
  such as ``parallel_request_limiter``, ``dynamic_rate_limiter``,
  ``batch_rate_limiter``, ``max_iterations_limiter``,
  etc.
* :class:`litellm.llms.base_llm.chat.transformation.BaseLLMException` (status
  429) — raised by some provider transports.

This made it impossible for downstream code (and end users) to express
"is this a rate limit?" with a single ``except`` clause, and impossible to
distinguish *where* the rate limit originated (vendor vs. litellm, batch vs.
chat) without ad-hoc string-matching on the message.

This module provides a single proxy-side error class that:

1. Is a subclass of :class:`litellm.exceptions.RateLimitError`, so user code
   that catches ``RateLimitError`` works for *every* rate-limit source.
2. Is also a subclass of :class:`fastapi.HTTPException`, so existing proxy
   plumbing (``isinstance(e, HTTPException)`` branches in route handlers and
   FastAPI's own dispatcher) continues to behave the same way and the
   ``retry-after`` / ``rate_limit_type`` / ``reset_at`` headers are preserved
   on the wire.
3. Carries a :attr:`category` field (one of
   :class:`litellm.exceptions.RateLimitErrorCategory`) so callers can switch on
   the rate limit source.
"""

import json
from collections.abc import Mapping, MutableMapping
from contextvars import ContextVar, Token
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.exceptions import RateLimitError, RateLimitErrorCategory, RateLimitType

PER_MODEL_RATE_LIMIT_DESCRIPTOR_KEYS: Final = frozenset(
    {
        "model_per_key",
        "model_per_team",
        "model_per_end_user",
        "model_per_organization",
        "model_per_project",
        "model_per_project_itpm",
        "model_per_project_otpm",
        "model_per_customer_org",
        "model_per_customer_team",
        "model_per_customer_member",
    }
)
DISABLE_FALLBACKS_ON_PER_MODEL_RATE_LIMITS_SETTING: Final = "disable_fallbacks_on_per_model_rate_limits"
_DISABLE_FALLBACKS_ON_PER_MODEL_RATE_LIMITS_FLAG: Final[TypeAdapter[bool | None]] = TypeAdapter(bool | None)


def per_model_rate_limits_disable_fallbacks(general_settings: Mapping[str, object]) -> bool:
    raw_value: Final = general_settings.get(DISABLE_FALLBACKS_ON_PER_MODEL_RATE_LIMITS_SETTING)
    try:
        return _DISABLE_FALLBACKS_ON_PER_MODEL_RATE_LIMITS_FLAG.validate_python(raw_value) is True
    except ValidationError:
        verbose_proxy_logger.warning(
            "general_settings.%s=%r is not a boolean, treating it as disabled",
            DISABLE_FALLBACKS_ON_PER_MODEL_RATE_LIMITS_SETTING,
            raw_value,
        )
        return False


FALLBACK_ON_SATURATION_SETTING: Final = "fallback_on_saturation"
SATURATION_FALLBACK_DESCRIPTOR_KEY: Final = "model_saturation_fallback"
_FALLBACK_ON_SATURATION_FLAG: Final[TypeAdapter[bool | None]] = TypeAdapter(bool | None)


def fallback_on_saturation_enabled(router_settings: object) -> bool:
    """Whether the key's or team's ``router_settings`` ask for fallbacks once the requested model reaches
    ``saturation_threshold``, before the dynamic rate limiter admits or refuses the request on that model."""
    raw_value: Final = _metadata_bucket(router_settings).get(FALLBACK_ON_SATURATION_SETTING)
    try:
        return _FALLBACK_ON_SATURATION_FLAG.validate_python(raw_value, strict=True) is True
    except ValidationError:
        verbose_proxy_logger.warning(
            "router_settings.%s=%r is not a boolean, treating it as disabled", FALLBACK_ON_SATURATION_SETTING, raw_value
        )
        return False


def drop_client_saturation_opt_in(
    request_data: MutableMapping[str, object],  # mutable-ok: in-place edit keeps the caller's request object
) -> None:
    """Drop ``fallback_on_saturation`` from the client's ``router_settings_override``. Only the key's or team's saved
    ``router_settings`` turn the saturation walk on, and the proxy writes those over this field before the limiter
    reads it."""
    override: Final = _metadata_bucket(request_data.get("router_settings_override"))
    if FALLBACK_ON_SATURATION_SETTING not in override:
        return
    request_data["router_settings_override"] = {  # rebind-ok: the request object must stay the caller's
        key: value for key, value in override.items() if key != FALLBACK_ON_SATURATION_SETTING
    }


_SaturationWalkState: TypeAlias = Literal["off", "armed", "exhausted"]
_SATURATION_FALLBACK_WALK: Final[ContextVar[_SaturationWalkState]] = ContextVar(
    "litellm_saturation_fallback_walk", default="off"
)


def saturation_fallback_walk_armed() -> bool:
    """Whether a caller that catches the handoff (``_pre_call_with_fallbacks``) is running this request's hooks and
    has not walked the fallbacks yet."""
    return _SATURATION_FALLBACK_WALK.get() == "armed"


def saturation_fallbacks_exhausted() -> bool:
    """Whether this request already tried every fallback at saturation and is back on the model it asked for."""
    return _SATURATION_FALLBACK_WALK.get() == "exhausted"


def arm_saturation_fallback_walk() -> Token[_SaturationWalkState]:
    """Mark the current request as running under the fallback walk, so the limiter may hand a saturated model off to
    it. The state lives in the request's context, never in its body, so nothing reaches the provider and no client
    field can arm, skip or exhaust the walk. Routes that call the hooks directly never arm it and keep admitting or
    refusing as before."""
    return _SATURATION_FALLBACK_WALK.set("armed")


def mark_saturation_fallbacks_exhausted() -> None:
    _SATURATION_FALLBACK_WALK.set("exhausted")


def disarm_saturation_fallback_walk(token: Token[_SaturationWalkState]) -> None:
    _SATURATION_FALLBACK_WALK.reset(token)


def fallback_on_saturation_requested(request_data: Mapping[str, object]) -> bool:
    """Whether the dynamic rate limiter should hand a saturated model's request to the fallback walk instead of
    admitting or refusing it: the key or team opted in through ``router_settings_override`` and the fallback walk
    is the caller and has not already run for this request."""
    return (
        fallback_on_saturation_enabled(request_data.get("router_settings_override"))
        and saturation_fallback_walk_armed()
    )


def is_saturation_fallback(exc: "ProxyRateLimitError") -> bool:
    """Whether ``exc`` is the limiter asking for the fallback walk rather than refusing the request. The model it
    names still has room, so when no fallback admits the request it goes back to that model instead of a 429."""
    return exc.descriptor_key == SATURATION_FALLBACK_DESCRIPTOR_KEY


RATE_LIMIT_FALLBACK_METADATA_KEY: Final = "rate_limit_fallback"
RATE_LIMIT_FALLBACK_REASON_HEADER: Final = "x-litellm-fallback-reason"
ATTEMPTED_FALLBACKS_HEADER: Final = "x-litellm-attempted-fallbacks"


@dataclass(frozen=True, slots=True)
class RateLimitFallback:
    original_model_group: str
    descriptor_key: str
    attempts: int = 1

    def stamp(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "original_model_group": self.original_model_group,
                "descriptor_key": self.descriptor_key,
                "attempts": self.attempts,
            }
        )

    @property
    def reason(self) -> str:
        return f"rate_limit:{self.descriptor_key}"


class _StampedBucket(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    rate_limit_fallback: RateLimitFallback | None = None


class _StampedRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    metadata: _StampedBucket | None = None
    litellm_metadata: _StampedBucket | None = None


_STAMPED_REQUEST: Final[TypeAdapter[_StampedRequest]] = TypeAdapter(_StampedRequest)
_ROUTER_HOPS: Final[TypeAdapter[int]] = TypeAdapter(int)


def rate_limit_fallback_from_request_data(request_data: Mapping[str, object]) -> RateLimitFallback | None:
    try:
        stamped: Final = _STAMPED_REQUEST.validate_python(request_data)
    except ValidationError:
        return None
    buckets: Final = (stamped.metadata, stamped.litellm_metadata)
    return next(
        (bucket.rate_limit_fallback for bucket in buckets if bucket is not None and bucket.rate_limit_fallback), None
    )


_METADATA_BUCKET: Final[TypeAdapter[Mapping[str, object]]] = TypeAdapter(Mapping[str, object])


def _metadata_bucket(value: object) -> Mapping[str, object]:
    try:
        return _METADATA_BUCKET.validate_python(value)
    except ValidationError:
        return MappingProxyType({})


def with_rate_limit_fallback(
    request_data: Mapping[str, object], bucket_name: str, fallback: RateLimitFallback
) -> Mapping[str, object]:
    bucket: Final = _metadata_bucket(request_data.get(bucket_name))
    return MappingProxyType(
        {**request_data, bucket_name: {**bucket, RATE_LIMIT_FALLBACK_METADATA_KEY: dict(fallback.stamp())}}
    )


def rate_limit_fallback_headers(fallback: RateLimitFallback, attempted_fallbacks: object) -> Mapping[str, str]:
    try:
        router_hops: Final = _ROUTER_HOPS.validate_python(attempted_fallbacks)
    except ValidationError:
        return MappingProxyType(
            {
                ATTEMPTED_FALLBACKS_HEADER: str(max(fallback.attempts, 1)),
                RATE_LIMIT_FALLBACK_REASON_HEADER: fallback.reason,
            }
        )
    return MappingProxyType(
        {
            ATTEMPTED_FALLBACKS_HEADER: str(max(router_hops, 0) + max(fallback.attempts, 1)),
            RATE_LIMIT_FALLBACK_REASON_HEADER: fallback.reason,
        }
    )


def map_v3_rate_limit_type(
    v3_value: str | None,
) -> RateLimitType | None:
    """
    Map the v3 rate limiter's internal `status["rate_limit_type"]` strings
    onto the public :class:`RateLimitType` enum.

    The v3 limiter uses the literal values ``"requests"``, ``"tokens"``, and
    ``"max_parallel_requests"``. We collapse the last one onto
    :attr:`RateLimitType.CONCURRENT_REQUESTS` because that's the public name
    documented for users and dashboards. Unrecognized values return ``None``
    so the field stays absent rather than carrying garbage downstream.
    """
    if v3_value == "tokens":
        return RateLimitType.TOKENS
    if v3_value == "max_parallel_requests":
        return RateLimitType.CONCURRENT_REQUESTS
    if v3_value == "requests":
        return RateLimitType.REQUESTS
    return None


def _coerce_message(detail: object) -> str:
    """Best-effort, JSON-friendly stringification of an HTTPException-style detail."""
    if detail is None:
        return ""
    if isinstance(detail, str):
        return detail
    if isinstance(detail, Mapping):
        for key in ("error", "message"):
            if isinstance(detail.get(key), str):
                return detail[key]
            inner = detail.get(key)
            if isinstance(inner, Mapping) and isinstance(inner.get("message"), str):
                return inner["message"]
        try:
            return json.dumps(detail)
        except (TypeError, ValueError):
            return f"{detail}"
    return str(detail)


# NOTE: mypy emits two `[misc]` errors on the class line below because the
# bases declare overlapping attributes with related-but-not-identical
# annotations:
#   * `status_code` is `int` on starlette HTTPException but `Literal[429]` on
#     openai.RateLimitError (every openai status-error subclass narrows it
#     this way and silences pyright with the same convention).
#   * `headers` is `Mapping[str, str] | None` on HTTPException; we narrow it
#     to `Optional[Dict[str, str]]` on RateLimitError because we always carry
#     a stringified dict.
# Both narrowings are intentional and handled at construction time — every
# instance always has status_code == 429 and a Dict-typed headers — so we
# silence the ATTR-overlap check rather than relax the annotations.
class ProxyRateLimitError(HTTPException, RateLimitError):
    """
    A 429 raised by litellm's proxy-side rate limiting hooks.

    This class deliberately inherits from BOTH
    :class:`litellm.exceptions.RateLimitError` and :class:`fastapi.HTTPException`
    so the same instance can flow through:

    * ``except RateLimitError`` (user / SDK code that wants a category-aware
      handler), and
    * ``isinstance(e, HTTPException)`` (FastAPI / proxy_server.py route
      handlers that need to forward ``status_code``, ``detail`` and
      ``headers`` back to the client).

    Downstream code should prefer this class over
    ``raise HTTPException(status_code=429, ...)`` for litellm-internal rate
    limits.

    Parameters
    ----------
    detail:
        The structured error payload. Forwarded as ``HTTPException.detail`` so
        FastAPI's default exception handler will serialize it verbatim.
    headers:
        Optional response headers (e.g. ``retry-after``). Values are stringified
        to satisfy FastAPI's typing.
    category:
        One of :class:`RateLimitErrorCategory`. Defaults to
        ``LITELLM_RATE_LIMIT`` since this class is only used by litellm's own
        proxy-side limiters; pass ``LITELLM_BATCH_RATE_LIMIT`` for the batch
        limiter, etc.
    model / llm_provider:
        Optional context, propagated to the inherited ``RateLimitError`` for
        compatibility with logging / standard payload extraction.
    """

    # Prometheus' ``exception_class`` label is pinned to "HTTPException" for
    # this type: before the unified class existed, proxy-side 429s surfaced as
    # ``fastapi.HTTPException`` and existing dashboards/alerts key off that exact
    # value. Distinguishing vendor vs. litellm 429s is now the job of the
    # ``rate_limit_category`` / ``rate_limit_type`` labels.
    prometheus_exception_class_name = "HTTPException"

    def __init__(
        self,
        detail: object,
        headers: Mapping[str, object] | None = None,
        category: str | RateLimitErrorCategory = RateLimitErrorCategory.LITELLM_RATE_LIMIT,
        rate_limit_type: str | RateLimitType | None = None,
        model: str | None = None,
        llm_provider: str | None = "litellm_proxy",
        *,
        descriptor_key: str | None = None,
    ):
        # Normalize None → safe defaults so callers (and the resolver helper
        # in `rate_limiter_utils`) can pass `None` without producing an
        # instance whose `.llm_provider` attribute is `None` — that would
        # break Prometheus' `_get_exception_class_name` (it calls
        # `.capitalize()` on the provider string).
        model = model or ""
        llm_provider = llm_provider or "litellm_proxy"
        message: Final = _coerce_message(detail)
        stringified_headers: Final[dict[str, str] | None] = {k: str(v) for k, v in headers.items()} if headers else None

        # Initialize the FastAPI HTTPException portion first so its attributes
        # (status_code, detail, headers) are already on the instance before
        # RateLimitError.__init__ runs and possibly overrides them.
        HTTPException.__init__(
            self,
            status_code=429,
            detail=detail,
            headers=stringified_headers,
        )

        # Now initialize the litellm RateLimitError portion. We deliberately
        # pass the structured detail through so RateLimitError preserves it as
        # its `.detail` attribute too — keeping both sides of the MRO
        # consistent.
        RateLimitError.__init__(
            self,
            message=message,
            llm_provider=llm_provider,
            model=model,
            category=category,
            rate_limit_type=rate_limit_type,
            headers=stringified_headers,
            detail=detail,
        )
        # RateLimitError.__init__ overwrites self.headers with its own copy and
        # leaves self.status_code at 429 — restore the HTTPException-style
        # headers value so downstream code that pulls headers off the
        # instance gets back exactly what the limiter passed in.
        self.headers = stringified_headers
        self.detail = detail
        self.status_code = 429
        self.descriptor_key = descriptor_key
