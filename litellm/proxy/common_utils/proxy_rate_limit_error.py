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
  ``batch_rate_limiter``, ``max_budget_limiter``, ``max_iterations_limiter``,
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
from typing import Any, Dict, Mapping, Optional, Union

from fastapi import HTTPException

from litellm.exceptions import RateLimitError, RateLimitErrorCategory, RateLimitType


def map_v3_rate_limit_type(
    v3_value: Optional[str],
) -> Optional[RateLimitType]:
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


def _coerce_message(detail: Any) -> str:
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
            return str(detail)
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
class ProxyRateLimitError(HTTPException, RateLimitError):  # type: ignore[misc]
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
        detail: Any,
        headers: Optional[Mapping[str, Any]] = None,
        category: Union[
            str, RateLimitErrorCategory
        ] = RateLimitErrorCategory.LITELLM_RATE_LIMIT,
        rate_limit_type: Optional[Union[str, RateLimitType]] = None,
        model: Optional[str] = None,
        llm_provider: Optional[str] = "litellm_proxy",
    ):
        # Normalize None → safe defaults so callers (and the resolver helper
        # in `rate_limiter_utils`) can pass `None` without producing an
        # instance whose `.llm_provider` attribute is `None` — that would
        # break Prometheus' `_get_exception_class_name` (it calls
        # `.capitalize()` on the provider string).
        model = model or ""
        llm_provider = llm_provider or "litellm_proxy"
        message = _coerce_message(detail)
        stringified_headers: Optional[Dict[str, str]] = (
            {k: str(v) for k, v in headers.items()} if headers else None
        )

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
