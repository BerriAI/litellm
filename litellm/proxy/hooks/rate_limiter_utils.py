"""
Shared utility functions for rate limiter hooks.
"""

from typing import Any, Optional, Tuple, Union

from fastapi import HTTPException

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.exceptions import RateLimitError
from litellm.types.router import ModelGroupInfo
from litellm.types.utils import PriorityReservationDict

PROXY_LLM_PROVIDER_FALLBACK = "litellm_proxy"


def resolve_llm_provider_for_rate_limit(
    model: Optional[str],
) -> Tuple[str, str]:
    """
    Resolve ``(model, llm_provider)`` for a request being rejected by an
    internal proxy-side rate-limit hook.

    These hooks fire from ``async_pre_call_hook`` — well before
    :func:`litellm.get_llm_provider` is invoked anywhere else in the request
    lifecycle — so the raised 429 would otherwise have an empty
    ``llm_provider`` field, making the resulting Prometheus
    ``litellm_proxy_failed_requests_metric`` show up with
    ``exception_class="RateLimitError"`` and no provider attribution.

    Wrapped defensively: if ``model`` is missing, malformed, or
    ``get_llm_provider`` raises (unknown alias, router-only model, etc.) we
    fall back to ``("", "litellm_proxy")`` so we never break the request path
    by piling a second exception on top of the rate-limit one we're trying to
    raise.
    """
    if not model:
        return "", PROXY_LLM_PROVIDER_FALLBACK
    try:
        resolved_model, custom_llm_provider, _, _ = litellm.get_llm_provider(
            model=model,
        )
        return (
            resolved_model or model,
            custom_llm_provider or PROXY_LLM_PROVIDER_FALLBACK,
        )
    except Exception as e:
        verbose_proxy_logger.debug(
            "rate_limiter_utils.resolve_llm_provider_for_rate_limit: "
            "could not resolve provider for model=%s, falling back to %s. err=%s",
            model,
            PROXY_LLM_PROVIDER_FALLBACK,
            str(e),
        )
        return model, PROXY_LLM_PROVIDER_FALLBACK


class ProxyHTTPRateLimitError(HTTPException, RateLimitError):  # type: ignore[misc]
    """
    HTTPException raised by proxy-side rate-limit hooks that *also* exposes
    ``model`` and ``llm_provider`` attributes.

    Why both base classes:

    - The proxy server's exception handler keys off ``HTTPException`` to render
      a 429 response, so we must remain an ``HTTPException``.
    - Downstream loggers (Prometheus ``async_post_call_failure_hook``,
      structured logging, observability callbacks) read ``exception.llm_provider``
      via :meth:`litellm.integrations.prometheus.PrometheusLogger._get_exception_class_name`
      and ``isinstance(exc, RateLimitError)`` for category routing. Inheriting
      from :class:`litellm.exceptions.RateLimitError` keeps that wiring intact.

    We intentionally do not call ``RateLimitError.__init__`` (which constructs
    an httpx.Response) — it isn't needed here and just adds failure surface.
    Attribute parity is what downstream consumers rely on.
    """

    def __init__(
        self,
        status_code: int,
        detail: Any = None,
        headers: Optional[dict] = None,
        *,
        model: str = "",
        llm_provider: str = PROXY_LLM_PROVIDER_FALLBACK,
    ) -> None:
        HTTPException.__init__(
            self, status_code=status_code, detail=detail, headers=headers
        )
        self.status_code = status_code
        self.model = model or ""
        self.llm_provider = llm_provider or PROXY_LLM_PROVIDER_FALLBACK
        # `message` is what RateLimitError.__str__ would print and what some
        # observability callbacks log. Keep it human-readable.
        self.message = detail if isinstance(detail, str) else str(detail)
        # `RateLimitError.__str__` (resolved via MRO since Starlette's
        # HTTPException doesn't define `__str__`) unconditionally reads
        # these attributes. Set them so `str(exc)` doesn't raise
        # AttributeError from logging/traceback paths.
        self.num_retries: Optional[int] = None
        self.max_retries: Optional[int] = None


def convert_priority_to_percent(
    value: Union[float, PriorityReservationDict], model_info: Optional[ModelGroupInfo]
) -> float:
    """
    Convert priority reservation value to percentage (0.0-1.0).

    Supports three formats:
    1. Plain float/int: 0.9 -> 0.9 (90%)
    2. Dict with percent: {"type": "percent", "value": 0.9} -> 0.9
    3. Dict with rpm: {"type": "rpm", "value": 900} -> 900/model_rpm
    4. Dict with tpm: {"type": "tpm", "value": 900000} -> 900000/model_tpm

    Args:
        value: Priority value as float or dict with type/value keys
        model_info: Model configuration containing rpm/tpm limits

    Returns:
        float: Percentage value between 0.0 and 1.0
    """
    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, dict):
        val_type = value.get("type", "percent")
        val_num = value.get("value", 1.0)

        if val_type == "percent":
            return float(val_num)
        elif val_type == "rpm" and model_info and model_info.rpm and model_info.rpm > 0:
            return float(val_num) / model_info.rpm
        elif val_type == "tpm" and model_info and model_info.tpm and model_info.tpm > 0:
            return float(val_num) / model_info.tpm

        # Fallback: treat as percent
        return float(val_num)
