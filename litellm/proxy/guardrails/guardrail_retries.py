"""
Guardrail retry logic: mirrors router-level model retries (408, 409, 429, 5xx, no-status).
Never retries ModifyResponseException, ContentPolicyViolationError, NotFoundError.
"""

import asyncio
from typing import Any, Callable, Optional, TypeVar

import litellm
from litellm._logging import verbose_proxy_logger

T = TypeVar("T")

# Default retry config when not set on guardrail
DEFAULT_GUARDRAIL_NUM_RETRIES = 2
DEFAULT_GUARDRAIL_RETRY_AFTER = 0.0

# Cached router settings to avoid importing proxy_server in the request path on every guardrail call.
_CACHED_ROUTER_SETTINGS: Optional[dict] = None


def _get_router_settings() -> Optional[dict]:
    """Resolve router settings once per process; used for guardrail retry defaults."""
    global _CACHED_ROUTER_SETTINGS
    if _CACHED_ROUTER_SETTINGS is not None:
        return _CACHED_ROUTER_SETTINGS
    try:
        from litellm.proxy.proxy_server import llm_router

        if llm_router is not None:
            s = llm_router.get_settings()
            if s is not None:
                _CACHED_ROUTER_SETTINGS = s
                return _CACHED_ROUTER_SETTINGS
    except Exception:
        pass
    _CACHED_ROUTER_SETTINGS = {}
    return _CACHED_ROUTER_SETTINGS


def should_retry_guardrail_error(error: Exception) -> bool:
    """
    Decide if a guardrail error is retriable.

    - Never retry: ModifyResponseException, ContentPolicyViolationError, NotFoundError.
    - Retry: 408, 409, 429, 5xx (via litellm._should_retry), and errors without status (e.g. network).
    """
    from litellm.integrations.custom_guardrail import ModifyResponseException

    if isinstance(error, ModifyResponseException):
        return False
    try:
        import litellm.exceptions as litellm_exceptions

        if isinstance(error, litellm_exceptions.ContentPolicyViolationError):
            return False
        if isinstance(error, litellm_exceptions.NotFoundError):
            return False
    except Exception:
        pass
    status_code: Optional[int] = getattr(error, "status_code", None)
    if status_code is not None:
        return litellm._should_retry(status_code)
    # No status code (e.g. network error, timeout) -> retry
    return True


def _time_to_sleep_before_guardrail_retry(
    remaining_retries: int,
    num_retries: int,
    retry_after: float,
    response_headers: Optional[Any] = None,
) -> float:
    """
    Compute sleep before next guardrail retry using litellm._calculate_retry_after.
    retry_after (config) is used as min_timeout for backoff.
    """
    min_timeout = max(0.0, retry_after)
    return litellm._calculate_retry_after(
        remaining_retries=remaining_retries,
        max_retries=num_retries,
        response_headers=response_headers,
        min_timeout=min_timeout,
    )


async def run_guardrail_with_retries(
    coro_factory: Callable[[], Any],
    num_retries: int,
    retry_after: float,
    guardrail_name: str,
) -> Any:
    """
    Run an async guardrail "call" with retries.

    coro_factory: callable that returns a new coroutine for each attempt (so each attempt is fresh).
    num_retries: max number of attempts (e.g. 2 means try once, then up to 2 retries = 3 total).
    retry_after: minimum seconds to wait before retry (used in backoff).
    guardrail_name: for logging.

    Uses same semantics as router retry loop: retries on 408/409/429/5xx and no-status errors.
    """
    if num_retries <= 0:
        coro = coro_factory()
        return await coro

    attempt = 0
    remaining = num_retries

    while True:
        try:
            coro = coro_factory()
            return await coro
        except Exception as e:
            if not should_retry_guardrail_error(e):
                raise
            remaining -= 1
            if remaining < 0:
                raise
            response_headers = getattr(e, "response", None)
            if response_headers is not None and hasattr(response_headers, "headers"):
                response_headers = response_headers.headers
            sleep_seconds = _time_to_sleep_before_guardrail_retry(
                remaining_retries=remaining,
                num_retries=num_retries,
                retry_after=retry_after,
                response_headers=response_headers,
            )
            verbose_proxy_logger.warning(
                "Guardrail %s attempt %s failed (retriable): %s. Retrying in %.2fs (%s attempts left).",
                guardrail_name,
                attempt + 1,
                type(e).__name__,
                sleep_seconds,
                remaining,
            )
            await asyncio.sleep(sleep_seconds)
            attempt += 1


def get_guardrail_retry_config(guardrail_to_apply: Any) -> tuple[int, float]:
    """
    Read num_retries and retry_after from the guardrail instance.

    Looks at optional_params and guardrail_config. Returns defaults when not set.
    """
    num_retries = DEFAULT_GUARDRAIL_NUM_RETRIES
    retry_after = DEFAULT_GUARDRAIL_RETRY_AFTER

    opts = getattr(guardrail_to_apply, "optional_params", None)
    if isinstance(opts, dict):
        if "num_retries" in opts and opts["num_retries"] is not None:
            num_retries = int(opts["num_retries"])
        if "retry_after" in opts and opts["retry_after"] is not None:
            retry_after = float(opts["retry_after"])

    guardrail_config = getattr(guardrail_to_apply, "guardrail_config", None)
    if isinstance(guardrail_config, dict):
        if not (isinstance(opts, dict) and "num_retries" in opts) and "num_retries" in guardrail_config and guardrail_config["num_retries"] is not None:
            num_retries = int(guardrail_config["num_retries"])
        if not (isinstance(opts, dict) and "retry_after" in opts) and "retry_after" in guardrail_config and guardrail_config["retry_after"] is not None:
            retry_after = float(guardrail_config["retry_after"])

    # Proxy-level defaults from router_settings (when guardrail does not set its own).
    # Uses cached settings to avoid importing proxy_server on every guardrail invocation.
    s = _get_router_settings()
    if s:
        if num_retries == DEFAULT_GUARDRAIL_NUM_RETRIES and s.get("guardrail_num_retries") is not None:
            num_retries = int(s["guardrail_num_retries"])
        if retry_after == DEFAULT_GUARDRAIL_RETRY_AFTER and s.get("guardrail_retry_after") is not None:
            retry_after = float(s["guardrail_retry_after"])

    return num_retries, retry_after
