"""SDK-free entrypoints for proxy-core call sites (auth, …).

Proxy code may run without the OpenTelemetry SDK installed, so it must not import
``litellm.integrations.otel.logger`` (which imports the SDK at module scope) at
module load. These wrappers import it lazily and no-op when the SDK is absent or
V2 is not the active logger — so a call site can wrap a request phase or seed
identity unconditionally.
"""

from contextlib import contextmanager
from functools import cache
from typing import Any, Callable, Iterator, Optional


@cache
def _otel_runtime() -> "Optional[tuple[Callable[[str], Any], Callable[..., None]]]":
    """Resolve the SDK-backed hooks once and cache the outcome, absence included.

    CPython never caches a failed import, so without this memoization every call
    site re-attempts the import on each request; when the OTel SDK is not installed
    that re-scans ``sys.path`` and contends on the import lock on the hot path.
    """
    try:
        from litellm.integrations.otel import logger
    except Exception:
        return None
    return (logger.phase_span, logger.seed_request_identity)


@contextmanager
def phase_span(name: str) -> "Iterator[Any]":
    """Run a request phase inside a live active span so its DB/service calls nest.

    Yields ``None`` (a plain no-op) when the OTel SDK is unavailable or V2 is not
    the active logger.
    """
    runtime = _otel_runtime()
    if runtime is None:
        yield None
        return
    with runtime[0](name) as span:
        yield span


def seed_request_identity(user_api_key_dict: Any, model: Any = None) -> None:
    """Seed request-identity Baggage at the auth boundary (no-op without V2)."""
    runtime = _otel_runtime()
    if runtime is None:
        return
    runtime[1](user_api_key_dict, model=model)
