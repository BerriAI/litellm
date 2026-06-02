"""SDK-free entrypoints for proxy-core call sites (auth, …).

Proxy code may run without the OpenTelemetry SDK installed, so it must not import
``litellm.integrations.otel.logger`` (which imports the SDK at module scope) at
module load. These wrappers resolve it lazily on first use and no-op when the SDK
is absent or V2 is not the active logger -- so a call site can wrap a request
phase or seed identity unconditionally.

The resolution is cached after the first attempt. A failed ``import`` is never
recorded in ``sys.modules``, so re-attempting it per request re-runs the full
import finder/loader machinery; on a proxy without the SDK that costs hundreds of
microseconds on every request. Caching the outcome makes the steady-state cost a
single attribute load.
"""

from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional

_resolved = False
_phase_span_impl: Optional[Callable[[str], Any]] = None
_seed_request_identity_impl: Optional[Callable[..., None]] = None


def _resolve() -> None:
    global _resolved, _phase_span_impl, _seed_request_identity_impl
    try:
        from litellm.integrations.otel.logger import (
            phase_span as _phase_span,
            seed_request_identity as _seed,
        )

        _phase_span_impl = _phase_span
        _seed_request_identity_impl = _seed
    except Exception:
        _phase_span_impl = None
        _seed_request_identity_impl = None
    _resolved = True


@contextmanager
def phase_span(name: str) -> "Iterator[Any]":
    """Run a request phase inside a live active span so its DB/service calls nest.

    Yields ``None`` (a plain no-op) when the OTel SDK is unavailable or V2 is not
    the active logger.
    """
    if not _resolved:
        _resolve()
    if _phase_span_impl is None:
        yield None
        return
    with _phase_span_impl(name) as span:
        yield span


def seed_request_identity(user_api_key_dict: Any, model: Any = None) -> None:
    """Seed request-identity Baggage at the auth boundary (no-op without V2)."""
    if not _resolved:
        _resolve()
    if _seed_request_identity_impl is None:
        return
    _seed_request_identity_impl(user_api_key_dict, model=model)
