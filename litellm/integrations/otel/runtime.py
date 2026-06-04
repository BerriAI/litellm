"""SDK-free entrypoints for proxy-core call sites (auth, …).

Proxy code may run without the OpenTelemetry SDK installed, so it must not import
``litellm.integrations.otel.logger`` (which imports the SDK at module scope) at
module load. These wrappers import it lazily and no-op when the SDK is absent or
V2 is not the active logger — so a call site can wrap a request phase or seed
identity unconditionally.
"""

from contextlib import contextmanager
from typing import Any, Iterator


@contextmanager
def phase_span(name: str) -> "Iterator[Any]":
    """Run a request phase inside a live active span so its DB/service calls nest.

    Yields ``None`` (a plain no-op) when the OTel SDK is unavailable or V2 is not
    the active logger.
    """
    try:
        from litellm.integrations.otel.logger import phase_span as _phase_span
    except Exception:
        yield None
        return
    with _phase_span(name) as span:
        yield span


def seed_request_identity(user_api_key_dict: Any, model: Any = None) -> None:
    """Seed request-identity Baggage at the auth boundary (no-op without V2)."""
    try:
        from litellm.integrations.otel.logger import (
            seed_request_identity as _seed_request_identity,
        )
    except Exception:
        return
    _seed_request_identity(user_api_key_dict, model=model)
