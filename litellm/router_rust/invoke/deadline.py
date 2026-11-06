"""The caller's end-to-end budget on a proxy request.

The proxy turns a caller's timeout (the ``x-litellm-timeout`` header or a ``timeout`` body field)
into the per-attempt ``timeout`` and marks it ``client_side_timeout``. The budget it stands for
also covers the retries, their backoff sleeps and the fallbacks, so a fresh routed call pins it as
an absolute ``time.monotonic()`` instant and the step loop reads what is left of it."""

from collections.abc import Mapping
from typing import Final

REQUEST_DEADLINE: Final = "request_deadline"


def _seconds(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def with_request_deadline(
    prepared: Mapping[str, object], now: float
) -> dict[str, object]:  # mutable-ok: the request kwargs the router shapes in place
    """``prepared`` with the caller's budget pinned, when the proxy marked its timeout caller-controlled."""
    timeout: Final = _seconds(prepared.get("timeout")) or _seconds(prepared.get("request_timeout"))
    if prepared.get("client_side_timeout") is not True or timeout is None:
        return dict(prepared)
    return {**prepared, REQUEST_DEADLINE: now + timeout}


def seconds_left(kwargs: Mapping[str, object], now: float) -> float | None:
    """How much of the caller's budget is left, or ``None`` on a call without one."""
    deadline: Final = _seconds(kwargs.get(REQUEST_DEADLINE))
    return None if deadline is None else deadline - now


def outlives_deadline(kwargs: Mapping[str, object], now: float, seconds: float = 0.0) -> bool:
    """Whether a wait of ``seconds`` from ``now`` would end at or past the caller's deadline."""
    left: Final = seconds_left(kwargs, now)
    return left is not None and seconds >= left


def capped_to_deadline(timeout: object, kwargs: Mapping[str, object], now: float) -> object:
    """A per-attempt timeout, cut to what is left of the caller's budget."""
    left: Final = seconds_left(kwargs, now)
    configured: Final = _seconds(timeout)
    if left is None or configured is None:
        return timeout
    return min(configured, max(left, 0.0))
