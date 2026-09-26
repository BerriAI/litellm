"""Marks work the proxy does after handing the response to the caller.

Success logging, dispatched (fire-and-forget) failure logging and the response
cache write run as background tasks, so whether they finish before or after the
server span closes is a race. Tracing keys off this marker, not the clock.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Final

_post_response_phase: Final[ContextVar[bool]] = ContextVar("litellm_post_response_phase", default=False)


def in_post_response_phase() -> bool:
    return _post_response_phase.get()


@contextmanager
def post_response_phase() -> Iterator[None]:
    token: Final = _post_response_phase.set(True)
    try:
        yield
    finally:
        _post_response_phase.reset(token)
