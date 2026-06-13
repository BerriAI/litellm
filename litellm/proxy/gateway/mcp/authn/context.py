"""context — the single identity-context accessor for the gateway.

One ``ContextVar`` holds the resolved ``Subject`` for the duration of a request.
The transport wraps its dispatch in ``use_subject(...)``; downstream layers read
``current_subject()``. Mirrors the SDK's ``AuthContextMiddleware`` set/reset
discipline so the var is always restored, even on error.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar

from litellm.proxy.gateway.mcp.foundation import Subject

auth_context_var: ContextVar[Subject | None] = ContextVar("mcp_auth_context", default=None)


def current_subject() -> Subject | None:
    return auth_context_var.get()


@contextmanager
def use_subject(subject: Subject) -> Generator[None]:
    token = auth_context_var.set(subject)
    try:
        yield
    finally:
        auth_context_var.reset(token)
