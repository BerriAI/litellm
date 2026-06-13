"""resolver — the identity seam: bearer -> Subject.

S1 ships a stub that ignores the bearer and returns an anonymous ``Subject``.
S6 replaces the body of ``resolve_subject`` with real virtual-key / JWT / OAuth
resolution; the signature stays put so nothing downstream has to change.
"""

from __future__ import annotations

from litellm.proxy.gateway.mcp.foundation import Subject

_ANONYMOUS = "anonymous"


def anonymous_subject() -> Subject:
    return Subject(subject_id=_ANONYMOUS, tenant=_ANONYMOUS)


def resolve_subject(bearer: str | None) -> Subject:
    """Resolve the caller identity from a bearer token.

    S6 replaces this body with real virtual-key/JWT/oauth resolution; today it
    ignores the bearer and returns the anonymous subject.
    """
    _ = bearer
    return anonymous_subject()
