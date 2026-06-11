from __future__ import annotations

from typing import Any, Dict, Optional, TypedDict


class SessionState(TypedDict):
    """A logged-in session, keyed by the session cookie's opaque id."""

    method: str
    subject: str
    issuer: Optional[str]
    claims: Dict[str, Any]


class OAuthTransaction(TypedDict):
    """In-flight OIDC authorization-code login, keyed by the login cookie's id."""

    provider: str
    state: str
    nonce: Optional[str]
    code_verifier: Optional[str]
    redirect_uri: str
    relay: str
