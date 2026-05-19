"""PKCESession — convenience helper for server-side OAuth flows.

Browser-driven flows belong in the TS SDK; on the Python side this is mostly
useful for backend services that do a server-to-server token fetch where
the human-in-the-middle has already authorized via a separate channel
(e.g. a Slack bot operator clicked Approve in the dashboard).
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import urllib.parse
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx


@dataclass
class PKCESession:
    """One-shot PKCE state. Keep an instance around between authorize → token.

    Attributes:
        client_id: OAuth client_id for the XCT app.
        redirect_uri: must EXACT-match the app's whitelist.
        code_verifier: 43-char URL-safe; auto-generated when omitted.
        state: opaque round-trip value; auto-generated when omitted.
    """

    client_id: str
    redirect_uri: str
    code_verifier: str = ""
    state: str = ""

    def __post_init__(self):
        if not self.code_verifier:
            self.code_verifier = _gen_verifier()
        if not self.state:
            self.state = secrets.token_urlsafe(16)

    @property
    def code_challenge(self) -> str:
        return _challenge(self.code_verifier)

    def authorize_url(self, base_url: str, *, scope: str = "") -> str:
        """The URL to send the user's browser to."""
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "state": self.state,
            "code_challenge": self.code_challenge,
            "code_challenge_method": "S256",
        }
        if scope:
            params["scope"] = scope
        return (
            f"{base_url.rstrip('/')}/oauth/authorize?{urllib.parse.urlencode(params)}"
        )

    def complete(
        self,
        base_url: str,
        *,
        code: str,
        client_secret: Optional[str] = None,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """Exchange code for tokens. Returns the /oauth/token JSON payload.

        Raises ``httpx.HTTPStatusError`` on non-2xx — let the caller decide
        how to surface auth failures.
        """
        data = {
            "grant_type": "authorization_code",
            "client_id": self.client_id,
            "code": code,
            "code_verifier": self.code_verifier,
            "redirect_uri": self.redirect_uri,
        }
        if client_secret:
            data["client_secret"] = client_secret
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(f"{base_url.rstrip('/')}/oauth/token", data=data)
            resp.raise_for_status()
            return resp.json()

    async def acomplete(
        self,
        base_url: str,
        *,
        code: str,
        client_secret: Optional[str] = None,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        data = {
            "grant_type": "authorization_code",
            "client_id": self.client_id,
            "code": code,
            "code_verifier": self.code_verifier,
            "redirect_uri": self.redirect_uri,
        }
        if client_secret:
            data["client_secret"] = client_secret
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{base_url.rstrip('/')}/oauth/token", data=data)
            resp.raise_for_status()
            return resp.json()


def _gen_verifier() -> str:
    """RFC 7636 §4.1: 43-128 char URL-safe verifier; 32 bytes ≈ 43 chars."""
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("=")


def _challenge(verifier: str) -> str:
    return (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .decode()
        .rstrip("=")
    )
