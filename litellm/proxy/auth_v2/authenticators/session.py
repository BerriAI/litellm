from __future__ import annotations

from typing import Optional, Sequence

from fastapi import Request

from litellm.proxy.auth_v2.authenticators.base import (
    Authenticator,
    Carrier,
    CredentialLocation,
)
from litellm.proxy.auth_v2.models import (
    AuthMethod,
    Credential,
    CredentialRef,
    SecuritySchemeType,
)
from litellm.proxy.auth_v2.sessions import SessionStore
from litellm.proxy.auth_v2.sessions.types import SessionState


class SessionAuthenticator(Authenticator):
    def __init__(self, cookie_name: str, store: "SessionStore[SessionState]") -> None:
        self._cookie_name = cookie_name
        self._store = store

    async def authenticate(self, request: Request) -> Optional[Credential]:
        session_id = request.cookies.get(self._cookie_name)
        if not session_id:
            return None
        identity = await self._store.get(session_id)
        if identity is None:
            return None
        return Credential(
            scheme=SecuritySchemeType.API_KEY,
            method=AuthMethod(identity["method"]),
            subject=identity["subject"],
            issuer=identity.get("issuer"),
            claims=identity.get("claims", {}),
            credential_ref=CredentialRef(token_id=session_id),
        )

    def carriers(self) -> Sequence[Carrier]:
        return (Carrier(CredentialLocation.COOKIE, self._cookie_name),)

    def challenge(self) -> str:
        return ""
