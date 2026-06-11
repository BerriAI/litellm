from __future__ import annotations

import secrets
import time
from typing import Any, Dict, Optional, Tuple

from fastapi import Request
from pydantic import BaseModel

from .models import AuthMethod, Credential, CredentialRef, SecuritySchemeType


class SessionConfig(BaseModel):
    cookie: str = "litellm_session"
    secure: bool = True
    ttl_seconds: int = 3600
    max_size: int = 10000
    default_redirect_path: str = "/"
    login_cookie: str = "litellm_oidc_txn"
    login_state_ttl: int = 300


def safe_relay_state(target: Optional[str], default: str) -> str:
    if (
        target
        and target.startswith("/")
        and not target.startswith("//")
        and "://" not in target
        and "\\" not in target
    ):
        return target
    return default


class SessionStore:
    def __init__(self, ttl_seconds: int = 3600, max_size: int = 10000) -> None:
        self._sessions: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._ttl = ttl_seconds
        self._max_size = max_size

    def create_session(self, identity: Dict[str, Any]) -> str:
        now = time.time()
        self._evict(now)
        session_id = secrets.token_urlsafe(32)
        self._sessions[session_id] = (now + self._ttl, identity)
        return session_id

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        entry = self._sessions.get(session_id)
        if entry is None:
            return None
        expires_at, identity = entry
        if expires_at < time.time():
            self._sessions.pop(session_id, None)
            return None
        return identity

    def pop(self, session_id: str) -> Optional[Dict[str, Any]]:
        entry = self._sessions.pop(session_id, None)
        if entry is None:
            return None
        expires_at, identity = entry
        if expires_at < time.time():
            return None
        return identity

    def _evict(self, now: float) -> None:
        for key in [k for k, (exp, _) in self._sessions.items() if exp < now]:
            self._sessions.pop(key, None)
        overflow = len(self._sessions) - self._max_size + 1
        if overflow > 0:
            oldest = sorted(self._sessions, key=lambda k: self._sessions[k][0])
            for key in oldest[:overflow]:
                self._sessions.pop(key, None)


class SessionAuthenticator:
    def __init__(self, cookie_name: str, store: SessionStore) -> None:
        self._cookie_name = cookie_name
        self._store = store

    async def authenticate(self, request: Request) -> Optional[Credential]:
        session_id = request.cookies.get(self._cookie_name)
        if not session_id:
            return None
        identity = self._store.get(session_id)
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

    def challenge(self) -> str:
        return ""
