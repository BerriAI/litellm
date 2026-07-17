from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import jwt
from fastapi import Request

TEST_ISSUER = "https://idp.test.litellm.ai"
TEST_AUDIENCE = "litellm-proxy"


class _StaticSigningKey:
    def __init__(self, key: Any) -> None:
        self.key = key


class FakeJwksClient:
    """Stands in for PyJWKClient. Returns one fixed key for every token so
    JWTVerifier performs a real RS256 signature check against it via PyJWT."""

    def __init__(self, public_key: Any) -> None:
        self._public_key = public_key
        self.calls = 0

    def get_signing_key_from_jwt(self, token: str) -> _StaticSigningKey:
        self.calls += 1
        return _StaticSigningKey(self._public_key)


class TokenFactory:
    def __init__(self, private_pem: bytes) -> None:
        self._private_pem = private_pem

    def mint(
        self,
        *,
        issuer: str = TEST_ISSUER,
        audience: Any = TEST_AUDIENCE,
        subject: str = "user-1",
        expires_in: int = 3600,
        headers: Optional[Dict[str, Any]] = None,
        private_pem: Optional[bytes] = None,
        **extra_claims: Any,
    ) -> str:
        now = int(time.time())
        claims: Dict[str, Any] = {
            "iss": issuer,
            "aud": audience,
            "sub": subject,
            "iat": now,
            "exp": now + expires_in,
        }
        claims.update(extra_claims)
        return jwt.encode(
            claims,
            private_pem or self._private_pem,
            algorithm="RS256",
            headers=headers or {},
        )


def make_request(
    *,
    headers: Optional[Dict[str, str]] = None,
    cookies: Optional[Dict[str, str]] = None,
    client: Optional[Tuple[str, int]] = ("203.0.113.7", 5555),
    scope_extra: Optional[Dict[str, Any]] = None,
) -> Request:
    raw_headers: List[Tuple[bytes, bytes]] = []
    for key, value in (headers or {}).items():
        raw_headers.append((key.lower().encode(), value.encode()))
    if cookies:
        cookie_header = "; ".join(f"{k}={v}" for k, v in cookies.items())
        raw_headers.append((b"cookie", cookie_header.encode()))
    scope: Dict[str, Any] = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "headers": raw_headers,
        "client": client,
        "server": ("testserver", 80),
        "scheme": "http",
    }
    if scope_extra:
        scope.update(scope_extra)
    return Request(scope)
