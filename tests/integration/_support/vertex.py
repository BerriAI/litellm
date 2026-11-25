from __future__ import annotations

import json
import time
from typing import Final

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

_PUBLIC_KEYS: Final[dict[str, list[rsa.RSAPublicKey]]] = {}


def service_account_json(project: str, token_url: str) -> str:
    key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    _PUBLIC_KEYS.setdefault(project, []).append(key.public_key())
    private_key: Final = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    return json.dumps(
        {
            "type": "service_account",
            "project_id": project,
            "private_key_id": "scripted",
            "private_key": private_key,
            "client_email": f"scripted@{project}.iam.gserviceaccount.com",
            "client_id": "0",
            "auth_uri": f"{token_url}/_oauth/authorize",
            "token_uri": f"{token_url}/_oauth/token",
        }
    )


def assert_vertex_authorization(
    authorization: str | None,
    project: str,
    *,
    oauth_token: str | None = "scripted-token",
    now: int | None = None,
) -> None:
    if oauth_token is not None and authorization == f"Bearer {oauth_token}":
        return
    assert isinstance(authorization, str) and authorization.startswith("Bearer "), authorization
    token: Final = authorization.removeprefix("Bearer ")
    assert jwt.get_unverified_header(token) == {"alg": "RS256", "typ": "JWT", "kid": "scripted"}
    for public_key in _PUBLIC_KEYS.get(project, []):
        try:
            claims = jwt.decode(
                token,
                public_key,
                algorithms=["RS256"],
                issuer=f"scripted@{project}.iam.gserviceaccount.com",
                options={"verify_exp": False, "verify_iat": False},
            )
        except jwt.InvalidSignatureError:
            continue
        assert claims.pop("aud", None) is None, claims
        assert set(claims) == {"iss", "sub", "scope", "iat", "exp"}, claims
        assert claims["sub"] == claims["iss"], claims
        assert claims["scope"] == "https://www.googleapis.com/auth/cloud-platform", claims
        assert type(claims["iat"]) is int and type(claims["exp"]) is int, claims
        assert claims["iat"] <= (time.time() if now is None else now) < claims["exp"], claims
        return
    raise AssertionError("Vertex bearer was not signed by this project's fixture service account")
