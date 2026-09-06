"""Harness coverage for the test-only JWT issuer (jwt_issuer.py).

No proxy and no ``e2e`` marker. The issuer is booted in-process on an
OS-assigned port with a fixed clock and driven over HTTP through
``e2e_http.post_external`` / ``get_external``, the same transport the live
suite uses, so what is pinned here is the contract the live JWT tests lean on:
a token minted at ``/token`` verifies against the key served at
``/.well-known/jwks.json`` under the ``kid`` in its header, ``iss``/``iat``/``exp``
are filled in only when the caller left them out, and malformed claim bodies or
unknown paths are refused instead of signed.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Final

import jwt
import pytest
from pydantic import BaseModel, RootModel

from e2e_http import UnknownApiError, get_external, post_external, unwrap
from jwt_issuer import (
    DEFAULT_TOKEN_LIFETIME_SECONDS,
    JWKS_PATH,
    JWT_ISSUER_PORT_ENV,
    TOKEN_PATH,
    JwksDocument,
    MintedToken,
    RunningIssuer,
    jwt_issuer_port,
    start_jwt_issuer,
)

FROZEN_NOW: Final = int(time.time())


class _Claims(BaseModel):
    sub: str
    groups: tuple[str, ...] = ()
    exp: int | None = None


class _DecodedClaims(BaseModel):
    sub: str
    iss: str
    iat: int
    exp: int
    groups: tuple[str, ...] = ()


class _NotAnObject(RootModel[tuple[str, ...]]):
    pass


@pytest.fixture(scope="module")
def issuer() -> Iterator[RunningIssuer]:
    running: Final = start_jwt_issuer(clock=lambda: FROZEN_NOW)
    yield running
    running.shutdown()


def _mint(issuer: RunningIssuer, claims: _Claims) -> str:
    return unwrap(post_external(f"{issuer.url}{TOKEN_PATH}", json=claims, response_type=MintedToken)).token


def _served_jwks(issuer: RunningIssuer) -> JwksDocument:
    return unwrap(get_external(issuer.jwks_url, response_type=JwksDocument))


def _decode(token: str, jwks: JwksDocument, *, verify_exp: bool = True) -> _DecodedClaims:
    key: Final = jwt.PyJWK.from_json(jwks.keys[0].model_dump_json())
    decoded: Final = jwt.decode(token, key, algorithms=["RS256"], options={"verify_exp": verify_exp})
    return _DecodedClaims.model_validate(decoded)


class TestJwtIssuer:
    def test_minted_token_verifies_against_the_served_jwks(self, issuer: RunningIssuer) -> None:
        token: Final = _mint(issuer, _Claims(sub="alice", groups=("team-a",)))
        jwks: Final = _served_jwks(issuer)

        assert len(jwks.keys) == 1
        assert jwt.get_unverified_header(token)["kid"] == jwks.keys[0].kid
        claims: Final = _decode(token, jwks)
        assert claims.sub == "alice"
        assert claims.groups == ("team-a",)
        assert claims.iss == issuer.url
        assert claims.iat == FROZEN_NOW
        assert claims.exp == FROZEN_NOW + DEFAULT_TOKEN_LIFETIME_SECONDS

    def test_a_token_signed_by_another_key_does_not_verify(self, issuer: RunningIssuer) -> None:
        other: Final = start_jwt_issuer(clock=lambda: FROZEN_NOW)
        try:
            foreign_token: Final = _mint(other, _Claims(sub="alice"))
        finally:
            other.shutdown()

        with pytest.raises(jwt.InvalidSignatureError):
            _decode(foreign_token, _served_jwks(issuer))

    def test_an_explicit_exp_is_signed_as_given(self, issuer: RunningIssuer) -> None:
        expired_at: Final = FROZEN_NOW - 60
        token: Final = _mint(issuer, _Claims(sub="alice", exp=expired_at))
        jwks: Final = _served_jwks(issuer)

        assert _decode(token, jwks, verify_exp=False).exp == expired_at
        with pytest.raises(jwt.ExpiredSignatureError):
            _ = _decode(token, jwks)

    def test_non_object_claims_are_refused(self, issuer: RunningIssuer) -> None:
        result: Final = post_external(
            f"{issuer.url}{TOKEN_PATH}", json=_NotAnObject(("not", "claims")), response_type=MintedToken
        )
        assert isinstance(result, UnknownApiError)
        assert result.status_code == 400

    @pytest.mark.parametrize(
        ("method", "path"),
        [("GET", TOKEN_PATH), ("POST", JWKS_PATH), ("GET", "/token/anything")],
        ids=["get-token", "post-jwks", "get-other"],
    )
    def test_unknown_routes_are_404(self, issuer: RunningIssuer, method: str, path: str) -> None:
        url: Final = f"{issuer.url}{path}"
        result: Final = (
            get_external(url, response_type=MintedToken)
            if method == "GET"
            else post_external(url, json=_Claims(sub="alice"), response_type=MintedToken)
        )
        assert isinstance(result, UnknownApiError)
        assert result.status_code == 404


class TestIssuerPort:
    def test_defaults_to_the_documented_port(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(JWT_ISSUER_PORT_ENV, raising=False)
        assert jwt_issuer_port() == 4190, "CONTRIBUTING.md hardcodes 4190 in JWT_PUBLIC_KEY_URL"

    def test_env_override_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(JWT_ISSUER_PORT_ENV, " 4321 ")
        assert jwt_issuer_port() == 4321
