"""Test-only fake identity provider for the e2e JWT suite. Never deploy it.

Run it next to the proxy (`uv run python tests/e2e/jwt_issuer.py`). On start it
generates one RSA signing key and keeps it for the life of the process, serving
the public half at `GET /.well-known/jwks.json` and signing whatever claims are
POSTed to `/token`. The proxy's `JWT_PUBLIC_KEY_URL` points at the JWKS URL, and
tests mint RS256 tokens by POSTing claims, so the private key never leaves this
process and no test holds it.

One key per process, rather than per pytest run, is what survives the proxy's
JWKS cache: the proxy caches the JWKS for `litellm_jwtauth.public_key_ttl`
(600s by default) and does not refetch on an unknown `kid`, so a key rotated
every run would be rejected until the cache expired. Restart the proxy whenever
you restart the issuer.

The mint endpoint takes no credential: anyone who can reach it gets a token the
proxy trusts. It therefore binds 127.0.0.1 only, must never be exposed beyond
loopback, and must only ever be trusted by a proxy under test.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final, Literal

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.utils import to_base64url_uint
from pydantic import BaseModel, RootModel, ValidationError

JWT_ISSUER_PORT_ENV: Final = "E2E_JWT_ISSUER_PORT"
DEFAULT_JWT_ISSUER_PORT: Final = 4190
LOOPBACK_HOST: Final = "127.0.0.1"
JWKS_PATH: Final = "/.well-known/jwks.json"
TOKEN_PATH: Final = "/token"
DEFAULT_TOKEN_LIFETIME_SECONDS: Final = 300

ClaimValue = str | int | float | bool | None | list[str]


def jwt_issuer_port() -> int:
    raw: Final = os.environ.get(JWT_ISSUER_PORT_ENV, "").strip()
    return int(raw) if raw else DEFAULT_JWT_ISSUER_PORT


def jwt_issuer_url() -> str:
    return f"http://{LOOPBACK_HOST}:{jwt_issuer_port()}"


class RsaJwk(BaseModel):
    kty: Literal["RSA"] = "RSA"
    alg: Literal["RS256"] = "RS256"
    use: Literal["sig"] = "sig"
    kid: str
    n: str
    e: str


class JwksDocument(BaseModel):
    keys: tuple[RsaJwk, ...]


class TokenRequest(RootModel[Mapping[str, ClaimValue]]):
    """The JSON body of POST /token: the claims to sign, verbatim. Nested objects
    are not supported; every value is a scalar or a list of strings."""


class MintedToken(BaseModel):
    token: str


class IssuerError(BaseModel):
    error: str


@dataclass(frozen=True, slots=True)
class SigningKey:
    kid: str
    private_pem: str
    jwk: RsaJwk


def generate_signing_key(kid: str | None = None) -> SigningKey:
    private_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    numbers: Final = private_key.public_key().public_numbers()
    resolved_kid: Final = kid if kid is not None else uuid.uuid4().hex
    return SigningKey(
        kid=resolved_kid,
        private_pem=private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode(),
        jwk=RsaJwk(
            kid=resolved_kid,
            n=to_base64url_uint(numbers.n).decode(),
            e=to_base64url_uint(numbers.e).decode(),
        ),
    )


def mint(
    key: SigningKey,
    claims: Mapping[str, ClaimValue],
    *,
    issuer: str,
    now: int,
    lifetime_seconds: int = DEFAULT_TOKEN_LIFETIME_SECONDS,
) -> str:
    """Sign `claims` as a compact RS256 JWT carrying `key.kid` in its header.
    `iss`, `iat`, and `exp` are filled in when absent and left alone when the
    caller sets them, so a test can mint an already-expired token."""
    payload: Final[dict[str, ClaimValue]] = {
        "iss": issuer,
        "iat": now,
        "exp": now + lifetime_seconds,
        **claims,
    }
    return jwt.encode(payload, key.private_pem, algorithm="RS256", headers={"kid": key.kid})


class _IssuerServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, bind: tuple[str, int], *, key: SigningKey, clock: Callable[[], int]) -> None:
        super().__init__(bind, _IssuerHandler)
        self.key: Final = key
        self.clock: Final = clock

    @property
    def url(self) -> str:
        host, port = self.server_address[0], self.server_address[1]
        return f"http://{host}:{port}"


class _IssuerHandler(BaseHTTPRequestHandler):
    def _issuer(self) -> _IssuerServer:
        issuer: Final = self.server
        assert isinstance(issuer, _IssuerServer)
        return issuer

    def do_GET(self) -> None:
        if self.path != JWKS_PATH:
            self._send(404, IssuerError(error=f"unknown path {self.path}; the JWKS is at {JWKS_PATH}"))
            return
        self._send(200, JwksDocument(keys=(self._issuer().key.jwk,)))

    def do_POST(self) -> None:
        if self.path != TOKEN_PATH:
            self._send(404, IssuerError(error=f"unknown path {self.path}; mint tokens at {TOKEN_PATH}"))
            return
        length: Final = int(self.headers.get("Content-Length", "0"))
        try:
            request: Final = TokenRequest.model_validate_json(self.rfile.read(length))
        except ValidationError as exc:
            self._send(400, IssuerError(error=f"claims must be a JSON object of scalar or string-list values: {exc}"))
            return
        issuer: Final = self._issuer()
        token: Final = mint(issuer.key, request.root, issuer=issuer.url, now=issuer.clock())
        self._send(200, MintedToken(token=token))

    def _send(self, status: int, body: BaseModel) -> None:
        payload: Final = body.model_dump_json().encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@dataclass(frozen=True, slots=True)
class RunningIssuer:
    url: str
    key: SigningKey
    server: _IssuerServer

    @property
    def jwks_url(self) -> str:
        return f"{self.url}{JWKS_PATH}"

    def shutdown(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def _wall_clock() -> int:
    return int(time.time())


def start_jwt_issuer(
    *,
    port: int = 0,
    key: SigningKey | None = None,
    clock: Callable[[], int] = _wall_clock,
) -> RunningIssuer:
    """Serve the issuer on loopback in a daemon thread. `port=0` takes an
    OS-assigned port for in-process tests; the CLI passes the documented one."""
    server: Final = _IssuerServer((LOOPBACK_HOST, port), key=key or generate_signing_key(), clock=clock)
    thread: Final = threading.Thread(target=server.serve_forever, name="e2e-jwt-issuer", daemon=True)
    thread.start()
    return RunningIssuer(url=server.url, key=server.key, server=server)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    running: Final = start_jwt_issuer(port=jwt_issuer_port())
    logging.getLogger(__name__).info(
        "e2e jwt issuer listening on %s  jwks=%s  mint=POST %s%s  kid=%s  (test-only, loopback, ctrl-c to stop)",
        running.url,
        running.jwks_url,
        running.url,
        TOKEN_PATH,
        running.key.kid,
    )
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        running.shutdown()


if __name__ == "__main__":
    main()
