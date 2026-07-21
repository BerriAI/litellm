"""Keycloak driver for the JWT-auth e2e suite: idempotent provisioning of the
realm/clients the gateway trusts, plus minting real RS256 access tokens from them.

The gateway validates JWTs against a real OIDC identity provider, so the suite
uses the Keycloak already in the e2e stack rather than a mock. This module talks
to Keycloak's admin + token endpoints over urllib (never requests - that is
reserved for e2e_http and enforced in CI), validating every response body through
pydantic so no untyped dict crosses the boundary.

Provisioning is idempotent (create, ignore "already exists"): a realm with an
admin client (a service account whose access token carries the
``litellm_proxy_admin`` scope), a team client (a service account whose token
carries a hardcoded ``team_id`` claim), and a short-lived client (tiny access-token
lifespan, admin scope) used to exercise the expiry-denied path.
"""

from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import cast

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from pydantic import BaseModel, TypeAdapter

ADMIN_SCOPE = "litellm_proxy_admin"
ADMIN_CLIENT = "litellm-admin"
TEAM_CLIENT = "litellm-team"
SHORTLIVED_CLIENT = "litellm-shortlived"
TEAM_CLAIM = "team_id"
JWT_TEAM_ID = "litellm-e2e-jwt-team"
SHORTLIVED_TOKEN_SECONDS = 10

_HTTP_TIMEOUT = 15


class _TokenResponse(BaseModel):
    access_token: str


class _SecretResponse(BaseModel):
    value: str


class _ClientEntry(BaseModel):
    id: str
    clientId: str  # noqa: N815 - Keycloak wire field


class _ScopeEntry(BaseModel):
    id: str
    name: str


_CLIENTS_ADAPTER: TypeAdapter[list[_ClientEntry]] = TypeAdapter(list[_ClientEntry])
_SCOPES_ADAPTER: TypeAdapter[list[_ScopeEntry]] = TypeAdapter(list[_ScopeEntry])


def mint_untrusted_jwt(issuer: str) -> str:
    """An RS256 JWT signed by a key the IdP's JWKS does not contain, carrying an
    otherwise-valid issuer, a future expiry, and admin-looking claims. The proxy
    routes it to the issuer's JWKS by `iss`, finds no key matching the token's
    `kid`, and must reject it - proving the gateway verifies the signature against
    the IdP rather than trusting the claims."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = int(time.time())
    return jwt.encode(
        {
            "iss": issuer,
            "sub": "untrusted-e2e",
            "team_id": JWT_TEAM_ID,
            "scope": ADMIN_SCOPE,
            "iat": now,
            "exp": now + 3600,
        },
        key,
        algorithm="RS256",
        headers={"kid": "untrusted-e2e-kid"},
    )


def _request(method: str, url: str, *, bearer: str | None = None, data: bytes | None = None, form: bool = False) -> bytes | None:
    """One urllib round-trip returning the raw response body (None for an empty body
    or a 409, so idempotent provisioning can ignore "already exists"). Any other
    4xx/5xx is a hard failure."""
    headers = {"Accept": "application/json"}
    if bearer is not None:
        headers["Authorization"] = f"Bearer {bearer}"
    if data is not None:
        headers["Content-Type"] = "application/x-www-form-urlencoded" if form else "application/json"
    request = urllib.request.Request(url, data=data, method=method, headers=headers)  # noqa: S310 - trusted local IdP
    try:
        opened = urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT)  # noqa: S310  # pyright: ignore[reportAny]  # typeshed types urlopen() as Any
        response = cast("http.client.HTTPResponse", opened)
        with response:
            return response.read() or None
    except urllib.error.HTTPError as exc:
        if exc.code == 409:
            return None
        detail = exc.read().decode(errors="replace")[:300]
        raise AssertionError(f"keycloak {method} {url} failed {exc.code}: {detail}") from exc


def _expect(body: bytes | None, context: str) -> bytes:
    if body is None:
        raise AssertionError(f"keycloak returned an empty body for {context}")
    return body


def _form(fields: dict[str, str]) -> bytes:
    return urllib.parse.urlencode(fields).encode()


def _token(base_url: str, realm: str, fields: dict[str, str]) -> str:
    body = _request(
        "POST", f"{base_url}/realms/{realm}/protocol/openid-connect/token", data=_form(fields), form=True
    )
    return _TokenResponse.model_validate_json(_expect(body, "token")).access_token


def _find_client_uuid(base_url: str, realm: str, token: str, client_id: str) -> str:
    query = urllib.parse.urlencode({"clientId": client_id})
    body = _request("GET", f"{base_url}/admin/realms/{realm}/clients?{query}", bearer=token)
    entries = _CLIENTS_ADAPTER.validate_json(_expect(body, f"client lookup {client_id}"))
    match = next((entry for entry in entries if entry.clientId == client_id), None)
    if match is None:
        raise AssertionError(f"keycloak client {client_id!r} not found after provisioning")
    return match.id


def _find_scope_id(base_url: str, realm: str, token: str, scope_name: str) -> str:
    body = _request("GET", f"{base_url}/admin/realms/{realm}/client-scopes", bearer=token)
    entries = _SCOPES_ADAPTER.validate_json(_expect(body, "client-scopes"))
    match = next((entry for entry in entries if entry.name == scope_name), None)
    if match is None:
        raise AssertionError(f"keycloak client scope {scope_name!r} not found after provisioning")
    return match.id


@dataclass(frozen=True, slots=True)
class KeycloakEnv:
    """What the suite needs: where the IdP is and how to mint tokens from each
    client. `issuer`/`jwks_url` are what the proxy's litellm_jwtauth.issuers must be
    configured with."""

    base_url: str
    realm: str
    admin_bootstrap_token: str

    @property
    def issuer(self) -> str:
        return f"{self.base_url}/realms/{self.realm}"

    @property
    def jwks_url(self) -> str:
        return f"{self.issuer}/protocol/openid-connect/certs"

    def admin_token(self) -> str:
        return self._client_credentials_token(ADMIN_CLIENT)

    def team_token(self) -> str:
        return self._client_credentials_token(TEAM_CLIENT)

    def shortlived_token(self) -> str:
        return self._client_credentials_token(SHORTLIVED_CLIENT)

    def _client_credentials_token(self, client_id: str) -> str:
        secret = self._client_secret(client_id)
        return _token(
            self.base_url,
            self.realm,
            {"client_id": client_id, "client_secret": secret, "grant_type": "client_credentials"},
        )

    def _client_secret(self, client_id: str) -> str:
        uuid = _find_client_uuid(self.base_url, self.realm, self.admin_bootstrap_token, client_id)
        body = _request(
            "GET",
            f"{self.base_url}/admin/realms/{self.realm}/clients/{uuid}/client-secret",
            bearer=self.admin_bootstrap_token,
        )
        return _SecretResponse.model_validate_json(_expect(body, f"client-secret {client_id}")).value


@dataclass(frozen=True, slots=True)
class KeycloakAdmin:
    base_url: str
    realm: str
    admin_user: str
    admin_password: str

    def provision(self) -> KeycloakEnv:
        token = _token(
            self.base_url,
            "master",
            {
                "client_id": "admin-cli",
                "username": self.admin_user,
                "password": self.admin_password,
                "grant_type": "password",
            },
        )
        self._ensure_realm(token)
        self._ensure_client_scope(token, ADMIN_SCOPE)
        self._ensure_service_client(token, ADMIN_CLIENT, default_scopes=(ADMIN_SCOPE,))
        self._ensure_service_client(token, TEAM_CLIENT)
        self._ensure_hardcoded_claim(token, TEAM_CLIENT, TEAM_CLAIM, JWT_TEAM_ID)
        self._ensure_service_client(
            token, SHORTLIVED_CLIENT, default_scopes=(ADMIN_SCOPE,), access_token_lifespan=SHORTLIVED_TOKEN_SECONDS
        )
        return KeycloakEnv(base_url=self.base_url, realm=self.realm, admin_bootstrap_token=token)

    def _ensure_realm(self, token: str) -> None:
        _ = _request(
            "POST",
            f"{self.base_url}/admin/realms",
            bearer=token,
            data=json.dumps({"realm": self.realm, "enabled": True}).encode(),
        )

    def _ensure_client_scope(self, token: str, name: str) -> None:
        _ = _request(
            "POST",
            f"{self.base_url}/admin/realms/{self.realm}/client-scopes",
            bearer=token,
            data=json.dumps(
                {
                    "name": name,
                    "protocol": "openid-connect",
                    "attributes": {"include.in.token.scope": "true", "display.on.consent.screen": "false"},
                }
            ).encode(),
        )

    def _ensure_service_client(
        self,
        token: str,
        client_id: str,
        *,
        default_scopes: tuple[str, ...] = (),
        access_token_lifespan: int | None = None,
    ) -> None:
        attributes = (
            {"access.token.lifespan": str(access_token_lifespan)} if access_token_lifespan is not None else {}
        )
        _ = _request(
            "POST",
            f"{self.base_url}/admin/realms/{self.realm}/clients",
            bearer=token,
            data=json.dumps(
                {
                    "clientId": client_id,
                    "enabled": True,
                    "protocol": "openid-connect",
                    "publicClient": False,
                    "serviceAccountsEnabled": True,
                    "standardFlowEnabled": False,
                    "directAccessGrantsEnabled": False,
                    "attributes": attributes,
                }
            ).encode(),
        )
        for scope in default_scopes:
            uuid = _find_client_uuid(self.base_url, self.realm, token, client_id)
            scope_id = _find_scope_id(self.base_url, self.realm, token, scope)
            _ = _request(
                "PUT",
                f"{self.base_url}/admin/realms/{self.realm}/clients/{uuid}/default-client-scopes/{scope_id}",
                bearer=token,
            )

    def _ensure_hardcoded_claim(self, token: str, client_id: str, claim: str, value: str) -> None:
        uuid = _find_client_uuid(self.base_url, self.realm, token, client_id)
        _ = _request(
            "POST",
            f"{self.base_url}/admin/realms/{self.realm}/clients/{uuid}/protocol-mappers/models",
            bearer=token,
            data=json.dumps(
                {
                    "name": claim,
                    "protocol": "openid-connect",
                    "protocolMapper": "oidc-hardcoded-claim-mapper",
                    "config": {
                        "claim.name": claim,
                        "claim.value": value,
                        "jsonType.label": "String",
                        "access.token.claim": "true",
                        "id.token.claim": "false",
                        "userinfo.token.claim": "false",
                    },
                }
            ).encode(),
        )
