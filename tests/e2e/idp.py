"""Provision isolated identities and obtain signed tokens from the test Keycloak realm."""

from __future__ import annotations

import base64
import os
import secrets
import signal
import subprocess
import sys
import warnings
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from types import FrameType
from typing import Final, Literal

import pytest
from e2e_http import (
    AuthHeaders,
    ExternalWrite,
    NetworkError,
    NoBody,
    Result,
    Success,
    UnknownApiError,
    delete_external,
    get_external,
    post_form_external,
    post_json_external,
    unwrap,
)
from pydantic import BaseModel, Field

KEYCLOAK_URL_ENV: Final = "E2E_KEYCLOAK_URL"
KEYCLOAK_REALM_ENV: Final = "E2E_KEYCLOAK_REALM"
KEYCLOAK_ADMIN_USER_ENV: Final = "E2E_KEYCLOAK_ADMIN_USER"
KEYCLOAK_ADMIN_PASSWORD_ENV: Final = "E2E_KEYCLOAK_ADMIN_PASSWORD"

DEFAULT_KEYCLOAK_URL: Final = "http://127.0.0.1:8480"
DEFAULT_REALM: Final = "litellm-e2e"
TESTS_CLIENT_ID: Final = "litellm-e2e-tests"
SHORT_LIVED_CLIENT_ID: Final = "litellm-e2e-shortlived"
ADMIN_CLIENT_ID: Final = "litellm-e2e-admin"
WRONG_AUDIENCE_CLIENT_ID: Final = "litellm-e2e-other-app"

_START_HINT: Final = (
    "Start it with the `docker run ... quay.io/keycloak/keycloak` command in tests/e2e/CONTRIBUTING.md, "
    f"and point {KEYCLOAK_URL_ENV} / {KEYCLOAK_ADMIN_USER_ENV} / {KEYCLOAK_ADMIN_PASSWORD_ENV} at it"
)


class TokenGrantForm(BaseModel):
    """The direct-access (password) grant an OAuth 2 token endpoint takes, form encoded."""

    grant_type: Literal["password"] = "password"
    client_id: str
    username: str
    password: str = Field(repr=False)
    client_secret: str | None = Field(default=None, repr=False)
    scope: str | None = None


class TokenResponse(BaseModel):
    access_token: str = Field(repr=False)


class TokenRequestHeaders(BaseModel):
    host: str | None = None


class GroupCreateBody(BaseModel):
    name: str


class PasswordCredential(BaseModel):
    type: Literal["password"] = "password"
    value: str = Field(repr=False)
    temporary: bool = False


class UserCreateBody(BaseModel):
    """Keycloak's admin representation of a new user. `firstName` / `lastName` and
    an empty `requiredActions` matter: a realm's default VERIFY_PROFILE action
    otherwise leaves the account "not fully set up" and every grant fails."""

    username: str
    email: str
    email_verified: bool = Field(default=True, alias="emailVerified")
    first_name: str = Field(default="E2E", alias="firstName")
    last_name: str = Field(default="Tester", alias="lastName")
    enabled: bool = True
    groups: tuple[str, ...]
    credentials: tuple[PasswordCredential, ...]
    required_actions: tuple[str, ...] = Field(default=(), alias="requiredActions")


def created_id(write: ExternalWrite, context: str) -> str:
    """The new resource's id, which Keycloak returns only as the last segment of
    the Location header on a 201."""
    if write.status_code != 201:
        pytest.fail(f"Keycloak refused to create {context}: HTTP {write.status_code} {write.body[:300]}")
    if not write.location or write.location.endswith("/"):
        pytest.fail(f"Keycloak created {context} without a resource id in its Location header")
    return write.location.rsplit("/", 1)[-1]


@dataclass(frozen=True, slots=True)
class Identity:
    """One provisioned IdP user: the `sub` the proxy will see, the credential the
    test signs in with, and the group whose name the litellm team carries."""

    user_id: str
    username: str
    password: str = field(repr=False)
    groups: tuple[str, ...]
    group_ids: tuple[str, ...]

    @property
    def group(self) -> str:
        if len(self.groups) != 1:
            raise ValueError("A single-group identity is required")
        return self.groups[0]

    @property
    def group_id(self) -> str:
        if len(self.group_ids) != 1:
            raise ValueError("A single-group identity is required")
        return self.group_ids[0]


@dataclass(frozen=True, slots=True)
class Keycloak:
    base_url: str
    realm: str
    admin_username: str
    admin_password: str = field(repr=False)
    strict_cleanup: bool = False

    def with_strict_cleanup(self) -> Keycloak:
        return replace(self, strict_cleanup=True)

    @property
    def issuer(self) -> str:
        return f"{self.base_url}/realms/{self.realm}"

    @property
    def jwks_url(self) -> str:
        return f"{self.issuer}/protocol/openid-connect/certs"

    def token_url(self, realm: str) -> str:
        return f"{self.base_url}/realms/{realm}/protocol/openid-connect/token"

    def _admin_url(self, path: str) -> str:
        return f"{self.base_url}/admin/realms/{self.realm}{path}"

    def _admin_headers(self) -> AuthHeaders:
        """A fresh admin token per call: the master realm's tokens are short lived,
        and a cached one would expire in the middle of a slow test."""
        form: Final = TokenGrantForm(client_id="admin-cli", username=self.admin_username, password=self.admin_password)
        result: Final = post_form_external(self.token_url("master"), form=form, response_type=TokenResponse)
        return AuthHeaders(authorization=f"Bearer {self._token(result, 'the Keycloak admin credential')}")

    def _token(self, result: Result[TokenResponse], context: str) -> str:
        match result:
            case Success(data=granted):
                return granted.access_token
            case NetworkError(message=message):
                return pytest.fail(f"No live Keycloak at {self.base_url} for {context}: {message}. {_START_HINT}")
            case _:
                return pytest.fail(f"Keycloak refused {context}: {result}")

    def create_group(self, name: str) -> str:
        return created_id(
            post_json_external(
                self._admin_url("/groups"), headers=self._admin_headers(), json=GroupCreateBody(name=name)
            ),
            f"group {name}",
        )

    def create_user(
        self, *, username: str, email: str, password: str, group: str | None = None, groups: tuple[str, ...] = ()
    ) -> str:
        return created_id(
            post_json_external(
                self._admin_url("/users"),
                headers=self._admin_headers(),
                json=UserCreateBody(
                    username=username,
                    email=email,
                    groups=(group,) if group is not None else groups,
                    credentials=(PasswordCredential(value=password),),
                ),
            ),
            f"user {username}",
        )

    def delete_user(self, user_id: str) -> None:
        self._delete(f"/users/{user_id}")

    def delete_group(self, group_id: str) -> None:
        self._delete(f"/groups/{group_id}")

    def assert_absent(self, kind: Literal["users", "groups", "clients"], resource_id: str) -> None:
        result: Final = get_external(
            self._admin_url(f"/{kind}/{resource_id}"),
            headers=self._admin_headers(),
            response_type=NoBody,
        )
        assert isinstance(result, UnknownApiError) and result.status_code == 404, (
            f"Owned IdP {kind} still exists: {result}"
        )

    def _delete(self, path: str) -> None:
        try:
            headers: Final = self._admin_headers()
        except pytest.fail.Exception as exc:
            if self.strict_cleanup:
                raise RuntimeError(f"Keycloak cleanup could not authenticate for {path}") from exc
            warnings.warn(f"Keycloak cleanup could not authenticate for {path}: {exc}", RuntimeWarning, stacklevel=2)
            return
        result: Final = delete_external(self._admin_url(path), headers=headers)
        if result.status_code not in (204, 404):
            if self.strict_cleanup:
                raise RuntimeError(f"Keycloak cleanup failed for {path}: HTTP {result.status_code}")
            warnings.warn(
                f"Keycloak cleanup failed for {path}: HTTP {result.status_code} {result.body[:300]}",
                RuntimeWarning,
                stacklevel=2,
            )

    def provision(self, *, marker: str, group: str, defer: Callable[[Callable[[], object]], None]) -> Identity:
        """Create `group` and a user in it, credentialed with a password generated
        for this test alone, and hand back the identity a token can be minted for."""
        return self.provision_groups(marker=marker, groups=(group,), defer=defer)

    def provision_groups(
        self, *, marker: str, groups: tuple[str, ...], defer: Callable[[Callable[[], object]], None]
    ) -> Identity:
        def provision_group(name: str) -> str:
            created: Final = self.create_group(name)
            defer(lambda: self.delete_group(created))
            return created

        group_ids: Final = tuple(provision_group(group) for group in groups)
        return self.provision_user(marker=marker, groups=groups, group_ids=group_ids, defer=defer)

    def provision_user(
        self,
        *,
        marker: str,
        groups: tuple[str, ...],
        group_ids: tuple[str, ...],
        defer: Callable[[Callable[[], object]], None],
    ) -> Identity:
        username: Final = f"e2e-jwt-user-{marker}"
        password: Final = secrets.token_urlsafe(24)
        user_id: Final = self.create_user(
            username=username, email=f"{username}@example.com", password=password, groups=groups
        )
        defer(lambda: self.delete_user(user_id))
        return Identity(user_id=user_id, username=username, password=password, groups=groups, group_ids=group_ids)

    def access_token(
        self, identity: Identity, *, client_id: str = TESTS_CLIENT_ID, issuer_host: str | None = None
    ) -> str:
        """Sign `identity` in through the direct-access grant and hand back the
        access token Keycloak signed, exactly as it came off the wire."""
        result: Final = post_form_external(
            self.token_url(self.realm),
            form=TokenGrantForm(client_id=client_id, username=identity.username, password=identity.password),
            response_type=TokenResponse,
            headers=TokenRequestHeaders(host=issuer_host),
        )
        return self._token(result, f"a token for {identity.username}")

    def discovery(self) -> Discovery:
        return unwrap(get_external(f"{self.issuer}/.well-known/openid-configuration", response_type=Discovery))

    def browser_client(self, *, callback_url: str, defer: Callable[[Callable[[], object]], None]) -> BrowserClient:
        client: Final = BrowserClient(
            client_id=f"e2e-browser-{secrets.token_hex(8)}",
            secret=secrets.token_urlsafe(32),
            callback_url=callback_url,
        )
        resource_id: Final = created_id(
            post_json_external(
                self._admin_url("/clients"),
                headers=self._admin_headers(),
                json=BrowserClientBody(
                    clientId=client.client_id,
                    secret=client.secret,
                    redirectUris=(callback_url,),
                ),
            ),
            "browser client",
        )
        defer(lambda: self._delete(f"/clients/{resource_id}"))
        configured: Final = unwrap(
            get_external(
                self._admin_url(f"/clients/{resource_id}"),
                headers=self._admin_headers(),
                response_type=BrowserClientBody,
            )
        )
        assert configured.redirect_uris == (callback_url,)
        assert configured.standard_flow_enabled and not configured.public_client
        assert configured.attributes.pkce == "S256"
        return client

    def browser_token(self, identity: Identity, client: BrowserClient) -> str:
        return self._token(
            post_form_external(
                self.token_url(self.realm),
                form=TokenGrantForm(
                    client_id=client.client_id,
                    client_secret=client.secret,
                    username=identity.username,
                    password=identity.password,
                    scope="openid email",
                ),
                response_type=TokenResponse,
            ),
            "browser-profile identity mapping",
        )

    def userinfo(self, token: str) -> UserInfo:
        return unwrap(
            get_external(
                f"{self.issuer}/protocol/openid-connect/userinfo",
                headers=AuthHeaders(authorization=f"Bearer {token}"),
                response_type=UserInfo,
            )
        )


def keycloak_from_env() -> Keycloak:
    admin_username: Final = os.environ.get(KEYCLOAK_ADMIN_USER_ENV, "").strip()
    admin_password: Final = os.environ.get(KEYCLOAK_ADMIN_PASSWORD_ENV, "").strip()
    if not admin_username or not admin_password:
        pytest.fail(
            f"The JWT suite needs {KEYCLOAK_ADMIN_USER_ENV} and {KEYCLOAK_ADMIN_PASSWORD_ENV} to provision "
            f"identities in its Keycloak realm, and neither may be empty. {_START_HINT}"
        )
    return Keycloak(
        base_url=os.environ.get(KEYCLOAK_URL_ENV, DEFAULT_KEYCLOAK_URL).rstrip("/"),
        realm=os.environ.get(KEYCLOAK_REALM_ENV, "").strip() or DEFAULT_REALM,
        admin_username=admin_username,
        admin_password=admin_password,
    )


class TokenClaims(BaseModel):
    sub: str
    iss: str
    aud: str | tuple[str, ...]
    exp: int
    scope: str = ""
    groups: tuple[str, ...] = ()


class Discovery(BaseModel):
    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    userinfo_endpoint: str
    jwks_uri: str


class UserInfo(BaseModel):
    sub: str
    email: str


class BrowserAttributes(BaseModel):
    pkce: str = Field(default="S256", alias="pkce.code.challenge.method")


class AudienceConfig(BaseModel):
    audience: str = Field(default="litellm-e2e", alias="included.custom.audience")
    access_token: str = Field(default="true", alias="access.token.claim")
    id_token: str = Field(default="false", alias="id.token.claim")


class AudienceMapper(BaseModel):
    name: str = "litellm-audience"
    protocol: str = "openid-connect"
    mapper: str = Field(default="oidc-audience-mapper", alias="protocolMapper")
    config: AudienceConfig = Field(default_factory=AudienceConfig)


class BrowserClientBody(BaseModel):
    client_id: str = Field(alias="clientId")
    secret: str = Field(repr=False)
    redirect_uris: tuple[str, ...] = Field(alias="redirectUris")
    enabled: bool = True
    public_client: bool = Field(default=False, alias="publicClient")
    standard_flow_enabled: bool = Field(default=True, alias="standardFlowEnabled")
    direct_access_grants_enabled: bool = Field(default=True, alias="directAccessGrantsEnabled")
    default_client_scopes: tuple[str, ...] = Field(default=("email", "basic"), alias="defaultClientScopes")
    attributes: BrowserAttributes = Field(default_factory=BrowserAttributes)
    protocol_mappers: tuple[AudienceMapper, ...] = Field(default=(AudienceMapper(),), alias="protocolMappers")


@dataclass(frozen=True, slots=True)
class BrowserClient:
    client_id: str
    secret: str = field(repr=False)
    callback_url: str

    def environment(self, discovery: Discovery) -> dict[str, str]:
        return {
            "GENERIC_CLIENT_ID": self.client_id,
            "GENERIC_CLIENT_SECRET": self.secret,
            "GENERIC_USER_ID_ATTRIBUTE": "sub",
            "GENERIC_AUTHORIZATION_ENDPOINT": discovery.authorization_endpoint,
            "GENERIC_TOKEN_ENDPOINT": discovery.token_endpoint,
            "GENERIC_USERINFO_ENDPOINT": discovery.userinfo_endpoint,
            "GENERIC_CLIENT_USE_PKCE": "true",
            "GENERIC_SCOPE": "openid email",
        }


def token_claims(token: str) -> TokenClaims:
    payload: Final = token.split(".")[1]
    return TokenClaims.model_validate_json(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))


def run_oidc_profile(proxy_url: str, command: list[str]) -> int:
    idp: Final = keycloak_from_env().with_strict_cleanup()
    with ExitStack() as cleanup:

        def terminate(signum: int, frame: FrameType | None) -> None:
            raise SystemExit(128 + signum)

        previous: Final = signal.signal(signal.SIGTERM, terminate)
        cleanup.callback(signal.signal, signal.SIGTERM, previous)

        def defer(callback: Callable[[], object]) -> None:
            cleanup.callback(callback)

        client: Final = idp.browser_client(callback_url=f"{proxy_url.rstrip('/')}/sso/callback", defer=defer)
        environment: Final = {**os.environ, **client.environment(idp.discovery()), "PROXY_BASE_URL": proxy_url}
        with subprocess.Popen(command, env=environment) as child:
            try:
                return child.wait()
            finally:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit("Usage: idp.py PROXY_URL COMMAND [ARG ...]; requires a running test IdP")
    raise SystemExit(run_oidc_profile(sys.argv[1], sys.argv[2:]))
