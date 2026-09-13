"""Provision isolated identities and obtain signed tokens from the test Keycloak realm."""

from __future__ import annotations

import os
import secrets
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final, Literal

import pytest
from e2e_http import (
    AuthHeaders,
    ExternalWrite,
    NetworkError,
    Result,
    Success,
    delete_external,
    post_form_external,
    post_json_external,
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
    password: str


class TokenResponse(BaseModel):
    access_token: str = Field(repr=False)


class TokenRequestHeaders(BaseModel):
    host: str | None = None


class GroupCreateBody(BaseModel):
    name: str


class PasswordCredential(BaseModel):
    type: Literal["password"] = "password"
    value: str
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
    group: str
    group_id: str


@dataclass(frozen=True, slots=True)
class Keycloak:
    base_url: str
    realm: str
    admin_username: str
    admin_password: str = field(repr=False)

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

    def create_user(self, *, username: str, email: str, password: str, group: str) -> str:
        return created_id(
            post_json_external(
                self._admin_url("/users"),
                headers=self._admin_headers(),
                json=UserCreateBody(
                    username=username,
                    email=email,
                    groups=(group,),
                    credentials=(PasswordCredential(value=password),),
                ),
            ),
            f"user {username}",
        )

    def delete_user(self, user_id: str) -> None:
        self._delete(f"/users/{user_id}")

    def delete_group(self, group_id: str) -> None:
        self._delete(f"/groups/{group_id}")

    def _delete(self, path: str) -> None:
        try:
            headers: Final = self._admin_headers()
        except pytest.fail.Exception as exc:
            warnings.warn(f"Keycloak cleanup could not authenticate for {path}: {exc}", RuntimeWarning, stacklevel=2)
            return
        result: Final = delete_external(self._admin_url(path), headers=headers)
        if result.status_code not in (204, 404):
            warnings.warn(
                f"Keycloak cleanup failed for {path}: HTTP {result.status_code} {result.body[:300]}",
                RuntimeWarning,
                stacklevel=2,
            )

    def provision(self, *, marker: str, group: str, defer: Callable[[Callable[[], object]], None]) -> Identity:
        """Create `group` and a user in it, credentialed with a password generated
        for this test alone, and hand back the identity a token can be minted for."""
        group_id: Final = self.create_group(group)
        defer(lambda: self.delete_group(group_id))
        username: Final = f"e2e-jwt-user-{marker}"
        password: Final = secrets.token_urlsafe(24)
        user_id: Final = self.create_user(
            username=username, email=f"{username}@example.com", password=password, group=group
        )
        defer(lambda: self.delete_user(user_id))
        return Identity(user_id=user_id, username=username, password=password, group=group, group_id=group_id)

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
