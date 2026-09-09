"""Harness coverage for idp.py: the pure parts of the Keycloak client, which are
the ones a wrong value in silently mistargets. No proxy and no IdP needed, so
these carry no `e2e` marker and run everywhere."""

from __future__ import annotations

from typing import Final

import pytest

from e2e_http import ExternalWrite
from idp import (
    KEYCLOAK_ADMIN_PASSWORD_ENV,
    KEYCLOAK_ADMIN_USER_ENV,
    KEYCLOAK_REALM_ENV,
    KEYCLOAK_URL_ENV,
    Keycloak,
    PasswordCredential,
    created_id,
    UserCreateBody,
    keycloak_from_env,
)

_REALM: Final = Keycloak(
    base_url="http://keycloak:8080", realm="litellm-e2e", admin_username="admin", admin_password="pw"
)


def test_realm_urls_match_keycloaks_own_layout() -> None:
    assert _REALM.issuer == "http://keycloak:8080/realms/litellm-e2e"
    assert _REALM.jwks_url == "http://keycloak:8080/realms/litellm-e2e/protocol/openid-connect/certs"
    assert _REALM.token_url("master") == "http://keycloak:8080/realms/master/protocol/openid-connect/token"


def test_created_id_is_the_last_segment_of_the_location_header() -> None:
    created: Final = ExternalWrite(
        status_code=201, location="http://keycloak:8080/admin/realms/litellm-e2e/groups/abc-123"
    )
    assert created_id(created, "a group") == "abc-123"


def test_a_refused_create_fails_the_test_with_the_idps_own_words() -> None:
    with pytest.raises(BaseException, match=r"409.*already exists"):
        created_id(ExternalWrite(status_code=409, body="Group already exists"), "a group")


def test_new_users_are_born_fully_set_up() -> None:
    """A user without a profile or with a pending required action authenticates
    nowhere: Keycloak answers every grant with "Account is not fully set up"."""
    body: Final = UserCreateBody(
        username="e2e", email="e2e@example.com", groups=("team",), credentials=(PasswordCredential(value="pw"),)
    ).model_dump(by_alias=True)

    assert body["requiredActions"] == ()
    assert body["firstName"] and body["lastName"] and body["emailVerified"] is True
    assert body["credentials"][0]["temporary"] is False


def test_connection_details_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEYCLOAK_URL_ENV, "http://keycloak.litellm.svc.cluster.local:8080/")
    monkeypatch.setenv(KEYCLOAK_REALM_ENV, "other-realm")
    monkeypatch.setenv(KEYCLOAK_ADMIN_USER_ENV, "admin")
    monkeypatch.setenv(KEYCLOAK_ADMIN_PASSWORD_ENV, "pw")

    resolved: Final = keycloak_from_env()

    assert resolved.issuer == "http://keycloak.litellm.svc.cluster.local:8080/realms/other-realm"
    assert resolved.admin_username == "admin" and resolved.admin_password == "pw"


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_missing_admin_credential_fails_loudly_instead_of_skipping(
    monkeypatch: pytest.MonkeyPatch, blank: str
) -> None:
    monkeypatch.setenv(KEYCLOAK_ADMIN_USER_ENV, "admin")
    monkeypatch.setenv(KEYCLOAK_ADMIN_PASSWORD_ENV, blank)

    with pytest.raises(BaseException, match=KEYCLOAK_ADMIN_PASSWORD_ENV):
        keycloak_from_env()
