from __future__ import annotations

import pytest
from pydantic import ValidationError

from litellm.proxy.auth_v2.models import (
    AuthMethod,
    Credential,
    EndUserIdentity,
    Principal,
    PrincipalType,
    ProjectIdentity,
    SecuritySchemeType,
    TeamIdentity,
    TeamRole,
    UserIdentity,
)
from litellm.proxy.auth_v2.authorization import Role


def _credential() -> Credential:
    return Credential(
        scheme=SecuritySchemeType.API_KEY,
        method=AuthMethod.API_KEY,
        subject="sk-test",
    )


def test_credential_is_frozen():
    credential = _credential()
    with pytest.raises(ValidationError):
        credential.subject = "mutated"


def test_credential_defaults_are_independent_instances():
    a = _credential()
    b = _credential()
    assert a.audience == [] and a.scopes == [] and a.claims == {}
    assert a.audience is not b.audience
    assert a.claims is not b.claims


def test_principal_requires_identity_core_fields():
    with pytest.raises(ValidationError):
        Principal(subject="u1")  # missing principal_type + auth_method


def test_principal_roles_are_validated_against_role_enum():
    principal = Principal(
        principal_type=PrincipalType.HUMAN,
        subject="u1",
        auth_method=AuthMethod.OIDC,
        roles=["org_admin"],
    )
    assert principal.roles == [Role.ORG_ADMIN]
    assert isinstance(principal.roles[0], Role)

    with pytest.raises(ValidationError):
        Principal(
            principal_type=PrincipalType.HUMAN,
            subject="u1",
            auth_method=AuthMethod.OIDC,
            roles=["not_a_real_role"],
        )


def test_principal_default_network_and_collections():
    principal = Principal(
        principal_type=PrincipalType.SERVICE_ACCOUNT,
        subject="svc",
        auth_method=AuthMethod.MUTUAL_TLS,
    )
    assert principal.teams == []
    assert principal.scopes == []
    assert principal.project is None
    assert principal.end_user is None
    assert principal.network.client_ip is None
    assert principal.network.via_trusted_proxy is False


def test_team_identity_defaults_to_member_role():
    team = TeamIdentity(id="g1")
    assert team.role == TeamRole.MEMBER


def test_security_scheme_values_match_openapi_spec():
    assert SecuritySchemeType.API_KEY.value == "apiKey"
    assert SecuritySchemeType.HTTP.value == "http"
    assert SecuritySchemeType.OAUTH2.value == "oauth2"
    assert SecuritySchemeType.OPENID_CONNECT.value == "openIdConnect"
    assert SecuritySchemeType.MUTUAL_TLS.value == "mutualTLS"


def test_user_identity_optional_fields_default_none():
    user = UserIdentity(id="u1")
    assert user.email is None
    assert user.external_id is None


def test_project_identity_name_is_optional():
    assert ProjectIdentity(id="p1").name is None
    assert ProjectIdentity(id="p1", name="Acme").name == "Acme"


def test_end_user_identity_requires_id():
    with pytest.raises(ValidationError):
        EndUserIdentity()
