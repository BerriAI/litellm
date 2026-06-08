import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

from litellm.constants import (
    LITELLM_INTERNAL_JOBS_SERVICE_ACCOUNT_NAME,
    LITTELM_CLI_SERVICE_ACCOUNT_NAME,
    LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME,
)
from litellm.identity import (
    AnonymousPrincipal,
    ApiKeyPrincipal,
    IdentityContext,
    JWTPrincipal,
    ServiceAccountPrincipal,
)
from litellm.identity.adapter import (
    identity_context_to_user_api_key_auth,
    user_api_key_auth_to_identity_context,
)
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth


def test_roundtrip_preserves_api_key_identity_fields():
    uak = UserAPIKeyAuth(
        api_key="sk-abc",
        user_id="u1",
        team_id="t1",
        org_id="o1",
        project_id="p1",
        agent_id="a1",
        key_alias="alias-1",
        end_user_id="eu-1",
        access_group_ids=["g1", "g2"],
    )
    ctx = user_api_key_auth_to_identity_context(uak)
    assert isinstance(ctx.principal, ApiKeyPrincipal)
    assert ctx.principal.user_id == "u1"
    assert ctx.principal.team_id == "t1"
    assert ctx.principal.org_id == "o1"
    assert ctx.principal.project_id == "p1"
    assert ctx.principal.agent_id == "a1"
    assert ctx.principal.key_alias == "alias-1"
    assert ctx.end_user_id == "eu-1"
    assert ctx.access_group_ids == ["g1", "g2"]

    back = identity_context_to_user_api_key_auth(ctx)
    assert back.user_id == "u1"
    assert back.team_id == "t1"
    assert back.org_id == "o1"
    assert back.project_id == "p1"
    assert back.agent_id == "a1"
    assert back.key_alias == "alias-1"
    assert back.end_user_id == "eu-1"
    assert back.access_group_ids == ["g1", "g2"]
    assert back.token == uak.token


def test_access_group_ids_empty_list_survives_roundtrip():
    uak = UserAPIKeyAuth(api_key="sk-x", user_id="u", access_group_ids=[])
    ctx = user_api_key_auth_to_identity_context(uak)
    assert ctx.access_group_ids == []

    back = identity_context_to_user_api_key_auth(ctx)
    assert back.access_group_ids == []


def test_token_hash_not_double_hashed():
    ctx = IdentityContext(principal=ApiKeyPrincipal(token_hash="abc123"))
    back = identity_context_to_user_api_key_auth(ctx)
    assert back.token == "abc123"


def test_jwt_principal_roundtrip():
    uak = UserAPIKeyAuth(
        api_key="aaaa.bbbb.cccc",
        user_id="jwt-user",
        team_id="jwt-team",
        org_id="jwt-org",
        jwt_claims={"sub": "jwt-user", "iss": "idp", "scope": "read write"},
    )
    ctx = user_api_key_auth_to_identity_context(uak)
    assert isinstance(ctx.principal, JWTPrincipal)
    assert ctx.principal.sub == "jwt-user"
    assert ctx.principal.iss == "idp"
    assert ctx.principal.scopes == ("read", "write")
    assert ctx.principal.mapped_user_id == "jwt-user"
    assert ctx.principal.mapped_team_id == "jwt-team"
    assert ctx.principal.mapped_org_id == "jwt-org"

    back = identity_context_to_user_api_key_auth(ctx)
    assert back.user_id == "jwt-user"
    assert back.team_id == "jwt-team"
    assert back.org_id == "jwt-org"
    assert back.jwt_claims is not None
    assert back.jwt_claims.get("sub") == "jwt-user"


def test_service_account_jobs_principal_roundtrip():
    uak = UserAPIKeyAuth.get_litellm_internal_jobs_user_api_key_auth()
    ctx = user_api_key_auth_to_identity_context(uak)
    assert isinstance(ctx.principal, ServiceAccountPrincipal)
    assert ctx.principal.name == LITELLM_INTERNAL_JOBS_SERVICE_ACCOUNT_NAME

    back = identity_context_to_user_api_key_auth(ctx)
    assert back.user_id == "system"
    assert back.team_id == "system"
    assert back.user_role == LitellmUserRoles.PROXY_ADMIN
    assert back.key_alias == LITELLM_INTERNAL_JOBS_SERVICE_ACCOUNT_NAME


def test_service_account_health_check_roundtrip():
    uak = UserAPIKeyAuth.get_litellm_internal_health_check_user_api_key_auth()
    ctx = user_api_key_auth_to_identity_context(uak)
    assert isinstance(ctx.principal, ServiceAccountPrincipal)
    assert ctx.principal.name == LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME

    back = identity_context_to_user_api_key_auth(ctx)
    assert back.team_id == LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME
    assert back.team_alias == LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME


def test_service_account_cli_roundtrip():
    uak = UserAPIKeyAuth.get_litellm_cli_user_api_key_auth()
    ctx = user_api_key_auth_to_identity_context(uak)
    assert isinstance(ctx.principal, ServiceAccountPrincipal)
    assert ctx.principal.name == LITTELM_CLI_SERVICE_ACCOUNT_NAME


def test_anonymous_principal_when_no_token():
    uak = UserAPIKeyAuth()
    ctx = user_api_key_auth_to_identity_context(uak)
    assert isinstance(ctx.principal, AnonymousPrincipal)
    back = identity_context_to_user_api_key_auth(ctx)
    assert back.token is None
    assert back.user_id is None


def test_end_user_does_not_leak_into_principal():
    ctx = IdentityContext(
        principal=ApiKeyPrincipal(token_hash="t", user_id="u"),
        end_user_id="customer-99",
    )
    back = identity_context_to_user_api_key_auth(ctx)
    assert back.user_id == "u"
    assert back.end_user_id == "customer-99"
    assert ctx.principal.user_id == "u"


def test_uak_methods_delegate_to_adapter():
    uak = UserAPIKeyAuth(api_key="sk-test", user_id="u", team_id="t")
    ctx = uak.to_identity_context()
    assert isinstance(ctx, IdentityContext)
    back = UserAPIKeyAuth.from_identity_context(ctx)
    assert back.user_id == "u"
    assert back.team_id == "t"
