from __future__ import annotations

import pytest
from scim2_models import Group as ScimGroup
from scim2_models import User as ScimUser

from litellm.proxy.auth_v2.errors import AuthError
from litellm.proxy.auth_v2.models import (
    AuthMethod,
    ClientCertificate,
    Credential,
    Principal,
    PrincipalType,
    SecuritySchemeType,
    UserIdentity,
)
from litellm.proxy.auth_v2.rbac import Role
from litellm.proxy.auth_v2.resolver import InMemoryIdentityStore, _hash_api_key


def _api_key_credential(raw: str) -> Credential:
    return Credential(
        scheme=SecuritySchemeType.API_KEY,
        method=AuthMethod.API_KEY,
        subject=raw,
        claims={"_raw_api_key": raw},
    )


def _principal(subject: str = "user-1") -> Principal:
    return Principal(
        principal_type=PrincipalType.HUMAN,
        subject=subject,
        auth_method=AuthMethod.API_KEY,
    )


async def test_api_key_resolves_only_for_exact_key():
    raw = "sk-live-correct-horse"
    store = InMemoryIdentityStore(api_keys={_hash_api_key(raw): _principal("svc-a")})

    resolved = await store.resolve(_api_key_credential(raw))
    assert resolved.subject == "svc-a"


async def test_wrong_api_key_never_resolves():
    raw = "sk-live-correct-horse"
    store = InMemoryIdentityStore(api_keys={_hash_api_key(raw): _principal()})
    with pytest.raises(AuthError) as exc:
        await store.resolve(_api_key_credential("sk-live-wrong-key"))
    assert exc.value.status_code == 401


async def test_api_key_lookup_is_keyed_on_sha256_not_raw():
    raw = "sk-live-correct-horse"
    # store keyed by the raw value (not its hash) must NOT resolve: resolver hashes first
    store = InMemoryIdentityStore(api_keys={raw: _principal()})
    with pytest.raises(AuthError):
        await store.resolve(_api_key_credential(raw))


async def test_missing_raw_api_key_claim_is_rejected():
    store = InMemoryIdentityStore(api_keys={})
    credential = Credential(
        scheme=SecuritySchemeType.API_KEY,
        method=AuthMethod.API_KEY,
        subject="sk-x",
    )
    with pytest.raises(AuthError):
        await store.resolve(credential)


async def test_subject_lookup_prefers_stored_principal():
    stored = _principal("from-store")
    store = InMemoryIdentityStore(subjects={"https://idp|sub-9": stored})
    credential = Credential(
        scheme=SecuritySchemeType.OPENID_CONNECT,
        method=AuthMethod.OIDC,
        subject="sub-9",
        issuer="https://idp",
    )
    resolved = await store.resolve(credential)
    assert resolved.subject == "from-store"


def _oidc_credential(**claims) -> Credential:
    return Credential(
        scheme=SecuritySchemeType.OPENID_CONNECT,
        method=AuthMethod.OIDC,
        subject="sub-42",
        issuer="https://idp",
        scopes=["models:read"],
        claims={
            "email": "dana@example.com",
            "preferred_username": "dana",
            "name": "Dana D",
            **claims,
        },
    )


async def test_self_describing_token_builds_principal_from_claims():
    store = InMemoryIdentityStore()
    principal = await store.resolve(_oidc_credential(roles=["org_admin", "bogus_role"]))
    assert principal.user.email == "dana@example.com"
    assert principal.user.user_name == "dana"
    # invalid role strings are filtered out, valid ones become Role enums
    assert principal.roles == [Role.ORG_ADMIN]
    assert principal.scopes == ["models:read"]


async def test_group_claim_without_provisioned_scim_group_is_not_a_team():
    # H1: a token group claim is not authoritative on its own
    store = InMemoryIdentityStore()
    principal = await store.resolve(_oidc_credential(groups=["eng", "oncall"]))
    assert principal.teams == []


async def test_group_claim_becomes_team_only_when_provisioned():
    store = InMemoryIdentityStore(
        groups={"eng": ScimGroup(id="eng", display_name="Engineering")}
    )
    principal = await store.resolve(_oidc_credential(groups=["eng", "unprovisioned"]))
    # only the provisioned group resolves to a team; the unknown one is dropped
    assert len(principal.teams) == 1
    assert principal.teams[0].id == "eng"
    assert principal.teams[0].name == "Engineering"


async def test_mtls_credential_resolves_to_service_account():
    store = InMemoryIdentityStore()
    credential = Credential(
        scheme=SecuritySchemeType.MUTUAL_TLS,
        method=AuthMethod.MUTUAL_TLS,
        subject="CN=svc-a,O=Co",
        client_certificate=ClientCertificate(subject_dn="CN=svc-a,O=Co"),
    )
    principal = await store.resolve(credential)
    assert principal.principal_type == PrincipalType.SERVICE_ACCOUNT
    assert principal.user is None
    assert principal.subject == "CN=svc-a,O=Co"


# --------------------------------------------------------------------------- #
# Deactivated users (M1) and claims scrubbing
# --------------------------------------------------------------------------- #


async def test_deactivated_user_is_rejected():
    principal = Principal(
        principal_type=PrincipalType.HUMAN,
        subject="u-1",
        auth_method=AuthMethod.API_KEY,
        user=UserIdentity(id="u-1", email="u@example.com"),
    )
    store = InMemoryIdentityStore(
        api_keys={_hash_api_key("sk-deact"): principal},
        users={"u-1": ScimUser(id="u-1", user_name="u@example.com", active=False)},
    )
    with pytest.raises(AuthError) as exc:
        await store.resolve(_api_key_credential("sk-deact"))
    assert exc.value.status_code == 403


async def test_active_user_is_allowed():
    principal = Principal(
        principal_type=PrincipalType.HUMAN,
        subject="u-2",
        auth_method=AuthMethod.API_KEY,
        user=UserIdentity(id="u-2", email="ok@example.com"),
    )
    store = InMemoryIdentityStore(
        api_keys={_hash_api_key("sk-ok"): principal},
        users={"u-2": ScimUser(id="u-2", user_name="ok@example.com", active=True)},
    )
    resolved = await store.resolve(_api_key_credential("sk-ok"))
    assert resolved.subject == "u-2"


async def test_principal_claims_scrub_underscore_keys():
    # internal underscore-prefixed claims (e.g. _raw_api_key) must never surface
    # on the Principal built from a self-describing credential
    store = InMemoryIdentityStore()
    credential = Credential(
        scheme=SecuritySchemeType.OPENID_CONNECT,
        method=AuthMethod.OIDC,
        subject="sub-x",
        issuer="https://idp",
        claims={"_raw_api_key": "leak", "_basic_password": "leak", "email": "e@x.com"},
    )
    principal = await store.resolve(credential)
    assert "_raw_api_key" not in principal.claims
    assert "_basic_password" not in principal.claims
    assert principal.claims.get("email") == "e@x.com"
