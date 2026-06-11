from __future__ import annotations

import pytest

from litellm.auth_v2.errors import AuthError
from litellm.auth_v2.models import (
    AuthMethod,
    ClientCertificate,
    Credential,
    Principal,
    PrincipalType,
    SecuritySchemeType,
)
from litellm.auth_v2.rbac import Role
from litellm.auth_v2.resolver import InMemoryIdentityStore, _hash_api_key


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


async def test_self_describing_token_builds_principal_from_claims():
    store = InMemoryIdentityStore()
    credential = Credential(
        scheme=SecuritySchemeType.OPENID_CONNECT,
        method=AuthMethod.OIDC,
        subject="sub-42",
        issuer="https://idp",
        scopes=["models:read"],
        claims={
            "email": "dana@example.com",
            "preferred_username": "dana",
            "name": "Dana D",
            "groups": ["eng", "oncall"],
            "roles": ["org_admin", "bogus_role"],
        },
    )
    principal = await store.resolve(credential)
    assert principal.user.email == "dana@example.com"
    assert principal.user.user_name == "dana"
    assert [team.id for team in principal.teams] == ["eng", "oncall"]
    # invalid role strings are filtered out, valid ones become Role enums
    assert principal.roles == [Role.ORG_ADMIN]
    assert principal.scopes == ["models:read"]


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
