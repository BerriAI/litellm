from __future__ import annotations

from typing import Dict, Optional

import pytest

from litellm.proxy._types import LiteLLM_UserTable, UserAPIKeyAuth, hash_token
from litellm.proxy.auth_v2.authorization import Role
from litellm.proxy.auth_v2.errors import AuthError
from litellm.proxy.auth_v2.models import (
    AuthMethod,
    ClientCertificate,
    Credential,
    PrincipalType,
    SecuritySchemeType,
)
from litellm.proxy.auth_v2.resolvers import DbResolver


class _FakeCache:
    """Stands in for the DualCache that get_key_object / get_user_object read.

    Both helpers return a cache hit before touching the DB, so seeding this and
    injecting it into DbResolver exercises the real resolver mapping without
    a database. A non-None prisma client is still required (the helpers guard on
    it); it is never reached on a hit.
    """

    def __init__(self, entries: Optional[Dict[str, object]] = None) -> None:
        self._entries = entries or {}

    async def async_get_cache(self, key, *args, **kwargs):
        return self._entries.get(key)

    async def async_set_cache(self, *args, **kwargs):
        return None


_PRISMA_STUB = object()


def _store(entries: Optional[Dict[str, object]] = None) -> DbResolver:
    return DbResolver(_PRISMA_STUB, _FakeCache(entries))


def _api_key_credential(raw: str) -> Credential:
    return Credential(
        scheme=SecuritySchemeType.API_KEY,
        method=AuthMethod.API_KEY,
        subject=raw,
        claims={"_raw_api_key": raw},
    )


def _oidc_credential(subject: str) -> Credential:
    return Credential(
        scheme=SecuritySchemeType.OPENID_CONNECT,
        method=AuthMethod.OIDC,
        subject=subject,
        issuer="https://idp",
        scopes=["models:read"],
        claims={"email": "dana@example.com"},
    )


async def test_api_key_resolves_to_principal_with_db_role():
    raw = "sk-live-abc"
    key = UserAPIKeyAuth(token=hash_token(raw), user_id="u-1", user_role="org_admin")
    store = _store({hash_token(raw): key})

    principal = await store.resolve(_api_key_credential(raw))

    assert principal.principal_type == PrincipalType.HUMAN
    assert principal.subject == "u-1"
    assert principal.user is not None and principal.user.id == "u-1"
    # role comes from the key's user_role mapped through the DB role map
    assert principal.roles == [Role.ORG_ADMIN]


async def test_api_key_principal_carries_project_and_end_user():
    raw = "sk-live-proj"
    key = UserAPIKeyAuth(
        token=hash_token(raw),
        user_id="u-1",
        project_id="proj-1",
        project_alias="Acme Prod",
        end_user_id="cust-7",
    )
    store = _store({hash_token(raw): key})

    principal = await store.resolve(_api_key_credential(raw))

    assert principal.project is not None
    assert principal.project.id == "proj-1"
    assert principal.project.name == "Acme Prod"
    assert principal.end_user is not None
    assert principal.end_user.id == "cust-7"


async def test_api_key_principal_omits_project_and_end_user_when_absent():
    raw = "sk-live-bare"
    key = UserAPIKeyAuth(token=hash_token(raw), user_id="u-1")
    store = _store({hash_token(raw): key})

    principal = await store.resolve(_api_key_credential(raw))

    assert principal.project is None
    assert principal.end_user is None


async def test_api_key_lookup_is_keyed_on_hashed_token():
    raw = "sk-live-abc"
    key = UserAPIKeyAuth(token=hash_token(raw), user_id="u-1")
    # cache seeded under the RAW key, not its hash -> resolver hashes first -> miss
    store = DbResolver(None, _FakeCache({raw: key}))
    with pytest.raises(AuthError) as exc:
        await store.resolve(_api_key_credential(raw))
    assert exc.value.status_code == 401


async def test_blocked_key_is_rejected_403():
    raw = "sk-live-blocked"
    key = UserAPIKeyAuth(token=hash_token(raw), user_id="u-1", blocked=True)
    store = _store({hash_token(raw): key})

    with pytest.raises(AuthError) as exc:
        await store.resolve(_api_key_credential(raw))
    assert exc.value.status_code == 403


async def test_unknown_key_is_rejected_401():
    # cache miss + no prisma -> get_key_object raises -> resolver maps to 401
    store = DbResolver(None, _FakeCache())
    with pytest.raises(AuthError) as exc:
        await store.resolve(_api_key_credential("sk-live-unknown"))
    assert exc.value.status_code == 401


async def test_subject_resolves_to_user_principal():
    user = LiteLLM_UserTable(
        user_id="u-9",
        user_role="org_admin",
        user_email="dana@example.com",
        sso_user_id="ext-9",
        user_alias="Dana",
        teams=[],
    )
    store = _store({"u-9": user})

    principal = await store.resolve(_oidc_credential("u-9"))

    assert principal.principal_type == PrincipalType.HUMAN
    assert principal.user is not None
    assert principal.user.id == "u-9"
    assert principal.user.email == "dana@example.com"
    assert principal.user.external_id == "ext-9"
    assert principal.roles == [Role.ORG_ADMIN]
    assert principal.scopes == ["models:read"]


async def test_mtls_credential_resolves_to_service_account():
    credential = Credential(
        scheme=SecuritySchemeType.MUTUAL_TLS,
        method=AuthMethod.MUTUAL_TLS,
        subject="CN=svc-a,O=Co",
        client_certificate=ClientCertificate(subject_dn="CN=svc-a,O=Co"),
    )
    # service-account path does no identity lookup, so no cache/prisma needed
    principal = await DbResolver(None, _FakeCache()).resolve(credential)

    assert principal.principal_type == PrincipalType.SERVICE_ACCOUNT
    assert principal.user is None
    assert principal.subject == "CN=svc-a,O=Co"
