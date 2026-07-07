from __future__ import annotations

from typing import Dict, Optional

import pytest
from prisma import Json
from scim2_models import Group as ScimGroup
from scim2_models import GroupMember as ScimGroupMember

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


async def test_api_key_role_falls_back_to_owning_user():
    # get_key_object does not join the user's role onto the token, so a key with a
    # user_id but no user_role must resolve the role from the user table
    raw = "sk-live-noroll"
    key = UserAPIKeyAuth(token=hash_token(raw), user_id="u-7")
    user = LiteLLM_UserTable(user_id="u-7", user_role="proxy_admin")
    store = _store({hash_token(raw): key, "u-7": user})

    principal = await store.resolve(_api_key_credential(raw))

    assert principal.roles == [Role.PLATFORM_ADMIN]


async def test_api_key_role_fails_closed_when_owning_user_unresolvable():
    # key carries a user_id but no user_role, and the user cannot be resolved
    # (cache miss + unusable prisma stub): the role lookup must fail closed to no
    # role rather than raising or inheriting one
    raw = "sk-live-orphan"
    key = UserAPIKeyAuth(token=hash_token(raw), user_id="u-missing")
    store = _store({hash_token(raw): key})

    principal = await store.resolve(_api_key_credential(raw))

    assert principal.user is not None and principal.user.id == "u-missing"
    assert principal.roles == []


async def test_service_account_key_without_user_has_no_role():
    # a key with no user_id never consults the user table and stays role-less
    raw = "sk-live-svc"
    key = UserAPIKeyAuth(token=hash_token(raw), key_alias="ci-bot")
    store = _store({hash_token(raw): key})

    principal = await store.resolve(_api_key_credential(raw))

    assert principal.principal_type == PrincipalType.SERVICE_ACCOUNT
    assert principal.roles == []


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


class _FakeTeamTable:
    """Minimal stand-in for the prisma team table.

    Mirrors the one prisma behavior the group methods depend on: a Json-wrapped
    write is stored as its plain Python value and read back the same way. Storing
    the raw value would let a regression that forgets the Json wrapper pass here
    while failing against a real database.
    """

    def __init__(self) -> None:
        self.rows: Dict[str, dict] = {}

    @staticmethod
    def _norm(data: dict) -> dict:
        return {k: (v.data if isinstance(v, Json) else v) for k, v in data.items()}

    async def find_unique(self, where):
        return self.rows.get(where["team_id"])

    async def create(self, data):
        row = self._norm(data)
        self.rows[row["team_id"]] = row
        return row

    async def update(self, where, data):
        self.rows[where["team_id"]].update(self._norm(data))
        return self.rows[where["team_id"]]

    async def find_many(self, **kwargs):
        return list(self.rows.values())

    async def delete(self, where):
        return self.rows.pop(where["team_id"], None)


class _FakeTeamPrisma:
    def __init__(self) -> None:
        self.db = type("_Db", (), {"litellm_teamtable": _FakeTeamTable()})()


async def test_group_upsert_round_trips_members_through_json():
    store = DbResolver(_FakeTeamPrisma(), _FakeCache())
    created = await store.upsert_group(
        ScimGroup(
            display_name="eng",
            members=[ScimGroupMember(value="u-1"), ScimGroupMember(value="u-2")],
        )
    )

    assert created.id is not None
    assert created.display_name == "eng"
    assert [m.value for m in created.members] == ["u-1", "u-2"]

    fetched = await store.get_group(created.id)
    assert fetched is not None
    assert [m.value for m in fetched.members] == ["u-1", "u-2"]


async def test_group_list_and_delete():
    store = DbResolver(_FakeTeamPrisma(), _FakeCache())
    a = await store.upsert_group(ScimGroup(display_name="a"))
    await store.upsert_group(ScimGroup(display_name="b"))

    assert {g.display_name for g in await store.list_groups(None)} == {"a", "b"}

    await store.delete_group(a.id)
    assert await store.get_group(a.id) is None
    assert {g.display_name for g in await store.list_groups(None)} == {"b"}
