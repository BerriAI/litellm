import hashlib
from collections.abc import Mapping, Sequence
from typing import Final

import pytest

from litellm.models.object_permission import LiteLLM_ObjectPermissionTable
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.mcp_server.events.access import (
    EventGrant,
    grant_allows,
    holds_webhooks,
    intersect_grants,
    principal_of,
    stored_key_hash,
)
from litellm.types.proxy.auth.user_api_key_auth import hash_token

from .fakes import JWT, admitted_subject


def _identity(permissions: Mapping[str, Sequence[str]]) -> Mapping[str, Sequence[str]]:
    return permissions


def _level(permissions: Mapping[str, Sequence[str]] | None) -> LiteLLM_ObjectPermissionTable:
    return LiteLLM_ObjectPermissionTable(object_permission_id="p", mcp_event_permissions=permissions)


def test_a_raw_key_and_its_hash_are_the_same_principal() -> None:
    raw: Final = "sk-example"
    hashed: Final = hashlib.sha256(raw.encode()).hexdigest()
    assert principal_of(UserAPIKeyAuth(api_key=raw)) == principal_of(UserAPIKeyAuth(api_key=hashed))


@pytest.mark.parametrize(
    ("caller", "expected_prefix"),
    ((UserAPIKeyAuth(api_key="sk-a", user_id="u"), "key:"), (UserAPIKeyAuth(user_id="u"), "user:"), (None, None)),
)
def test_principal_prefers_the_key_then_the_user(caller: UserAPIKeyAuth | None, expected_prefix: str | None) -> None:
    principal: Final = principal_of(caller)
    assert (principal.split(":")[0] + ":" if principal else None) == expected_prefix


def test_an_anonymous_caller_has_no_principal() -> None:
    assert principal_of(UserAPIKeyAuth()) is None


@pytest.mark.parametrize(
    ("levels", "expected"),
    (
        ((), None),
        ((None, _level(None)), None),
        ((_level({"other": ["x"]}),), None),
        ((_level({"srv": ["a", "b"]}),), (frozenset({"a", "b"}),)),
        ((_level({"srv": ["a", "b"]}), _level({"srv": ["b", "c"]})), (frozenset({"a", "b"}), frozenset({"b", "c"}))),
        ((_level({"srv": ["a"]}), _level({"other": ["z"]})), (frozenset({"a"}),)),
        ((_level({"srv": []}),), (frozenset(),)),
    ),
)
def test_event_grants_keep_every_level_that_restricts_the_server(
    levels: tuple[LiteLLM_ObjectPermissionTable | None, ...], expected: EventGrant
) -> None:
    assert intersect_grants(levels, "srv", _identity) == expected


@pytest.mark.parametrize(
    ("grant", "name", "allowed"),
    (
        (None, "a", True),
        ((frozenset({"a"}),), "a", True),
        ((frozenset({"a"}),), "b", False),
        ((frozenset(),), "a", False),
        ((frozenset({"build.*"}),), "build.failed", True),
        ((frozenset({"build.*"}),), "deploy.done", False),
        ((frozenset({"a", "b"}), frozenset({"b", "c"})), "b", True),
        ((frozenset({"a", "b"}), frozenset({"b", "c"})), "a", False),
        ((frozenset({"*"}), frozenset({"deploy.*"})), "build.failed", False),
    ),
)
def test_grant_allows(grant: EventGrant, name: str, allowed: bool) -> None:
    assert grant_allows(grant, name) is allowed


_JWT_A: Final = JWT
_JWT_B: Final = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1IiwiaWF0IjoyfQ.c2lnbmF0dXJlLWI"


@pytest.mark.parametrize(
    ("first", "second", "same"),
    (
        (UserAPIKeyAuth(api_key="sk-a"), UserAPIKeyAuth(api_key="sk-a", user_id="u"), True),
        (UserAPIKeyAuth(api_key=_JWT_A, user_id="u"), UserAPIKeyAuth(api_key=_JWT_B, user_id="u"), True),
        (UserAPIKeyAuth(user_id="u"), UserAPIKeyAuth(api_key=_JWT_A, user_id="u"), True),
        (UserAPIKeyAuth(api_key="sk-a"), UserAPIKeyAuth(api_key="sk-b"), False),
        (UserAPIKeyAuth(user_id="u1"), UserAPIKeyAuth(user_id="u2"), False),
        (UserAPIKeyAuth(api_key=_JWT_A, team_id="t1"), UserAPIKeyAuth(api_key=_JWT_B, team_id="t1"), True),
        (UserAPIKeyAuth(api_key=_JWT_A, team_id="t1"), UserAPIKeyAuth(api_key=_JWT_A, team_id="t2"), False),
        (UserAPIKeyAuth(api_key="custom-a"), UserAPIKeyAuth(api_key="custom-a"), True),
        (UserAPIKeyAuth(api_key="custom-a"), UserAPIKeyAuth(api_key="custom-b"), False),
    ),
)
def test_the_principal_survives_token_refresh_and_tracks_the_database_key(
    first: UserAPIKeyAuth, second: UserAPIKeyAuth, same: bool
) -> None:
    assert (principal_of(first) == principal_of(second)) is same


@pytest.mark.parametrize(
    ("caller", "key_hash"),
    (
        (UserAPIKeyAuth(api_key="sk-a", user_id="u"), hash_token("sk-a")),
        (UserAPIKeyAuth(api_key=_JWT_A, user_id="u"), None),
        (UserAPIKeyAuth(api_key="custom-raw-secret", user_id="u"), None),
        (UserAPIKeyAuth(user_id="u"), None),
        (None, None),
    ),
)
def test_only_database_keys_are_stored_as_key_hashes(caller: UserAPIKeyAuth | None, key_hash: str | None) -> None:
    assert stored_key_hash(caller) == key_hash


@pytest.mark.parametrize(
    "caller",
    (UserAPIKeyAuth(api_key="custom-a"), UserAPIKeyAuth(api_key=_JWT_A, team_id="t1"), UserAPIKeyAuth(user_id="u")),
)
def test_callers_without_a_database_key_never_share_the_anonymous_principal(caller: UserAPIKeyAuth) -> None:
    assert principal_of(caller) is not None
    assert principal_of(UserAPIKeyAuth(api_key=_JWT_A)) is None


@pytest.mark.parametrize(
    ("caller", "holds"),
    (
        (UserAPIKeyAuth(api_key="sk-a"), True),
        (admitted_subject("u"), True),
        (UserAPIKeyAuth(api_key=_JWT_A, user_id="u"), False),
        (UserAPIKeyAuth(api_key="custom-a", user_id="u"), False),
        (UserAPIKeyAuth(api_key=_JWT_A, team_id="t1"), False),
        (None, False),
    ),
)
def test_only_callers_the_worker_can_reauthorize_exactly_hold_webhooks(
    caller: UserAPIKeyAuth | None, holds: bool
) -> None:
    assert holds_webhooks(caller) is holds


_MASTER_HASH: Final = "f" * 64


def _master_hash() -> str:
    return _MASTER_HASH


@pytest.mark.parametrize(
    ("caller", "key_hash"),
    (
        (UserAPIKeyAuth(api_key="litellm_proxy_master_key", user_role=LitellmUserRoles.PROXY_ADMIN), _MASTER_HASH),
        (UserAPIKeyAuth(api_key="litellm_proxy_master_key", user_role=LitellmUserRoles.INTERNAL_USER), None),
        (UserAPIKeyAuth(api_key="sk-a", user_role=LitellmUserRoles.PROXY_ADMIN), hash_token("sk-a")),
    ),
)
def test_the_master_key_is_stored_as_its_own_hash_only_for_the_proxy_admin(
    caller: UserAPIKeyAuth, key_hash: str | None
) -> None:
    assert stored_key_hash(caller, _master_hash) == key_hash
