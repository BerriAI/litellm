import hashlib
from collections.abc import Mapping, Sequence
from typing import Final

import pytest

from litellm.models.object_permission import LiteLLM_ObjectPermissionTable
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.mcp_server.events.access import EventGrant, grant_allows, intersect_grants, principal_of


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
        ((_level({"srv": ["a", "b"]}),), frozenset({"a", "b"})),
        ((_level({"srv": ["a", "b"]}), _level({"srv": ["b", "c"]})), frozenset({"b"})),
        ((_level({"srv": ["a"]}), _level({"other": ["z"]})), frozenset({"a"})),
        ((_level({"srv": []}),), frozenset()),
    ),
)
def test_event_grants_intersect_across_levels(
    levels: tuple[LiteLLM_ObjectPermissionTable | None, ...], expected: EventGrant
) -> None:
    assert intersect_grants(levels, "srv", _identity) == expected


@pytest.mark.parametrize(
    ("grant", "name", "allowed"),
    ((None, "a", True), (frozenset({"a"}), "a", True), (frozenset({"a"}), "b", False), (frozenset(), "a", False)),
)
def test_grant_allows(grant: EventGrant, name: str, allowed: bool) -> None:
    assert grant_allows(grant, name) is allowed
