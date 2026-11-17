"""A team or key in an access group that grants stores can use those stores and no others."""

import base64
from collections.abc import Sequence
from typing import Final, Literal
from unittest.mock import AsyncMock

import pytest

import litellm
from litellm import Router
from litellm.proxy._types import (
    LiteLLM_TeamTableCachedObj,
    ProxyException,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_checks import vector_store_access_check
from litellm.proxy.auth.user_api_key_auth import _team_obj_from_token
from litellm.types.vector_stores import LiteLLM_ManagedVectorStore
from litellm.vector_stores.vector_store_registry import VectorStoreRegistry
from tests.unit.proxy.vector_store_access_group_state import (
    TEAM_ID,
    TOKEN,
    ProxyState,
    make_key,
    make_team,
    own_permission,
)


@pytest.fixture
def proxy(monkeypatch: pytest.MonkeyPatch) -> ProxyState:
    return ProxyState(monkeypatch)


def _virtual_key(groups: Sequence[str]) -> UserAPIKeyAuth:
    key: Final = UserAPIKeyAuth(token=TOKEN, api_key=TOKEN, team_id=TEAM_ID, access_group_ids=list(groups))
    key.via_virtual_key = True
    return key


async def _request_is_allowed(
    key: UserAPIKeyAuth, team: LiteLLM_TeamTableCachedObj | None, store: str, *, deny_by_default: bool = False
) -> bool:
    try:
        await vector_store_access_check(
            request_body={"vector_store_ids": [store]},
            team_object=team,
            valid_token=key,
            deny_by_default=deny_by_default,
        )
    except ProxyException as refused:
        if refused.code != "401":
            raise
        return False
    return True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("key_stores", "key_groups", "team_stores", "team_groups", "store", "allowed"),
    [
        pytest.param(None, ("g1",), None, (), "store-a", True, id="key-group-grants"),
        pytest.param(None, ("g1",), None, (), "store-b", False, id="key-group-restricts"),
        pytest.param((), ("g1",), None, (), "store-b", False, id="key-group-restricts-with-empty-own-list"),
        pytest.param(None, (), None, ("g1",), "store-a", True, id="team-group-grants"),
        pytest.param(None, (), None, ("g1",), "store-b", False, id="team-group-restricts"),
        pytest.param(None, (), (), ("g1",), "store-b", False, id="team-group-restricts-with-empty-own-list"),
        pytest.param(("store-x",), ("g1",), None, (), "store-a", True, id="key-group-widens-own-list"),
        pytest.param(("store-x",), ("g1",), None, (), "store-x", True, id="key-own-list-still-counts"),
        pytest.param(("store-x",), ("g1",), None, (), "store-b", False, id="key-neither-list-denies"),
        pytest.param(None, (), ("store-x",), ("g1",), "store-a", True, id="team-group-widens-own-list"),
        pytest.param(None, (), ("store-x",), ("g1",), "store-x", True, id="team-own-list-still-counts"),
        pytest.param(("store-x",), (), None, ("g1",), "store-a", False, id="key-own-list-narrows-team-group"),
        pytest.param(None, ("g1",), ("store-x",), (), "store-a", False, id="team-own-list-narrows-key-group"),
        pytest.param(None, ("g1", "g2"), None, (), "store-c", True, id="stores-from-a-second-group"),
        pytest.param(None, ("models-only",), None, (), "store-b", True, id="models-only-group-changes-nothing"),
        pytest.param(None, (), None, ("models-only",), "store-b", True, id="team-models-only-group-changes-nothing"),
        pytest.param(("store-x",), ("models-only",), None, (), "store-b", False, id="models-only-keeps-own-list"),
        pytest.param(None, (), None, (), "store-b", True, id="no-grants-anywhere-changes-nothing"),
    ],
)
async def test_request_check_combines_own_list_and_group_stores(
    proxy: ProxyState,
    key_stores: Sequence[str] | None,
    key_groups: Sequence[str],
    team_stores: Sequence[str] | None,
    team_groups: Sequence[str],
    store: str,
    allowed: bool,
) -> None:
    await proxy.add_group("g1", ("store-a",))
    await proxy.add_group("g2", ("store-c",))
    await proxy.add_group("models-only", models=("gpt-4",))

    assert (
        await _request_is_allowed(make_key(key_stores, key_groups, TEAM_ID), make_team(team_stores, team_groups), store)
        is allowed
    )


@pytest.mark.asyncio
async def test_key_group_counts_when_it_lists_only_the_keys_team(proxy: ProxyState) -> None:
    await proxy.add_group("g1", ("store-a",), key_ids=())

    assert await _request_is_allowed(make_key(groups=("g1",), team_id=TEAM_ID), None, "store-a") is True
    assert await _request_is_allowed(make_key(groups=("g1",), team_id=TEAM_ID), None, "store-b") is False


@pytest.mark.asyncio
async def test_group_granted_by_an_identity_rule_counts_for_the_key(proxy: ProxyState) -> None:
    await proxy.add_group("g1", ("store-a",), team_ids=(), key_ids=())
    key: Final = make_key(("store-x",))
    key.apply_access_group_rule_grants(("g1",))

    assert await _request_is_allowed(key, None, "store-a") is True
    assert await _request_is_allowed(key, None, "store-b") is False


@pytest.mark.asyncio
async def test_key_cannot_claim_a_group_that_does_not_list_it(proxy: ProxyState) -> None:
    await proxy.add_group("g1", ("store-a",), team_ids=(), key_ids=())

    assert await _request_is_allowed(make_key(("store-x",), ("g1",)), None, "store-a") is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("store", "allowed"),
    [pytest.param("store-a", True, id="granted"), pytest.param("store-b", False, id="not-granted")],
)
async def test_deny_by_default_counts_group_grants(proxy: ProxyState, store: str, allowed: bool) -> None:
    await proxy.add_group("g1", ("store-a",))
    key: Final = _virtual_key(("g1",))

    assert await _request_is_allowed(key, make_team(groups=("g1",)), store, deny_by_default=True) is allowed


@pytest.mark.asyncio
async def test_deny_by_default_requires_the_key_too_not_only_its_team(proxy: ProxyState) -> None:
    await proxy.add_group("g1", ("store-a",))
    key: Final = _virtual_key(())

    assert await _request_is_allowed(key, make_team(groups=("g1",)), "store-a", deny_by_default=True) is False


_MANAGED_A: Final = base64.urlsafe_b64encode(
    b"litellm_proxy:vector_store;unified_id,abc;target_model_names,gpt-4;resource_id,kb-shared"
).decode()
_PLAIN_B: Final = LiteLLM_ManagedVectorStore(vector_store_id="kb-shared", custom_llm_provider="bedrock")


async def _all_allowed(key: UserAPIKeyAuth, stores: Sequence[str]) -> bool:
    try:
        await vector_store_access_check(
            request_body={"vector_store_ids": list(stores)}, team_object=None, valid_token=key
        )
    except ProxyException as refused:
        if refused.code != "401":
            raise
        return False
    return True


@pytest.mark.asyncio
async def test_an_own_list_still_matches_by_provider_id(proxy: ProxyState, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "vector_store_registry", VectorStoreRegistry(vector_stores=[_PLAIN_B]))

    assert await _all_allowed(make_key((_MANAGED_A,)), ("kb-shared",)) is True


async def _answer(
    layer: Literal["key", "team"],
    own: Sequence[str] | None,
    groups: Sequence[str],
    requested: Sequence[str],
    deny_by_default: bool,
) -> tuple[object, ...] | None:
    """The refusal a virtual key gets when `layer` holds the `own` list and the `groups`, or None when allowed.
    Under deny-by-default the other layer lists every requested store, so `layer` is the one that decides."""
    other: Final = tuple(requested) if deny_by_default else None
    key_stores: Final = own if layer == "key" else other
    key: Final = _virtual_key(groups if layer == "key" else ()).model_copy(
        update={"object_permission_id": "own", "object_permission": own_permission(key_stores)}
        if key_stores is not None
        else {}
    )
    team: Final = make_team(own if layer == "team" else other, groups if layer == "team" else ())
    try:
        await vector_store_access_check(
            request_body={"vector_store_ids": list(requested)},
            team_object=team,
            valid_token=key,
            deny_by_default=deny_by_default,
        )
    except ProxyException as refused:
        return (refused.code, refused.type, refused.message)
    return None


@pytest.mark.asyncio
@pytest.mark.parametrize("deny_by_default", [False, True], ids=["legacy", "deny-by-default"])
@pytest.mark.parametrize("layer", ["key", "team"])
@pytest.mark.parametrize("own", [None, ("store-x",)], ids=["no-own-list", "own-list"])
@pytest.mark.parametrize(
    "group", [("store-a",), ("kb-shared",), (_MANAGED_A,)], ids=["plain", "provider-id", "managed"]
)
@pytest.mark.parametrize(
    "requested",
    [("store-a",), ("store-x",), ("store-b",), ("kb-shared",), (_MANAGED_A,), ("store-a", "store-b")],
    ids=["store-a", "store-x", "store-b", "provider-id", "managed", "two-stores"],
)
async def test_a_group_store_is_checked_exactly_like_the_same_store_on_the_own_list(
    proxy: ProxyState,
    monkeypatch: pytest.MonkeyPatch,
    deny_by_default: bool,
    layer: Literal["key", "team"],
    own: Sequence[str] | None,
    group: Sequence[str],
    requested: Sequence[str],
) -> None:
    monkeypatch.setattr(litellm, "vector_store_registry", VectorStoreRegistry(vector_stores=[_PLAIN_B]))
    await proxy.add_group("g1", group)

    through_the_group: Final = await _answer(layer, own, ("g1",), requested, deny_by_default)
    on_the_own_list: Final = await _answer(layer, (*(own or ()), *group), (), requested, deny_by_default)

    assert through_the_group == on_the_own_list


@pytest.mark.asyncio
async def test_a_refusal_lists_the_allowed_stores_as_main_does(proxy: ProxyState) -> None:
    with pytest.raises(ProxyException) as denied:
        await vector_store_access_check(
            request_body={"vector_store_ids": ["store-b"]}, team_object=None, valid_token=make_key(("store-x",))
        )

    assert "Only allowed to access ['store-x']" in denied.value.message


@pytest.mark.asyncio
async def test_keyless_team_member_gets_the_teams_group_stores(proxy: ProxyState) -> None:
    await proxy.add_group("g1", ("store-a",))
    member: Final = UserAPIKeyAuth(team_id=TEAM_ID)

    assert await _request_is_allowed(member, make_team(groups=("g1",)), "store-a") is True
    assert await _request_is_allowed(member, make_team(groups=("g1",)), "store-b") is False


@pytest.mark.asyncio
async def test_deny_by_default_still_denies_a_models_only_group(proxy: ProxyState) -> None:
    await proxy.add_group("models-only", models=("gpt-4",))
    key: Final = _virtual_key(("models-only",))

    assert await _request_is_allowed(key, make_team(groups=("models-only",)), "store-a", deny_by_default=True) is False


_UNREADABLE: Final = "Could not read the team or its access groups to check vector store access"


@pytest.mark.asyncio
@pytest.mark.parametrize("own", [None, ("store-x",)], ids=["no-own-list", "own-list"])
@pytest.mark.parametrize("layer", ["key", "team"])
async def test_a_group_that_cannot_be_read_blocks_the_request_instead_of_lifting_the_restriction(
    monkeypatch: pytest.MonkeyPatch, layer: Literal["key", "team"], own: Sequence[str] | None
) -> None:
    ProxyState(monkeypatch, group_reads=AsyncMock(side_effect=RuntimeError("database unavailable")))

    refusal: Final = await _answer(layer, own, ("g1",), ("store-b",), deny_by_default=False)

    assert refusal is not None
    assert refusal[0] == "503"
    assert refusal[2] == _UNREADABLE


@pytest.mark.asyncio
async def test_auths_stand_in_for_a_team_it_could_not_read_is_read_again_so_the_teams_groups_still_apply(
    proxy: ProxyState,
) -> None:
    await proxy.add_group("g1", ("store-a",))
    await proxy.add_team(make_team(groups=("g1",)))
    key: Final = make_key(team_id=TEAM_ID)

    assert await _request_is_allowed(key, _team_obj_from_token(key), "store-a") is True
    assert await _request_is_allowed(key, _team_obj_from_token(key), "store-b") is False


@pytest.mark.asyncio
async def test_a_stand_in_team_that_still_cannot_be_read_blocks_the_request(monkeypatch: pytest.MonkeyPatch) -> None:
    ProxyState(monkeypatch, team_reads=AsyncMock(side_effect=RuntimeError("database unavailable")))
    key: Final = make_key(team_id=TEAM_ID)

    with pytest.raises(ProxyException) as blocked:
        await vector_store_access_check(
            request_body={"vector_store_ids": ["store-b"]}, team_object=_team_obj_from_token(key), valid_token=key
        )

    assert (blocked.value.code, blocked.value.message) == ("503", _UNREADABLE)


@pytest.mark.asyncio
async def test_a_strict_team_layer_keeps_the_stand_in_team_and_is_refused_as_on_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ProxyState(monkeypatch, team_reads=AsyncMock(side_effect=RuntimeError("database unavailable")))
    key: Final = _virtual_key(())

    assert await _request_is_allowed(key, _team_obj_from_token(key), "store-b", deny_by_default=True) is False


@pytest.mark.asyncio
async def test_deleted_group_grants_nothing_and_does_not_fail_the_request(proxy: ProxyState) -> None:
    assert await _request_is_allowed(make_key(team_id=TEAM_ID), make_team(groups=("deleted",)), "store-b") is True
    assert (
        await _request_is_allowed(make_key(("store-x",), team_id=TEAM_ID), make_team(groups=("deleted",)), "store-b")
        is False
    )


@pytest.mark.asyncio
async def test_group_cached_before_the_upgrade_keeps_the_callers_own_list(proxy: ProxyState) -> None:
    await proxy.cache.async_set_cache(
        key="access_group_id:old",
        value={"access_group_id": "old", "access_group_name": "old", "assigned_team_ids": [TEAM_ID]},
    )

    assert (
        await _request_is_allowed(make_key(("store-x",), team_id=TEAM_ID), make_team(groups=("old",)), "store-b")
        is False
    )
    assert (
        await _request_is_allowed(make_key(("store-x",), team_id=TEAM_ID), make_team(groups=("old",)), "store-x")
        is True
    )


@pytest.mark.asyncio
async def test_malformed_store_ids_are_rejected_when_only_a_group_restricts(proxy: ProxyState) -> None:
    await proxy.add_group("g1", ("store-a",))

    with pytest.raises(ProxyException, match="vector_store_ids must be a list of strings"):
        await vector_store_access_check(
            request_body={"model": "gpt-4o-mini", "vector_store_ids": 123},
            team_object=make_team(groups=("g1",)),
            valid_token=make_key(team_id=TEAM_ID),
            llm_router=Router(model_list=[{"model_name": "gpt-4o-mini", "litellm_params": {"model": "gpt-4o-mini"}}]),
        )


@pytest.mark.asyncio
async def test_request_naming_no_store_never_reads_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    proxy: Final = ProxyState(monkeypatch, group_reads=AsyncMock(side_effect=RuntimeError("database unavailable")))

    result: Final = await vector_store_access_check(
        request_body={"model": "gpt-4"}, team_object=make_team(groups=("g1",)), valid_token=make_key(team_id=TEAM_ID)
    )

    assert result is True
    assert proxy.group_reads.await_count == 0
