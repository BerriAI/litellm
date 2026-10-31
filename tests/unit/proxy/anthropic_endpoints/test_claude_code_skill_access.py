"""Unit tests for claude_code_skill_access.py: key/team grant resolution."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy._types import LiteLLM_ObjectPermissionTable, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.anthropic_endpoints.claude_code_endpoints.claude_code_skill_access import (
    PUBLIC_ONLY,
    AccessGroupSkills,
    access_group_skills,
    granted_skills,
    resolve_skill_visibility,
    skill_visibility,
)


def _perm(skills: list[str] | None) -> LiteLLM_ObjectPermissionTable:
    return LiteLLM_ObjectPermissionTable(object_permission_id="perm", skills=skills)


def _key(key_skills: list[str] | None, team_skills: list[str] | None) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="sk-user",
        user_role=LitellmUserRoles.INTERNAL_USER,
        object_permission=_perm(key_skills) if key_skills is not None else None,
        team_object_permission=_perm(team_skills) if team_skills is not None else None,
    )


def _plugin(name: str, enabled: bool) -> MagicMock:
    record = MagicMock()
    record.name = name
    record.enabled = enabled
    return record


@pytest.mark.parametrize(
    ("key_skills", "team_skills", "expected"),
    [
        (None, None, frozenset()),
        ([], [], frozenset()),
        (["a", "b"], None, frozenset({"a", "b"})),
        (None, ["a", "b"], frozenset({"a", "b"})),
        (["a", "b"], ["b", "c"], frozenset({"b"})),
        (["a"], ["c"], frozenset()),
        (["a", "b"], [], frozenset({"a", "b"})),
        ([], ["a", "b"], frozenset({"a", "b"})),
    ],
)
def test_granted_skills_intersects_key_with_team(key_skills, team_skills, expected):
    assert granted_skills(_key(key_skills, team_skills)) == expected


def test_visibility_enabled_plugin_is_public_for_everyone():
    public = _plugin("public-skill", enabled=True)

    assert skill_visibility(None).allows(public)
    assert skill_visibility(_key(None, None)).allows(public)
    assert skill_visibility(_key([], ["other"])).allows(public)


def test_visibility_disabled_plugin_needs_grant_or_admin():
    private = _plugin("private-skill", enabled=False)
    admin = UserAPIKeyAuth(api_key="sk-9876", user_role=LitellmUserRoles.PROXY_ADMIN)

    assert not skill_visibility(None).allows(private)
    assert not skill_visibility(_key(None, None)).allows(private)
    assert not skill_visibility(_key(["other-skill"], None)).allows(private)
    assert not skill_visibility(_key(["private-skill"], ["other-skill"])).allows(private)
    assert skill_visibility(_key(["private-skill"], None)).allows(private)
    assert skill_visibility(_key(None, ["private-skill"])).allows(private)
    assert skill_visibility(admin).allows(private)


def test_where_clause_bounds_the_plugin_query_to_what_the_caller_may_see():
    admin = UserAPIKeyAuth(api_key="sk-9876", user_role=LitellmUserRoles.PROXY_ADMIN)

    assert skill_visibility(None).where() == {"enabled": True}
    assert skill_visibility(_key(None, None)).where() == {"enabled": True}
    assert skill_visibility(_key(["b", "a"], None)).where() == {"OR": [{"enabled": True}, {"name": {"in": ["a", "b"]}}]}
    assert skill_visibility(_key(["a", "b"], ["b"])).where() == {"OR": [{"enabled": True}, {"name": {"in": ["b"]}}]}
    assert skill_visibility(admin).where() == {}


@pytest.mark.parametrize(
    ("key_skills", "team_skills", "inherited", "expected"),
    [
        (None, None, AccessGroupSkills(key=frozenset({"g"})), frozenset({"g"})),
        (None, None, AccessGroupSkills(team=frozenset({"g"})), frozenset({"g"})),
        (None, None, AccessGroupSkills(), frozenset()),
        ([], [], AccessGroupSkills(key=frozenset({"g1"}), team=frozenset({"g2"})), frozenset()),
        (["a"], None, AccessGroupSkills(key=frozenset({"g"})), frozenset({"a", "g"})),
        (None, ["a"], AccessGroupSkills(team=frozenset({"g"})), frozenset({"a", "g"})),
        (None, ["a"], AccessGroupSkills(key=frozenset({"g"})), frozenset()),
        (["a"], ["a", "b"], AccessGroupSkills(key=frozenset({"b"})), frozenset({"a", "b"})),
        (["a"], ["a"], AccessGroupSkills(key=frozenset({"g"}), team=frozenset({"g"})), frozenset({"a", "g"})),
    ],
)
def test_granted_skills_adds_access_group_skills_to_their_own_scope_before_intersecting(
    key_skills, team_skills, inherited, expected
):
    assert granted_skills(_key(key_skills, team_skills), inherited) == expected


def test_visibility_with_inherited_skills_only_opens_the_inherited_private_plugin():
    inherited = AccessGroupSkills(key=frozenset({"group-skill"}))
    visibility = skill_visibility(_key(None, None), inherited)

    assert visibility.allows(_plugin("group-skill", enabled=False))
    assert not visibility.allows(_plugin("other-private", enabled=False))
    assert visibility.allows(_plugin("public", enabled=True))
    assert visibility.where() == {"OR": [{"enabled": True}, {"name": {"in": ["group-skill"]}}]}


@pytest.mark.asyncio
async def test_resolve_skill_visibility_folds_in_access_group_skills_for_non_admins():
    loader = AsyncMock(return_value=AccessGroupSkills(team=frozenset({"group-skill"})))
    caller = _key(None, None)

    visibility = await resolve_skill_visibility(caller, loader)

    loader.assert_awaited_once_with(caller)
    assert visibility.allows(_plugin("group-skill", enabled=False))
    assert not visibility.allows(_plugin("other-private", enabled=False))


@pytest.mark.asyncio
async def test_resolve_skill_visibility_skips_group_lookup_for_anonymous_and_admin_callers():
    loader = AsyncMock(return_value=AccessGroupSkills(key=frozenset({"group-skill"})))
    admin = UserAPIKeyAuth(api_key="sk-9876", user_role=LitellmUserRoles.PROXY_ADMIN)

    assert not (await resolve_skill_visibility(None, loader)).allows(_plugin("group-skill", enabled=False))
    assert (await resolve_skill_visibility(admin, loader)).sees_private
    loader.assert_not_awaited()


def _team(team_id: str, access_group_ids: list[str], blocked: bool = False) -> MagicMock:
    team = MagicMock()
    team.team_id = team_id
    team.access_group_ids = access_group_ids
    team.blocked = blocked
    return team


def _group(access_group_id: str, skills: list[str], assigned_key_ids: list[str] | None = None) -> MagicMock:
    group = MagicMock()
    group.access_group_id = access_group_id
    group.access_skill_names = skills
    group.assigned_team_ids = []
    group.assigned_key_ids = assigned_key_ids or []
    return group


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("team_blocked", "expected_team_skills"),
    [(False, frozenset({"team-group-skill"})), (True, frozenset())],
)
async def test_access_group_skills_resolves_key_groups_and_team_groups(monkeypatch, team_blocked, expected_team_skills):
    from litellm.proxy import proxy_server
    from litellm.proxy.auth import auth_checks

    caller = UserAPIKeyAuth(
        api_key="sk-user",
        user_role=LitellmUserRoles.INTERNAL_USER,
        team_id="team-1",
        access_group_ids=["key-group", "stray-group"],
    ).model_copy(update={"requires_fresh_policy": True})
    groups = {
        "key-group": _group("key-group", ["key-group-skill"], assigned_key_ids=[caller.token or ""]),
        "stray-group": _group("stray-group", ["stray-skill"]),
        "team-group": _group("team-group", ["team-group-skill"]),
    }

    stale_groups = {group_id: _group(group_id, ["stale-skill"]) for group_id in groups}

    async def fake_access_object(access_group_id, check_db_only=False, **_kwargs):
        return (groups if check_db_only else stale_groups)[access_group_id]

    async def fake_skill_names(access_group_ids, **_kwargs):
        return frozenset().union(*(groups[group_id].access_skill_names for group_id in access_group_ids))

    monkeypatch.setattr(auth_checks, "get_access_object", fake_access_object)
    monkeypatch.setattr(auth_checks, "get_skill_names_from_access_groups", fake_skill_names)
    monkeypatch.setattr(
        auth_checks, "get_team_object", AsyncMock(return_value=_team("team-1", ["team-group"], blocked=team_blocked))
    )
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    monkeypatch.setattr(proxy_server, "user_api_key_cache", MagicMock())
    inherited = await access_group_skills(caller)

    assert inherited == AccessGroupSkills(key=frozenset({"key-group-skill"}), team=expected_team_skills)


@pytest.mark.asyncio
@pytest.mark.parametrize("fresh", [True, False])
async def test_access_group_skills_surfaces_key_group_lookup_failures(monkeypatch, fresh):
    from litellm.proxy import proxy_server
    from litellm.proxy.auth import auth_checks

    caller = UserAPIKeyAuth(
        api_key="sk-user", user_role=LitellmUserRoles.INTERNAL_USER, access_group_ids=["key-group"]
    ).model_copy(update={"requires_fresh_policy": fresh})
    monkeypatch.setattr(auth_checks, "get_access_object", AsyncMock(side_effect=RuntimeError("writer down")))
    monkeypatch.setattr(auth_checks, "get_skill_names_from_access_groups", AsyncMock(return_value=frozenset()))
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    monkeypatch.setattr(proxy_server, "user_api_key_cache", MagicMock())

    with pytest.raises(RuntimeError, match="writer down"):
        await access_group_skills(caller)


@pytest.mark.asyncio
async def test_access_group_skills_without_groups_or_team_grants_nothing(monkeypatch):
    from litellm.proxy import proxy_server
    from litellm.proxy.auth import auth_checks

    get_team_object = AsyncMock()
    monkeypatch.setattr(auth_checks, "get_team_object", get_team_object)
    monkeypatch.setattr(auth_checks, "get_skill_names_from_access_groups", AsyncMock(return_value=frozenset()))
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    caller = UserAPIKeyAuth(api_key="sk-user", user_role=LitellmUserRoles.INTERNAL_USER)

    assert await access_group_skills(caller) == AccessGroupSkills()
    get_team_object.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("fresh", [True, False])
async def test_access_group_skills_keeps_team_ceiling_when_team_lookup_fails(monkeypatch, fresh):
    from litellm.proxy import proxy_server
    from litellm.proxy.auth import auth_checks

    caller = UserAPIKeyAuth(
        api_key="sk-user", user_role=LitellmUserRoles.INTERNAL_USER, team_id="team-1", access_group_ids=["key-group"]
    ).model_copy(update={"requires_fresh_policy": fresh})
    key_group = _group("key-group", ["secret-skill"], assigned_key_ids=[caller.token or ""])
    monkeypatch.setattr(auth_checks, "get_team_object", AsyncMock(side_effect=RuntimeError("writer down")))
    monkeypatch.setattr(auth_checks, "get_access_object", AsyncMock(return_value=key_group))
    monkeypatch.setattr(auth_checks, "get_skill_names_from_access_groups", AsyncMock(return_value=frozenset()))
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    monkeypatch.setattr(proxy_server, "user_api_key_cache", MagicMock())

    with pytest.raises(RuntimeError, match="writer down"):
        await access_group_skills(caller)


@pytest.mark.asyncio
@pytest.mark.parametrize("fresh", [True, False])
async def test_access_group_skills_surfaces_team_group_lookup_failures(monkeypatch, fresh):
    from litellm.proxy import proxy_server
    from litellm.proxy.auth import auth_checks

    caller = UserAPIKeyAuth(
        api_key="sk-user", user_role=LitellmUserRoles.INTERNAL_USER, team_id="team-1", access_group_ids=["key-group"]
    ).model_copy(update={"requires_fresh_policy": fresh})
    key_group = _group("key-group", ["secret-skill"], assigned_key_ids=[caller.token or ""])
    monkeypatch.setattr(auth_checks, "get_team_object", AsyncMock(return_value=_team("team-1", ["team-group"])))

    async def lookup(access_group_id: str, **_kwargs: object) -> MagicMock:
        if access_group_id == "team-group":
            raise RuntimeError("writer down")
        return key_group

    monkeypatch.setattr(auth_checks, "get_access_object", lookup)
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    monkeypatch.setattr(proxy_server, "user_api_key_cache", MagicMock())

    with pytest.raises(RuntimeError, match="writer down"):
        await access_group_skills(caller)


@pytest.mark.asyncio
async def test_access_group_skills_treats_confirmed_missing_team_like_no_team(monkeypatch):
    from litellm.proxy import proxy_server
    from litellm.proxy.auth import auth_checks

    caller = UserAPIKeyAuth(
        api_key="sk-user",
        user_role=LitellmUserRoles.INTERNAL_USER,
        team_id="removed-team",
        access_group_ids=["key-group"],
    )
    key_group = _group("key-group", ["secret-skill"], assigned_key_ids=[caller.token or ""])
    monkeypatch.setattr(
        auth_checks, "get_team_object", AsyncMock(side_effect=auth_checks.TeamNotFoundError(team_id="removed-team"))
    )
    monkeypatch.setattr(auth_checks, "get_access_object", AsyncMock(return_value=key_group))
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    monkeypatch.setattr(proxy_server, "user_api_key_cache", MagicMock())

    assert await access_group_skills(caller) == AccessGroupSkills(key=frozenset({"secret-skill"}))


@pytest.mark.asyncio
@pytest.mark.parametrize("fresh", [True, False])
@pytest.mark.parametrize("missing_scope", ["key", "team"])
async def test_access_group_skills_marks_scope_unresolved_when_a_listed_group_is_missing(
    monkeypatch, fresh, missing_scope
):
    from litellm.proxy import proxy_server
    from litellm.proxy.auth import auth_checks

    key_groups = ["typo-group", "key-group"] if missing_scope == "key" else ["key-group"]
    team_groups = ["team-group"] if missing_scope == "key" else ["stale-team-group", "team-group"]
    caller = UserAPIKeyAuth(
        api_key="sk-user", user_role=LitellmUserRoles.INTERNAL_USER, team_id="team-1", access_group_ids=key_groups
    ).model_copy(update={"requires_fresh_policy": fresh})
    groups = {
        "key-group": _group("key-group", ["secret-skill"], assigned_key_ids=[caller.token or ""]),
        "team-group": _group("team-group", ["team-skill"]),
    }

    async def lookup(access_group_id: str, **_kwargs: object) -> MagicMock:
        if access_group_id not in groups:
            raise auth_checks.AccessGroupNotFoundError(access_group_id=access_group_id)
        return groups[access_group_id]

    monkeypatch.setattr(auth_checks, "get_team_object", AsyncMock(return_value=_team("team-1", team_groups)))
    monkeypatch.setattr(auth_checks, "get_access_object", lookup)
    monkeypatch.setattr(proxy_server, "prisma_client", MagicMock())
    monkeypatch.setattr(proxy_server, "user_api_key_cache", MagicMock())

    assert await access_group_skills(caller) == AccessGroupSkills(unresolved=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_scope", ["key", "team"])
async def test_access_group_skills_marks_scope_unresolved_through_the_real_group_lookup(monkeypatch, missing_scope):
    from litellm.proxy import proxy_server
    from litellm.proxy._types import LiteLLM_AccessGroupTable
    from litellm.proxy.auth import auth_checks
    from litellm.proxy.auth.auth_checks import cache_access_object
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

    key_groups = ["typo-group", "key-group"] if missing_scope == "key" else ["key-group"]
    team_groups = ["team-group"] if missing_scope == "key" else ["stale-team-group", "team-group"]
    caller = UserAPIKeyAuth(
        api_key="sk-user", user_role=LitellmUserRoles.INTERNAL_USER, team_id="team-1", access_group_ids=key_groups
    )
    cache = UserApiKeyCache()
    for group in (
        LiteLLM_AccessGroupTable(
            access_group_id="key-group",
            access_group_name="key-group",
            access_skill_names=["secret-skill"],
            assigned_key_ids=[caller.token or ""],
        ),
        LiteLLM_AccessGroupTable(
            access_group_id="team-group", access_group_name="team-group", access_skill_names=["team-skill"]
        ),
    ):
        await cache_access_object(
            access_group_id=group.access_group_id, access_group_table=group, user_api_key_cache=cache
        )
    prisma = MagicMock()
    prisma.db.litellm_accessgrouptable.find_unique = AsyncMock(return_value=None)
    monkeypatch.setattr(auth_checks, "get_team_object", AsyncMock(return_value=_team("team-1", team_groups)))
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)

    assert await access_group_skills(caller) == AccessGroupSkills(unresolved=True)
    prisma.db.litellm_accessgrouptable.find_unique.assert_awaited_once()


def test_skill_visibility_is_public_only_while_a_group_scope_is_unresolved():
    caller = UserAPIKeyAuth(
        api_key="sk-user",
        user_role=LitellmUserRoles.INTERNAL_USER,
        object_permission=_perm(["private-a"]),
        team_object_permission=_perm(["private-a", "private-b"]),
    )

    assert skill_visibility(caller, AccessGroupSkills(unresolved=True)) == PUBLIC_ONLY
    assert skill_visibility(caller, AccessGroupSkills()).granted == frozenset({"private-a"})


@pytest.mark.asyncio
@pytest.mark.parametrize("check_db_only", [True, False])
@pytest.mark.parametrize(("strict_errors", "expect_raise"), [(True, True), (False, False)])
async def test_shared_group_loader_skips_missing_groups_unless_strict(
    monkeypatch, check_db_only, strict_errors, expect_raise
):
    """A group with no row grants nothing on cached and fresh reads alike; only ``strict_errors`` callers,
    which turn the typed absence into an unresolved policy, see ``AccessGroupNotFoundError``."""
    from litellm.proxy.auth import auth_checks

    groups = {"real-group": _group("real-group", [])}
    groups["real-group"].access_mcp_server_ids = ["mcp-1"]

    async def lookup(access_group_id: str, **_kwargs: object) -> MagicMock:
        if access_group_id not in groups:
            raise auth_checks.AccessGroupNotFoundError(access_group_id=access_group_id)
        return groups[access_group_id]

    monkeypatch.setattr(auth_checks, "get_access_object", lookup)
    call = auth_checks._get_resources_from_access_groups(
        access_group_ids=["typo-group", "real-group"],
        resource_field="access_mcp_server_ids",
        prisma_client=MagicMock(),
        user_api_key_cache=MagicMock(),
        proxy_logging_obj=MagicMock(),
        check_db_only=check_db_only,
        strict_errors=strict_errors,
    )
    if expect_raise:
        with pytest.raises(auth_checks.AccessGroupNotFoundError):
            await call
        return
    assert await call == ["mcp-1"]
