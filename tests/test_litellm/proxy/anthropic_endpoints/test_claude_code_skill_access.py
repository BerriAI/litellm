"""Unit tests for claude_code_skill_access.py: key/team grant resolution."""

from unittest.mock import MagicMock

import pytest

from litellm.proxy._types import LiteLLM_ObjectPermissionTable, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.anthropic_endpoints.claude_code_endpoints.claude_code_skill_access import (
    granted_skills,
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
    admin = UserAPIKeyAuth(api_key="sk-1234", user_role=LitellmUserRoles.PROXY_ADMIN)

    assert not skill_visibility(None).allows(private)
    assert not skill_visibility(_key(None, None)).allows(private)
    assert not skill_visibility(_key(["other-skill"], None)).allows(private)
    assert not skill_visibility(_key(["private-skill"], ["other-skill"])).allows(private)
    assert skill_visibility(_key(["private-skill"], None)).allows(private)
    assert skill_visibility(_key(None, ["private-skill"])).allows(private)
    assert skill_visibility(admin).allows(private)


def test_where_clause_bounds_the_plugin_query_to_what_the_caller_may_see():
    admin = UserAPIKeyAuth(api_key="sk-1234", user_role=LitellmUserRoles.PROXY_ADMIN)

    assert skill_visibility(None).where() == {"enabled": True}
    assert skill_visibility(_key(None, None)).where() == {"enabled": True}
    assert skill_visibility(_key(["b", "a"], None)).where() == {"OR": [{"enabled": True}, {"name": {"in": ["a", "b"]}}]}
    assert skill_visibility(_key(["a", "b"], ["b"])).where() == {"OR": [{"enabled": True}, {"name": {"in": ["b"]}}]}
    assert skill_visibility(admin).where() == {}
