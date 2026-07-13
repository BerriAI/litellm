"""
Unit tests for claude_code_skill_authz.get_allowed_skills.

This is the authorization gate that decides which non-public (disabled)
Claude Code skills a key/team/org can additionally see via the
`GET /claude-code/marketplace.json?key=...` endpoint. The intersection/ceiling
rules deliberately mirror MCPRequestHandler.get_allowed_mcp_servers:
- an empty/missing allowed_skills list at a given level means that level
  places no restriction and is skipped from the intersection
- key/team: if both restrict, intersect; if only one restricts, use it
- org: acts as a ceiling - if the org has an explicit list, it caps whatever
  the key/team level resolved to; if nothing lower restricts, the org list
  becomes the result outright

Team/org lookups are stubbed at their real collaborator functions
(auth_checks.get_team_object / get_org_object / get_object_permission) rather
than by faking the underlying prisma/cache stack those already have their own
tests for - this keeps these tests focused on get_allowed_skills's own
aggregation logic.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from litellm.models.object_permission import LiteLLM_ObjectPermissionTable
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.anthropic_endpoints.claude_code_endpoints.claude_code_skill_authz import (
    get_allowed_skills,
)
from litellm.proxy.auth import auth_checks


@pytest.fixture(autouse=True)
def _default_team_org_lookups(monkeypatch):
    """By default, nobody has a team/org - team_id/org_id being unset on the
    key is what actually short-circuits these, but stub them anyway so a test
    that forgets to override a lookup fails loudly instead of hitting a real DB."""
    monkeypatch.setattr(auth_checks, "get_team_object", AsyncMock(return_value=None))
    monkeypatch.setattr(auth_checks, "get_org_object", AsyncMock(return_value=None))
    monkeypatch.setattr(auth_checks, "get_object_permission", AsyncMock(return_value=None))


def _object_permission(allowed_skills):
    return LiteLLM_ObjectPermissionTable(object_permission_id="perm-1", allowed_skills=allowed_skills)


def _key_auth(*, allowed_skills=None, team_id=None, org_id=None):
    object_permission = _object_permission(allowed_skills) if allowed_skills is not None else None
    return UserAPIKeyAuth(
        api_key="sk-test",
        user_id="test-user",
        team_id=team_id,
        org_id=org_id,
        object_permission=object_permission,
    )


def _mock_org(monkeypatch, *, allowed_skills):
    monkeypatch.setattr(
        auth_checks, "get_org_object", AsyncMock(return_value=SimpleNamespace(object_permission_id="org-perm-1"))
    )
    monkeypatch.setattr(
        auth_checks,
        "get_object_permission",
        AsyncMock(return_value=_object_permission(allowed_skills)),
    )


def _mock_team(monkeypatch, *, allowed_skills):
    monkeypatch.setattr(
        auth_checks,
        "get_team_object",
        AsyncMock(return_value=SimpleNamespace(object_permission=_object_permission(allowed_skills))),
    )


@pytest.mark.asyncio
async def test_org_grants_with_no_team_or_key_restriction_becomes_the_effective_set(monkeypatch):
    """Org narrows, nothing else does -> org's list becomes the ceiling outright."""
    key_auth = _key_auth(allowed_skills=None, org_id="org-1")
    _mock_org(monkeypatch, allowed_skills=["a--x"])

    result = await get_allowed_skills(key_auth, prisma_client=object())

    assert result == frozenset({"a--x"})


@pytest.mark.asyncio
async def test_org_grants_nothing_key_grants_becomes_the_effective_set(monkeypatch):
    """Key narrows, org places no restriction -> key's list wins as-is."""
    key_auth = _key_auth(allowed_skills=["a--x"], org_id=None)

    result = await get_allowed_skills(key_auth, prisma_client=object())

    assert result == frozenset({"a--x"})


@pytest.mark.asyncio
async def test_org_grants_superset_of_key_intersects_to_key(monkeypatch):
    """Both org and key restrict -> org caps the key's (tighter) list via intersection."""
    key_auth = _key_auth(allowed_skills=["a--x"], org_id="org-1")
    _mock_org(monkeypatch, allowed_skills=["a--x", "a--y"])

    result = await get_allowed_skills(key_auth, prisma_client=object())

    assert result == frozenset({"a--x"})


@pytest.mark.asyncio
async def test_disjoint_org_and_key_grants_intersect_to_empty(monkeypatch):
    """Org and key each restrict to disjoint sets -> no overlap, effective set is empty."""
    key_auth = _key_auth(allowed_skills=["a--y"], org_id="org-1")
    _mock_org(monkeypatch, allowed_skills=["a--x"])

    result = await get_allowed_skills(key_auth, prisma_client=object())

    assert result == frozenset()


@pytest.mark.asyncio
async def test_no_permissions_anywhere_returns_empty_set(monkeypatch):
    """Nobody (key/team/org) has any object_permission at all.

    Mirrors MCPRequestHandler.get_allowed_mcp_servers's fallback: with zero
    restrictions defined at any level, the function returns an empty
    collection rather than "everything" - callers (get_marketplace) treat an
    empty allowed_skills result as "nothing extra beyond the public catalog",
    which is the correct, safe default.
    """
    key_auth = _key_auth(allowed_skills=None, team_id=None, org_id=None)

    result = await get_allowed_skills(key_auth, prisma_client=object())

    assert result == frozenset()


@pytest.mark.asyncio
async def test_team_grants_with_no_key_restriction_becomes_the_effective_set(monkeypatch):
    key_auth = _key_auth(allowed_skills=None, team_id="team-1")
    _mock_team(monkeypatch, allowed_skills=["a--x"])

    result = await get_allowed_skills(key_auth, prisma_client=object())

    assert result == frozenset({"a--x"})


@pytest.mark.asyncio
async def test_key_and_team_grants_intersect(monkeypatch):
    key_auth = _key_auth(allowed_skills=["a--x", "a--z"], team_id="team-1")
    _mock_team(monkeypatch, allowed_skills=["a--x", "a--y"])

    result = await get_allowed_skills(key_auth, prisma_client=object())

    assert result == frozenset({"a--x"})


@pytest.mark.asyncio
async def test_org_ceiling_applies_on_top_of_key_team_intersection(monkeypatch):
    """Full chain: key/team intersect first, then org caps the result further."""
    key_auth = _key_auth(allowed_skills=["a--x", "a--y"], team_id="team-1", org_id="org-1")
    _mock_team(monkeypatch, allowed_skills=["a--x", "a--y", "a--z"])
    _mock_org(monkeypatch, allowed_skills=["a--x"])

    result = await get_allowed_skills(key_auth, prisma_client=object())

    assert result == frozenset({"a--x"})
