"""Tests for the xct-native skills CRUD (S2-03).

Covers happy paths and the ACL branches per CLAUDE.md:
1. owner+admin can write
2. non-owner non-admin gets 403 on write
3. anthropic-source rows are NOT visible via /v1/xct-skills
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.skill_endpoints.endpoints import router


def _make_client(role=LitellmUserRoles.PROXY_ADMIN, user_id="u-1", team_id="t-1"):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        api_key="sk-x", user_id=user_id, team_id=team_id, user_role=role
    )
    return TestClient(app)


def _mock_row(**overrides):
    base = dict(
        skill_id="sk-001",
        display_title="Hello",
        description=None,
        instructions=None,
        system_prompt_template=None,
        tool_schema=None,
        source="custom",
        version="1",
        is_public=False,
        team_id="t-1",
        user_id="u-1",
        created_by="u-1",
        created_at=datetime(2026, 5, 19, 5, 0, 0),
        updated_at=datetime(2026, 5, 19, 5, 0, 0),
        xct_metadata={},
    )
    base.update(overrides)
    m = MagicMock()
    for k, v in base.items():
        setattr(m, k, v)
    m.model_dump = MagicMock(return_value=base)
    return m


def _mock_prisma():
    prisma = MagicMock()
    prisma.db.litellm_skillstable.create = AsyncMock()
    prisma.db.litellm_skillstable.find_unique = AsyncMock()
    prisma.db.litellm_skillstable.find_many = AsyncMock()
    prisma.db.litellm_skillstable.update = AsyncMock()
    prisma.db.litellm_skillstable.delete = AsyncMock()
    return prisma


def test_create_skill_persists_caller_identity():
    prisma = _mock_prisma()
    prisma.db.litellm_skillstable.create.return_value = _mock_row(
        skill_id="new-id", display_title="My Skill"
    )
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = _make_client(role=LitellmUserRoles.INTERNAL_USER).post(
            "/v1/xct-skills",
            json={"display_title": "My Skill"},
            headers={"Authorization": "Bearer k"},
        )
    assert resp.status_code == 200
    create_kwargs = prisma.db.litellm_skillstable.create.call_args.kwargs["data"]
    assert create_kwargs["source"] == "custom"
    assert create_kwargs["user_id"] == "u-1"
    assert create_kwargs["team_id"] == "t-1"
    assert create_kwargs["created_by"] == "u-1"


def test_list_skills_scopes_to_caller_for_non_admin():
    prisma = _mock_prisma()
    prisma.db.litellm_skillstable.find_many.return_value = [
        _mock_row(skill_id="sk-1", user_id="u-1"),
    ]
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = _make_client(role=LitellmUserRoles.INTERNAL_USER).get(
            "/v1/xct-skills", headers={"Authorization": "Bearer k"}
        )
    assert resp.status_code == 200
    args = prisma.db.litellm_skillstable.find_many.call_args.kwargs
    # Where clause must AND the source filter with a caller-clauses OR.
    where = args["where"]
    assert "AND" in where
    inner_or = next(c for c in where["AND"] if "OR" in c)["OR"]
    # contains exactly is_public + user + team clauses
    keys = {next(iter(d)) for d in inner_or}
    assert keys == {"is_public", "user_id", "team_id"}


def test_get_skill_404_when_source_is_anthropic():
    """Anthropic-source rows MUST NOT leak through /v1/xct-skills."""
    prisma = _mock_prisma()
    prisma.db.litellm_skillstable.find_unique.return_value = _mock_row(
        skill_id="sk-anthropic", source="anthropic"
    )
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = _make_client(role=LitellmUserRoles.PROXY_ADMIN).get(
            "/v1/xct-skills/sk-anthropic", headers={"Authorization": "Bearer k"}
        )
    assert resp.status_code == 404


def test_patch_skill_403_for_non_owner_non_admin():
    prisma = _mock_prisma()
    prisma.db.litellm_skillstable.find_unique.return_value = _mock_row(
        skill_id="sk-001", user_id="OTHER_USER"
    )
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = _make_client(role=LitellmUserRoles.INTERNAL_USER).patch(
            "/v1/xct-skills/sk-001",
            json={"display_title": "evil"},
            headers={"Authorization": "Bearer k"},
        )
    assert resp.status_code == 403
    prisma.db.litellm_skillstable.update.assert_not_awaited()


def test_patch_skill_owner_succeeds():
    prisma = _mock_prisma()
    prisma.db.litellm_skillstable.find_unique.return_value = _mock_row(
        skill_id="sk-001", user_id="u-1"
    )
    prisma.db.litellm_skillstable.update.return_value = _mock_row(
        skill_id="sk-001", display_title="renamed", user_id="u-1"
    )
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = _make_client(role=LitellmUserRoles.INTERNAL_USER).patch(
            "/v1/xct-skills/sk-001",
            json={"display_title": "renamed"},
            headers={"Authorization": "Bearer k"},
        )
    assert resp.status_code == 200
    assert resp.json()["display_title"] == "renamed"


def test_delete_skill_admin_succeeds():
    prisma = _mock_prisma()
    prisma.db.litellm_skillstable.find_unique.return_value = _mock_row(
        skill_id="sk-001"
    )
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = _make_client(role=LitellmUserRoles.PROXY_ADMIN).delete(
            "/v1/xct-skills/sk-001", headers={"Authorization": "Bearer k"}
        )
    assert resp.status_code == 200
    assert resp.json() == {"skill_id": "sk-001", "deleted": True}
    prisma.db.litellm_skillstable.delete.assert_awaited_once()


# ============================================================================
# S2-06: publish semantics
# ============================================================================


def test_publish_sets_metadata_flags():
    prisma = _mock_prisma()
    prisma.db.litellm_skillstable.find_unique.return_value = _mock_row(
        skill_id="sk-001", user_id="u-1"
    )
    prisma.db.litellm_skillstable.update.return_value = _mock_row(
        skill_id="sk-001", xct_metadata={"published": True}
    )
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = _make_client(role=LitellmUserRoles.PROXY_ADMIN).post(
            "/v1/xct-skills/sk-001/publish",
            headers={"Authorization": "Bearer k"},
        )
    assert resp.status_code == 200
    update_data = prisma.db.litellm_skillstable.update.call_args.kwargs["data"]
    # xct_metadata reaches Prisma wrapped in prisma.Json (raw dicts are
    # rejected by Json columns); unwrap via .data to assert on the payload.
    meta = update_data["xct_metadata"].data
    assert meta["published"] is True
    assert "published_at" in meta and meta["published_by"] == "u-1"


def test_patch_published_skill_409s_on_content_field():
    """After publish, tool_schema/system_prompt_template/instructions are frozen."""
    prisma = _mock_prisma()
    prisma.db.litellm_skillstable.find_unique.return_value = _mock_row(
        skill_id="sk-001", user_id="u-1", xct_metadata={"published": True}
    )
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = _make_client(role=LitellmUserRoles.PROXY_ADMIN).patch(
            "/v1/xct-skills/sk-001",
            json={"system_prompt_template": "new!"},
            headers={"Authorization": "Bearer k"},
        )
    assert resp.status_code == 409
    prisma.db.litellm_skillstable.update.assert_not_awaited()


def test_patch_published_skill_allows_safe_field():
    """display_title and description stay mutable even after publish."""
    prisma = _mock_prisma()
    prisma.db.litellm_skillstable.find_unique.return_value = _mock_row(
        skill_id="sk-001", user_id="u-1", xct_metadata={"published": True}
    )
    prisma.db.litellm_skillstable.update.return_value = _mock_row(
        skill_id="sk-001", display_title="New Title", xct_metadata={"published": True}
    )
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = _make_client(role=LitellmUserRoles.PROXY_ADMIN).patch(
            "/v1/xct-skills/sk-001",
            json={"display_title": "New Title"},
            headers={"Authorization": "Bearer k"},
        )
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# _prisma_json_compat — Prisma Json columns reject plain dicts and explicit
# None; values must be wrapped in prisma.Json and absent-when-None.
# (Regression: every real-DB create returned 500 before this existed.)
# ---------------------------------------------------------------------------


def test_prisma_json_compat_wraps_json_columns_and_drops_none():
    prisma = pytest.importorskip("prisma")
    from litellm.proxy.skill_endpoints.endpoints import _prisma_json_compat

    data = _prisma_json_compat(
        {
            "display_title": "t",
            "tool_schema": None,
            "xct_metadata": {"origin": "x"},
            "metadata": [{"a": 1}],
        }
    )
    assert "tool_schema" not in data  # explicit None must be omitted
    assert isinstance(data["xct_metadata"], prisma.Json)
    assert isinstance(data["metadata"], prisma.Json)
    assert data["display_title"] == "t"  # non-Json fields untouched
